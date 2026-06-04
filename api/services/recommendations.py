"""
Recommendation service.

Startup sequence (runs once on first request):
  1. Load the best two-tower checkpoint — model weights + user/item encoders.
  2. Encode every item with the item tower and build an in-memory FAISS index.
  3. Read user_features and item_features from the Delta Lake parquet directories.
  4. Load LinUCB from disk if a saved model exists.

Request sequence for get_recommendations:
  1. Encode the user with the user tower.
  2. Retrieve the top fetch_k candidates from the FAISS index.
  3. Pull user + item feature vectors for those candidates.
  4. Re-rank with LinUCB (UCB score). Falls back to cosine-sim order if no
     LinUCB model has been saved yet.
  5. Return the top-k results.
"""

import threading
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch

from models.linucb import LinUCB
from models.two_tower import TwoTowerModel

_ROOT          = Path(__file__).parent.parent.parent
_CKPT_PATH     = _ROOT / "training" / "checkpoints" / "best.pt"
_FEATURE_STORE = _ROOT / "feature_store"
_USER_FEAT_DIR = _FEATURE_STORE / "user_features"
_ITEM_FEAT_DIR = _FEATURE_STORE / "item_features"
_LINUCB_PATH   = _FEATURE_STORE / "linucb.pkl"

# Numerical columns used to build the LinUCB context vector.
# Categorical columns (favorite_category, main_category) and raw timestamps
# are intentionally excluded.
_USER_FEAT_COLS: list[str] = [
    "total_purchases", "avg_rating", "rating_stddev", "high_rating_count",
    "low_rating_count", "days_since_last_purchase", "days_active",
    "purchase_frequency", "distinct_categories", "avg_item_price",
    "max_item_price", "total_sessions", "avg_session_length",
    "max_session_length", "avg_session_duration_mins", "max_session_duration_mins",
]
_ITEM_FEAT_COLS: list[str] = [
    "total_purchases", "unique_buyers", "avg_rating", "rating_stddev",
    "rating_count", "high_rating_count", "days_since_last_purchased", "price_numeric",
]
_LINUCB_DIM = len(_USER_FEAT_COLS) + len(_ITEM_FEAT_COLS)  # 24

_FETCH_FACTOR = 5   # retrieve k × FETCH_FACTOR candidates before re-ranking
_ENCODE_BATCH = 512

# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

_lock      = threading.Lock()
_loaded    = False
_model:     TwoTowerModel | None = None
_device:    torch.device   | None = None
_user_enc:  dict[str, int] | None = None   # user_id → int index
_item_dec:  dict[int, str] | None = None   # int index → parent_asin
_faiss_idx: faiss.Index    | None = None
_linucb:    LinUCB         | None = None
_user_feat: pd.DataFrame   | None = None   # indexed by user_id
_item_feat: pd.DataFrame   | None = None   # indexed by item_id (= parent_asin)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def load() -> None:
    """Load all models and feature tables into memory. Safe to call repeatedly."""
    global _loaded, _model, _device, _user_enc, _item_dec
    global _faiss_idx, _linucb, _user_feat, _item_feat

    with _lock:
        if _loaded:
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Two-tower checkpoint ---
        print(f"Loading checkpoint: {_CKPT_PATH}")
        ckpt      = torch.load(_CKPT_PATH, map_location=device, weights_only=False)
        user_enc  = ckpt["user_enc"]
        item_enc  = ckpt["item_enc"]
        item_dec  = {v: k for k, v in item_enc.items()}
        state     = ckpt["model_state"]

        # Reconstruct model architecture from state dict weights
        num_users     = state["user_tower.embedding.weight"].shape[0] - 1
        num_items     = state["item_tower.embedding.weight"].shape[0] - 1
        embedding_dim = state["user_tower.embedding.weight"].shape[1]
        linear_keys   = sorted(
            (k for k in state if k.startswith("user_tower.mlp.") and k.endswith(".weight") and state[k].dim() == 2),
            key=lambda k: int(k.split(".")[2]),
        )
        shapes      = [state[k].shape for k in linear_keys]
        hidden_dims = [s[0] for s in shapes[:-1]]
        output_dim  = shapes[-1][0]

        model = TwoTowerModel(
            num_users=num_users, num_items=num_items,
            embedding_dim=embedding_dim, hidden_dims=hidden_dims,
            output_dim=output_dim,
        ).to(device)
        model.load_state_dict(state)
        model.eval()
        print(f"  {num_users:,} users  {num_items:,} items  dim={output_dim}")

        # --- Build FAISS index over two-tower item embeddings ---
        # FAISS position p → item integer index p+1 → parent_asin via item_dec
        print("Building item FAISS index...")
        all_ids = torch.arange(1, num_items + 1, dtype=torch.long)
        vecs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, num_items, _ENCODE_BATCH):
                batch = all_ids[start : start + _ENCODE_BATCH].to(device)
                vecs.append(model.encode_items(batch).cpu().numpy())
        item_vecs = np.concatenate(vecs, axis=0).astype(np.float32)
        faiss_idx = faiss.IndexFlatIP(output_dim)
        faiss_idx.add(item_vecs)
        print(f"  {faiss_idx.ntotal:,} vectors indexed")

        # --- Feature tables (read parquet directories directly) ---
        print("Loading feature tables...")
        user_feat = pd.read_parquet(_USER_FEAT_DIR).set_index("user_id")
        item_feat = pd.read_parquet(_ITEM_FEAT_DIR).set_index("item_id")

        # --- LinUCB (optional) ---
        linucb = LinUCB.load(_LINUCB_PATH) if _LINUCB_PATH.exists() else None
        if linucb:
            print("Loaded LinUCB model.")
        else:
            print("No LinUCB model found — will rank by two-tower cosine similarity.")

        _model     = model
        _device    = device
        _user_enc  = user_enc
        _item_dec  = item_dec
        _faiss_idx = faiss_idx
        _linucb    = linucb
        _user_feat = user_feat
        _item_feat = item_feat
        _loaded    = True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_recommendations(user_id: str, k: int = 10) -> list[dict]:
    """
    Return the top-k recommended items for a user.

    Raises:
        KeyError: if user_id was not seen during training.
    """
    load()

    if user_id not in _user_enc:
        raise KeyError(user_id)

    fetch_k = min(k * _FETCH_FACTOR, _faiss_idx.ntotal)

    # Encode user
    uid_t = torch.tensor([_user_enc[user_id]], dtype=torch.long, device=_device)
    with torch.no_grad():
        user_vec = _model.encode_users(uid_t).cpu().numpy().astype(np.float32)

    # Retrieve candidates from FAISS
    _, positions = _faiss_idx.search(user_vec, fetch_k)
    candidate_asins = [
        _item_dec[p + 1]
        for p in positions[0].tolist()
        if (p + 1) in _item_dec
    ]

    # Re-rank with LinUCB when available
    if _linucb is not None and user_id in _user_feat.index:
        user_row  = _user_feat.loc[user_id, _USER_FEAT_COLS].fillna(0.0).to_numpy(np.float64)
        item_rows = (
            _item_feat.reindex(candidate_asins)[_ITEM_FEAT_COLS]
            .fillna(0.0)
            .to_numpy(np.float64)
        )
        order = _linucb.rank(user_row, item_rows)
        candidate_asins = [candidate_asins[i] for i in order]

    return [
        {"parent_asin": asin, "rank": i + 1}
        for i, asin in enumerate(candidate_asins[:k])
    ]


def record_feedback(user_id: str, parent_asin: str, reward: float) -> None:
    """
    Update the LinUCB model with an observed reward and persist it to disk.

    Initialises a fresh LinUCB model if none has been saved yet.

    Raises:
        KeyError: if user_id or parent_asin is not in the feature store.
    """
    load()

    if user_id not in _user_feat.index:
        raise KeyError(f"Unknown user_id: {user_id}")
    if parent_asin not in _item_feat.index:
        raise KeyError(f"Unknown parent_asin: {parent_asin}")

    user_row = _user_feat.loc[user_id, _USER_FEAT_COLS].fillna(0.0).to_numpy(np.float64)
    item_row = _item_feat.loc[parent_asin, _ITEM_FEAT_COLS].fillna(0.0).to_numpy(np.float64)

    with _lock:
        global _linucb
        if _linucb is None:
            _linucb = LinUCB(context_dim=_LINUCB_DIM, alpha=1.0)
        _linucb.update(user_row, item_row, reward)
        _linucb.save(_LINUCB_PATH)
