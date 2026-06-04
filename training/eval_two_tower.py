"""
Two-Tower Evaluation
====================
Loads the best checkpoint and evaluates Recall@K, MAP@K, and NDCG@K on the
held-out test set. Every run is logged to MLflow under the "two-tower-eval"
experiment.

Run:
    python training/eval_two_tower.py
    python training/eval_two_tower.py --k 5 10 20 50
    python training/eval_two_tower.py --checkpoint training/checkpoints/epoch_005.pt
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlflow
import numpy as np
import pandas as pd
import torch

from models.two_tower import TwoTowerModel

_ROOT       = Path(__file__).parent.parent
_CKPT_PATH  = _ROOT / "training" / "checkpoints" / "best.pt"
_TRAIN_PATH = _ROOT / "data" / "processed" / "train.parquet"
_TEST_PATH  = _ROOT / "data" / "processed" / "test.parquet"
_MLFLOW_URI = "sqlite:///" + str(_ROOT / "mlflow.db")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate the two-tower model.")
    p.add_argument("--k", nargs="+", type=int, default=[10, 20, 50])
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--checkpoint", type=str, default=str(_CKPT_PATH))
    p.add_argument("--no-exclude-train", dest="exclude_train", action="store_false", default=True,
                   help="Don't mask training items from the ranked list.")
    return p.parse_args()


def _infer_model_config(state: dict) -> dict:
    num_users     = state["user_tower.embedding.weight"].shape[0] - 1
    num_items     = state["item_tower.embedding.weight"].shape[0] - 1
    embedding_dim = state["user_tower.embedding.weight"].shape[1]

    linear_keys = sorted(
        (k for k in state if k.startswith("user_tower.mlp.") and k.endswith(".weight") and state[k].dim() == 2),
        key=lambda k: int(k.split(".")[2]),
    )
    shapes      = [state[k].shape for k in linear_keys]
    hidden_dims = [s[0] for s in shapes[:-1]]
    output_dim  = shapes[-1][0]

    return dict(
        num_users=num_users, num_items=num_items,
        embedding_dim=embedding_dim, hidden_dims=hidden_dims, output_dim=output_dim,
    )


def _encode_all_items(model: TwoTowerModel, num_items: int, device: torch.device, batch_size: int) -> np.ndarray:
    all_ids = torch.arange(1, num_items + 1, dtype=torch.long)
    vecs = []
    with torch.no_grad():
        for start in range(0, num_items, batch_size):
            batch = all_ids[start:start + batch_size].to(device)
            vecs.append(model.encode_items(batch).cpu().numpy())
    return np.concatenate(vecs, axis=0)  # (num_items, output_dim)


def _recall_at_k(ranked: np.ndarray, relevant: set, k: int) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant)


def _ap_at_k(ranked: np.ndarray, relevant: set, k: int) -> float:
    hits, score = 0, 0.0
    for i, item in enumerate(ranked[:k], start=1):
        if item in relevant:
            hits += 1
            score += hits / i
    return score / min(k, len(relevant))


def _ndcg_at_k(ranked: np.ndarray, relevant: set, k: int) -> float:
    dcg  = sum(1.0 / np.log2(i + 1) for i, item in enumerate(ranked[:k], start=1) if item in relevant)
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, min(k, len(relevant)) + 1))
    return dcg / idcg if idcg > 0 else 0.0


def main() -> None:
    args   = _parse_args()
    ks     = sorted(args.k)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load checkpoint ---
    ckpt_path = Path(args.checkpoint)
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt      = torch.load(ckpt_path, map_location=device, weights_only=False)
    user_enc: dict[str, int] = ckpt["user_enc"]
    item_enc: dict[str, int] = ckpt["item_enc"]

    # --- Reconstruct model ---
    cfg   = _infer_model_config(ckpt["model_state"])
    model = TwoTowerModel(**cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  users={cfg['num_users']:,}  items={cfg['num_items']:,}  output_dim={cfg['output_dim']}")

    # --- Load interactions ---
    print("Loading interactions...")
    train_df = pd.read_parquet(_TRAIN_PATH, columns=["user_id", "item_id"])
    test_df  = pd.read_parquet(_TEST_PATH,  columns=["user_id", "item_id"])

    # Drop cold-start users/items not seen during training
    test_df = test_df[test_df["user_id"].isin(user_enc) & test_df["item_id"].isin(item_enc)]

    # Ground truth: user_idx → set of positive item_idxs
    ground_truth: dict[int, set[int]] = defaultdict(set)
    for row in test_df.itertuples(index=False):
        ground_truth[user_enc[row.user_id]].add(item_enc[row.item_id])

    # Items seen in training (masked from ranking so model can't trivially retrieve them)
    train_positives: dict[int, set[int]] = defaultdict(set)
    if args.exclude_train:
        seen = train_df[train_df["user_id"].isin(user_enc) & train_df["item_id"].isin(item_enc)]
        for row in seen.itertuples(index=False):
            train_positives[user_enc[row.user_id]].add(item_enc[row.item_id])

    print(f"  {len(ground_truth):,} test users")

    # --- Pre-compute all item embeddings ---
    print("Encoding all items...")
    item_vecs = _encode_all_items(model, cfg["num_items"], device, args.batch_size)
    # item_vecs[j-1] is the embedding for item with 1-based index j

    # --- Evaluate ---
    print(f"Evaluating k={ks}...")
    metric_accum: dict[str, list[float]] = defaultdict(list)
    user_id_buf = torch.zeros(1, dtype=torch.long, device=device)

    for user_idx, relevant in ground_truth.items():
        user_id_buf[0] = user_idx
        with torch.no_grad():
            user_vec = model.encode_users(user_id_buf).cpu().numpy()[0]

        sims = item_vecs @ user_vec  # cosine sim — items are L2-normalised

        if args.exclude_train:
            for seen_idx in train_positives.get(user_idx, set()):
                sims[seen_idx - 1] = -np.inf

        ranked = np.argsort(-sims) + 1  # descending; 1-based item indices

        for k in ks:
            metric_accum[f"recall_at_{k}"].append(_recall_at_k(ranked, relevant, k))
            metric_accum[f"map_at_{k}"].append(_ap_at_k(ranked, relevant, k))
            metric_accum[f"ndcg_at_{k}"].append(_ndcg_at_k(ranked, relevant, k))

    mean_metrics = {name: float(np.mean(vals)) for name, vals in metric_accum.items()}

    print("\n=== Results ===")
    for k in ks:
        print(
            f"  k={k:>3} | "
            f"Recall={mean_metrics[f'recall_at_{k}']:.4f} | "
            f"MAP={mean_metrics[f'map_at_{k}']:.4f} | "
            f"NDCG={mean_metrics[f'ndcg_at_{k}']:.4f}"
        )

    # --- Log to MLflow ---
    mlflow.set_tracking_uri(_MLFLOW_URI)
    mlflow.set_experiment("two-tower-eval")

    with mlflow.start_run():
        mlflow.log_params({
            "checkpoint"     : ckpt_path.name,
            "epoch"          : ckpt.get("epoch", "unknown"),
            "val_loss"       : ckpt.get("val_loss"),
            "k_values"       : str(ks),
            "exclude_train"  : args.exclude_train,
            "num_test_users" : len(ground_truth),
            "embedding_dim"  : cfg["embedding_dim"],
            "hidden_dims"    : str(cfg["hidden_dims"]),
            "output_dim"     : cfg["output_dim"],
        })
        mlflow.log_metrics(mean_metrics)

    print("\nMetrics logged to MLflow.")


if __name__ == "__main__":
    main()
