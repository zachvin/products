"""
Hybrid product search combining BM25 (40%) and FAISS cosine similarity (60%).

Indexes are loaded once on first call and reused for the lifetime of the process.
Both indexes share the same row-order defined by product_index_map.parquet, so
position i in the BM25 corpus and position i in the FAISS index refer to the
same parent_asin.
"""

import pickle
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

_ROOT          = Path(__file__).parent.parent.parent
_FEATURE_STORE = _ROOT / "feature_store"
_INDEX_PATH    = _FEATURE_STORE / "product_faiss.index"
_INDEX_MAP     = _FEATURE_STORE / "product_index_map.parquet"
_BM25_PATH     = _FEATURE_STORE / "product_bm25.pkl"
_MODEL_NAME    = "all-MiniLM-L6-v2"

_BM25_WEIGHT  = 0.4
_FAISS_WEIGHT = 0.6
_FETCH_FACTOR = 10  # retrieve k × FETCH_FACTOR candidates from each index before merging

_model:      SentenceTransformer  | None = None
_faiss_idx:  faiss.Index          | None = None
_bm25:       BM25Okapi            | None = None
_asin_map:   np.ndarray           | None = None  # int position → parent_asin


# Load in sentence transformer, FAISS indices, BM25 embeddings, and index map
def _load() -> None:
    global _model, _faiss_idx, _bm25, _asin_map
    if _model is not None:
        return
    _model     = SentenceTransformer(_MODEL_NAME)
    _faiss_idx = faiss.read_index(str(_INDEX_PATH))
    with open(_BM25_PATH, "rb") as fh:
        _bm25 = pickle.load(fh)

    _asin_map = (
        pd.read_parquet(_INDEX_MAP)
        .sort_values("faiss_idx")["parent_asin"]
        .to_numpy()
    )


# Normalization of scores to [0, 1] range
def _minmax(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - lo) / (hi - lo)


def hybrid_search(query: str, k: int = 10) -> list[dict]:
    """Return the top-k products for *query*, ranked by hybrid BM25 + FAISS score.

    Each result is a dict with keys:
        parent_asin (str)   – product identifier
        score       (float) – combined relevance score in [0, 1]
    """
    _load()

    fetch_k = min(k * _FETCH_FACTOR, _faiss_idx.ntotal)

    # --- FAISS retrieval (cosine similarity via normalised inner product) ---
    query_vec = _model.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)
    faiss_scores_raw, faiss_indices = _faiss_idx.search(query_vec, fetch_k)
    faiss_scores_raw = faiss_scores_raw[0]
    faiss_indices    = faiss_indices[0]

    # --- BM25 retrieval ---
    tokens          = query.lower().split()
    bm25_scores_all = _bm25.get_scores(tokens)
    bm25_indices    = np.argsort(bm25_scores_all)[::-1][:fetch_k]
    bm25_scores_raw = bm25_scores_all[bm25_indices]

    # --- Merge candidate pools ---
    faiss_score_map = dict(zip(faiss_indices.tolist(), faiss_scores_raw.tolist()))
    bm25_score_map  = dict(zip(bm25_indices.tolist(),  bm25_scores_raw.tolist()))
    candidates      = np.array(list(faiss_score_map.keys() | bm25_score_map.keys()))

    faiss_cand = np.array([faiss_score_map.get(int(i), 0.0) for i in candidates])
    bm25_cand  = np.array([bm25_score_map.get(int(i),  0.0) for i in candidates])

    # --- Weighted combination of normalised scores ---
    combined = _BM25_WEIGHT * _minmax(bm25_cand) + _FAISS_WEIGHT * _minmax(faiss_cand)

    # --- Top-k ---
    top_pos = np.argsort(combined)[::-1][:k]
    return [
        {
            "parent_asin": str(_asin_map[candidates[i]]),
            "score":       float(combined[i]),
        }
        for i in top_pos
    ]
