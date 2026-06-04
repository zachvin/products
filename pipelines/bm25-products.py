"""
Product BM25 Index Builder
==========================
Builds a BM25 index over product text (title + features + description) in the
same row order as the FAISS embedding index, so that a result position from
either index maps to the same parent_asin via product_index_map.parquet.

Writes:
    ./feature_store/product_bm25.pkl  - pickled BM25Okapi object

Run:
    python bm25-products.py
"""

import os
import pickle
import pandas as pd
from rank_bm25 import BM25Okapi

METADATA_PATH  = "./data/processed/metadata.parquet"
INDEX_MAP_PATH = "./feature_store/product_index_map.parquet"
BM25_PATH      = "./feature_store/product_bm25.pkl"

# ---------------------------------------------------------------------------
# 1. Load the index map to get parent_asins in FAISS order
# ---------------------------------------------------------------------------

print("Loading product index map...")
index_map = pd.read_parquet(INDEX_MAP_PATH, columns=["faiss_idx", "parent_asin"])
index_map = index_map.sort_values("faiss_idx").reset_index(drop=True)
print(f"  {len(index_map):,} products")

# ---------------------------------------------------------------------------
# 2. Load metadata and align to FAISS order
# ---------------------------------------------------------------------------

print("Loading metadata...")
meta = pd.read_parquet(
    METADATA_PATH,
    columns=["parent_asin", "title", "features", "description"],
)

ordered = index_map.merge(meta, on="parent_asin", how="left")
if len(ordered) != len(index_map):
    raise ValueError(
        f"Row count mismatch after merge: {len(ordered)} != {len(index_map)}. "
        "Re-run the feature pipeline and embedding pipeline before building BM25."
    )

# ---------------------------------------------------------------------------
# 3. Build tokenised corpus in FAISS row order
# ---------------------------------------------------------------------------

def build_text(row: pd.Series) -> str:
    parts = [row["title"] or "", row["features"] or "", row["description"] or ""]
    return ". ".join(p.strip() for p in parts if p.strip())

print("Building corpus...")
corpus = [build_text(row).lower().split() for _, row in ordered.iterrows()]

# ---------------------------------------------------------------------------
# 4. Fit BM25
# ---------------------------------------------------------------------------

print("Fitting BM25Okapi...")
bm25 = BM25Okapi(corpus)

# ---------------------------------------------------------------------------
# 5. Save
# ---------------------------------------------------------------------------

print(f"Saving BM25 index to {BM25_PATH} ...")
with open(BM25_PATH, "wb") as f:
    pickle.dump(bm25, f)

print(f"\nDone. Index covers {len(corpus):,} documents.")
