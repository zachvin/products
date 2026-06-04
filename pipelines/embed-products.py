"""
Product Embedding Pipeline
==========================
Loads product metadata, encodes each product's title + features + description
with a sentence-transformer model, builds a FAISS index, and writes:

    ./feature_store/product_faiss.index        - FAISS IndexFlatIP (cosine sim)
    ./feature_store/product_index_map.parquet  - maps FAISS row int → parent_asin

Run:
    python embed-products.py
"""

import os
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

METADATA_PATH   = "./data/processed/metadata.parquet"
FEATURE_STORE   = "./feature_store"
INDEX_PATH      = os.path.join(FEATURE_STORE, "product_faiss.index")
INDEX_MAP_PATH  = os.path.join(FEATURE_STORE, "product_index_map.parquet")

MODEL_NAME  = "all-MiniLM-L6-v2"
BATCH_SIZE  = 512

os.makedirs(FEATURE_STORE, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load metadata
# ---------------------------------------------------------------------------

print("Loading metadata...")
meta = pd.read_parquet(METADATA_PATH, columns=["parent_asin", "title", "features", "description"])
meta = meta.drop_duplicates(subset="parent_asin").reset_index(drop=True)
print(f"  {len(meta):,} unique products")

# ---------------------------------------------------------------------------
# 2. Build input text
# ---------------------------------------------------------------------------

def build_text(row: pd.Series) -> str:
    parts = [row["title"] or "", row["features"] or "", row["description"] or ""]
    return ". ".join(p.strip() for p in parts if p.strip())

print("Building input text...")
texts = meta.apply(build_text, axis=1).tolist()

# ---------------------------------------------------------------------------
# 3. Encode with sentence-transformers
# ---------------------------------------------------------------------------

print(f"Loading model '{MODEL_NAME}'...")
model = SentenceTransformer(MODEL_NAME)

print(f"Encoding {len(texts):,} products (batch_size={BATCH_SIZE})...")
embeddings = model.encode(
    texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,  # unit-norm → inner product = cosine similarity
)
embeddings = embeddings.astype(np.float32)
print(f"  Embedding matrix: {embeddings.shape}")

# ---------------------------------------------------------------------------
# 4. Build FAISS index
# ---------------------------------------------------------------------------

dim = embeddings.shape[1]
print(f"Building FAISS IndexFlatIP (dim={dim})...")
index = faiss.IndexFlatIP(dim)
index.add(embeddings)
print(f"  Index contains {index.ntotal:,} vectors")

# ---------------------------------------------------------------------------
# 5. Write outputs
# ---------------------------------------------------------------------------

print(f"Saving FAISS index to {INDEX_PATH} ...")
faiss.write_index(index, INDEX_PATH)

print(f"Saving product index map to {INDEX_MAP_PATH} ...")
index_map = pd.DataFrame({
    "faiss_idx":   np.arange(len(meta), dtype=np.int64),
    "parent_asin": meta["parent_asin"].values,
})
index_map.to_parquet(INDEX_MAP_PATH, index=False)

print(f"\nDone.")
print(f"  Vectors : {index.ntotal:,}")
print(f"  Index   : {INDEX_PATH}")
print(f"  Map     : {INDEX_MAP_PATH}")
