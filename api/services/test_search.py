import importlib
import pickle
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import api.services.search as search_mod

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_PRODUCTS = 50
DIM = 8  # tiny embedding dimension for speed


def _make_embeddings(n: int = N_PRODUCTS, dim: int = DIM) -> np.ndarray:
    rng = np.random.default_rng(0)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def _make_index_map(n: int = N_PRODUCTS) -> pd.DataFrame:
    return pd.DataFrame(
        {"faiss_idx": np.arange(n), "parent_asin": [f"ASIN{i:04d}" for i in range(n)]}
    )


def _make_faiss_index(embeddings: np.ndarray):
    import faiss

    idx = faiss.IndexFlatIP(embeddings.shape[1])
    idx.add(embeddings)
    return idx


def _make_bm25(n: int = N_PRODUCTS):
    from rank_bm25 import BM25Okapi

    corpus = [
        f"product title {i} feature description text".split() for i in range(n)
    ]
    return BM25Okapi(corpus)


def _make_sentence_transformer(dim: int = DIM):
    """Stub that returns a deterministic unit-norm vector for any input."""
    rng = np.random.default_rng(99)
    vec = rng.standard_normal((1, dim)).astype(np.float32)
    vec /= np.linalg.norm(vec)

    mock = MagicMock()
    mock.encode.return_value = vec
    return mock


@pytest.fixture(autouse=True)
def reset_globals():
    """Reset module-level singletons between tests so _load() runs fresh."""
    search_mod._model     = None
    search_mod._faiss_idx = None
    search_mod._bm25      = None
    search_mod._asin_map  = None
    yield
    search_mod._model     = None
    search_mod._faiss_idx = None
    search_mod._bm25      = None
    search_mod._asin_map  = None


@pytest.fixture()
def loaded_module():
    """Inject fake indexes into the module so hybrid_search can run."""
    embeddings = _make_embeddings()
    index_map  = _make_index_map()

    search_mod._model     = _make_sentence_transformer()
    search_mod._faiss_idx = _make_faiss_index(embeddings)
    search_mod._bm25      = _make_bm25()
    search_mod._asin_map  = index_map.sort_values("faiss_idx")["parent_asin"].to_numpy()
    return search_mod


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHybridSearchResults:
    def test_returns_k_results(self, loaded_module):
        results = loaded_module.hybrid_search("phone case", k=5)
        assert len(results) == 5

    def test_default_k_is_ten(self, loaded_module):
        results = loaded_module.hybrid_search("wireless charger")
        assert len(results) == 10

    def test_result_keys(self, loaded_module):
        results = loaded_module.hybrid_search("screen protector", k=3)
        for r in results:
            assert set(r.keys()) == {"parent_asin", "score"}

    def test_asins_are_strings(self, loaded_module):
        results = loaded_module.hybrid_search("cable", k=5)
        for r in results:
            assert isinstance(r["parent_asin"], str)

    def test_scores_are_floats_in_unit_interval(self, loaded_module):
        results = loaded_module.hybrid_search("bluetooth headphones", k=10)
        for r in results:
            assert isinstance(r["score"], float)
            assert 0.0 <= r["score"] <= 1.0

    def test_results_are_sorted_descending(self, loaded_module):
        results = loaded_module.hybrid_search("earbuds", k=10)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_no_duplicate_asins(self, loaded_module):
        results = loaded_module.hybrid_search("USB hub", k=10)
        asins = [r["parent_asin"] for r in results]
        assert len(asins) == len(set(asins))

    def test_asins_from_known_set(self, loaded_module):
        valid_asins = {f"ASIN{i:04d}" for i in range(N_PRODUCTS)}
        results = loaded_module.hybrid_search("tripod", k=5)
        for r in results:
            assert r["parent_asin"] in valid_asins

    def test_k_larger_than_corpus_clamped(self, loaded_module):
        results = loaded_module.hybrid_search("phone", k=N_PRODUCTS + 100)
        assert len(results) <= N_PRODUCTS

    def test_k_one_returns_single_result(self, loaded_module):
        results = loaded_module.hybrid_search("case", k=1)
        assert len(results) == 1


class TestScoreWeighting:
    def test_faiss_only_query_score_reflects_faiss_weight(self, loaded_module):
        # Force BM25 to return all-zero scores → combined score is purely FAISS weight
        loaded_module._bm25.get_scores = MagicMock(
            return_value=np.zeros(N_PRODUCTS, dtype=np.float32)
        )
        results = loaded_module.hybrid_search("something", k=5)
        top_score = results[0]["score"]
        # Top normalised FAISS score is 1.0; combined = 0.6 * 1.0 + 0.4 * 0.0 = 0.6
        assert pytest.approx(top_score, abs=1e-5) == 0.6

    def test_bm25_only_query_score_reflects_bm25_weight(self, loaded_module):
        # Force FAISS to return zero scores for all candidates
        raw_faiss = loaded_module._faiss_idx.search
        def _zero_search(vec, k):
            scores, ids = raw_faiss(vec, k)
            return np.zeros_like(scores), ids
        loaded_module._faiss_idx.search = _zero_search

        results = loaded_module.hybrid_search("product title 0", k=5)
        top_score = results[0]["score"]
        # Top normalised BM25 score is 1.0; combined = 0.4 * 1.0 + 0.6 * 0.0 = 0.4
        assert pytest.approx(top_score, abs=1e-5) == 0.4


class TestLazyLoading:
    def test_load_called_once_across_multiple_queries(self):
        embeddings = _make_embeddings()
        index_map  = _make_index_map()

        with (
            patch.object(search_mod, "SentenceTransformer", return_value=_make_sentence_transformer()) as mock_st,
            patch("faiss.read_index", return_value=_make_faiss_index(embeddings)),
            patch("builtins.open", MagicMock()),
            patch("pickle.load", return_value=_make_bm25()),
            patch("pandas.read_parquet", return_value=index_map),
        ):
            search_mod.hybrid_search("query one", k=3)
            search_mod.hybrid_search("query two", k=3)

            assert mock_st.call_count == 1
