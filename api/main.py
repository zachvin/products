"""
Product Recommender API
=======================
Run:
    poetry run uvicorn api.main:app --reload

Endpoints:
    GET  /search                          Hybrid BM25 + FAISS text search
    GET  /recommendations/{user_id}       Personalised recommendations
    POST /recommendations/feedback        Record a reward to update LinUCB
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from api.services import recommendations, search


@asynccontextmanager
async def lifespan(app: FastAPI):
    recommendations.load()
    yield


app = FastAPI(title="Product Recommender API", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

@app.get("/search")
def search_endpoint(
    q: str,
    k: int = Query(default=10, ge=1, le=100),
):
    """Hybrid BM25 + semantic search over the product catalogue."""
    return {"results": search.hybrid_search(q, k=k)}


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------

@app.get("/recommendations/{user_id}")
def recommend_endpoint(
    user_id: str,
    k: int = Query(default=10, ge=1, le=100),
):
    """Personalised item recommendations for a known user."""
    try:
        results = recommendations.get_recommendations(user_id, k=k)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown user '{user_id}'")
    return {"user_id": user_id, "results": results}


class FeedbackBody(BaseModel):
    user_id: str
    parent_asin: str
    reward: float


@app.post("/recommendations/feedback")
def feedback_endpoint(body: FeedbackBody):
    """Record an observed reward (e.g. click=1, skip=0) to update the LinUCB model."""
    try:
        recommendations.record_feedback(body.user_id, body.parent_asin, body.reward)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"ok": True}
