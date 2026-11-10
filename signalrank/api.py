"""FastAPI serving layer with lazy model loading."""

from __future__ import annotations

import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Query

from .explain import explain
from .model import HybridRecommender


app = FastAPI(title="SignalRank API", version="0.1.0")


@lru_cache(maxsize=1)
def get_model() -> HybridRecommender:
    path = os.environ.get("SIGNALRANK_MODEL", "artifacts/model.joblib")
    return HybridRecommender.load(path)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/recommend/{user_id}")
def recommend(user_id: str, k: int = Query(10, ge=1, le=50)) -> dict[str, object]:
    model = get_model()
    recommendations = model.recommend(user_id, k=k)
    return {
        "user_id": user_id,
        "model": "hybrid-tfidf-svd-logistic-ranker",
        "recommendations": [
            {
                "item_id": rec.item_id,
                "score": rec.score,
                "rank": rec.rank,
                "reasons": rec.reasons,
            }
            for rec in recommendations
        ],
    }


@app.get("/explain/{user_id}/{item_id}")
def explain_item(user_id: str, item_id: str) -> dict[str, object]:
    model = get_model()
    matches = [rec for rec in model.recommend(user_id, k=50) if rec.item_id == item_id]
    if not matches:
        raise HTTPException(status_code=404, detail="item is not in the top-50 recommendations")
    return explain(model, user_id, matches[0])

