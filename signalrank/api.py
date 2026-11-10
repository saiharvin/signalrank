"""FastAPI serving layer for recommendation, evidence retrieval, and chat."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .chat import RecommendationAssistant
from .explain import explain
from .model import HybridRecommender


app = FastAPI(title="SignalRank API", version="0.2.0")


class ChatRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    question: str = Field(min_length=3, max_length=2000)
    k: int = Field(default=5, ge=1, le=10)


@lru_cache(maxsize=1)
def get_model() -> HybridRecommender:
    path = os.environ.get("SIGNALRANK_MODEL", "artifacts/model.joblib")
    return HybridRecommender.load(path)


@lru_cache(maxsize=1)
def get_assistant() -> RecommendationAssistant:
    return RecommendationAssistant(get_model())


def serialize_recommendation(model: HybridRecommender, recommendation) -> dict[str, object]:
    article = model.article_lookup.loc[recommendation.item_id]
    return {
        "item_id": recommendation.item_id,
        "title": str(article.title),
        "abstract": str(article.abstract),
        "category": str(article.category),
        "subcategory": str(article.subcategory),
        "score": recommendation.score,
        "rank": recommendation.rank,
        "reasons": recommendation.reasons,
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/recommend/{user_id}")
def recommend(user_id: str, k: int = Query(10, ge=1, le=50)) -> dict[str, object]:
    model = get_model()
    recommendations = model.recommend(user_id, k=k)
    return {
        "user_id": user_id,
        "model": "hybrid-tfidf-sparse-svd-logistic-ranker",
        "recommendations": [
            serialize_recommendation(model, recommendation)
            for recommendation in recommendations
        ],
    }


@app.get("/profile/{user_id}")
def profile(user_id: str) -> dict[str, object]:
    model = get_model()
    affinities = model.category_counts.get(user_id, {})
    return {
        "user_id": user_id,
        "history_size": len(model.history.get(user_id, set())),
        "top_interests": [
            {"category": category, "affinity": affinity}
            for category, affinity in sorted(
                affinities.items(), key=lambda pair: pair[1], reverse=True
            )[:8]
        ],
        "cold_start": user_id not in model.history,
    }


@app.get("/explain/{user_id}/{item_id}")
def explain_item(user_id: str, item_id: str) -> dict[str, object]:
    model = get_model()
    matches = [rec for rec in model.recommend(user_id, k=50) if rec.item_id == item_id]
    if not matches:
        raise HTTPException(status_code=404, detail="item is not in the top-50 recommendations")
    return explain(model, user_id, matches[0])


@app.post("/chat")
def chat(request: ChatRequest) -> dict[str, object]:
    model = get_model()
    recommendations = model.recommend(request.user_id, k=request.k)
    return get_assistant().chat(request.user_id, request.question, recommendations)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/demo/")


STATIC_DIRECTORY = Path(__file__).parent / "static"
app.mount("/demo", StaticFiles(directory=STATIC_DIRECTORY, html=True), name="demo")

