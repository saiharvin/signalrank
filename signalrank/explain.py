"""Cited explanations derived only from model output and stored evidence."""

from __future__ import annotations

from dataclasses import asdict

from .model import HybridRecommender, Recommendation


def explain(model: HybridRecommender, user_id: str, recommendation: Recommendation) -> dict[str, object]:
    article = model.article_lookup.loc[recommendation.item_id]
    evidence = str(article.abstract).strip() or str(article.title).strip()
    return {
        "user_id": user_id,
        "recommendation": asdict(recommendation),
        "summary": (
            f"{article.title} was ranked #{recommendation.rank} because "
            + " and ".join(recommendation.reasons)
            + "."
        ),
        "evidence": [
            {
                "citation_id": recommendation.item_id,
                "title": str(article.title),
                "passage": evidence,
                "category": str(article.category),
            }
        ],
        "limitations": [
            "This is a predictive association, not a causal claim.",
            "The score estimates relative engagement within this catalog.",
        ],
    }

