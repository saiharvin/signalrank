"""Impression-aware offline evaluation for recommender systems."""

from __future__ import annotations

import math
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .model import HybridRecommender


def reciprocal_rank(labels: np.ndarray) -> float:
    positives = np.flatnonzero(labels)
    return 0.0 if len(positives) == 0 else 1.0 / (positives[0] + 1)


def ndcg_at_k(labels: np.ndarray, k: int) -> float:
    labels = np.asarray(labels)[:k]
    discounts = 1 / np.log2(np.arange(2, len(labels) + 2))
    dcg = float(np.sum(labels * discounts))
    ideal = np.sort(labels)[::-1]
    idcg = float(np.sum(ideal * discounts))
    return dcg / idcg if idcg else 0.0


def recall_at_k(labels: np.ndarray, k: int) -> float:
    denominator = int(np.sum(labels))
    return float(np.sum(labels[:k]) / denominator) if denominator else 0.0


def evaluate(model: HybridRecommender, test: pd.DataFrame) -> dict[str, float | int]:
    scored = test.copy()
    scored["score"] = model.score(scored)
    aucs, mrrs, ndcg5, ndcg10, recall10 = [], [], [], [], []
    recommended_items: set[str] = set()
    for _, group in scored.groupby("impression_id", sort=False):
        ranked = group.sort_values("score", ascending=False)
        labels = ranked.label.to_numpy(dtype=int)
        if len(np.unique(labels)) == 2:
            aucs.append(roc_auc_score(group.label, group.score))
        mrrs.append(reciprocal_rank(labels))
        ndcg5.append(ndcg_at_k(labels, 5))
        ndcg10.append(ndcg_at_k(labels, 10))
        recall10.append(recall_at_k(labels, 10))
        recommended_items.update(ranked.head(10).item_id)
    return {
        "impressions": int(scored.impression_id.nunique()),
        "auc": float(np.mean(aucs)) if aucs else math.nan,
        "mrr": float(np.mean(mrrs)),
        "ndcg_at_5": float(np.mean(ndcg5)),
        "ndcg_at_10": float(np.mean(ndcg10)),
        "recall_at_10": float(np.mean(recall10)),
        "catalog_coverage_at_10": len(recommended_items) / len(model.item_ids),
    }


def evaluate_scores(test: pd.DataFrame, scores: np.ndarray) -> tuple[dict[str, float | int], pd.DataFrame]:
    """Evaluate supplied scores and retain impression-level values for bootstrap."""
    scored = test.copy()
    scored["score"] = scores
    rows = []
    recommended_items: set[str] = set()
    for impression_id, group in scored.groupby("impression_id", sort=False):
        ranked = group.sort_values("score", ascending=False)
        labels = ranked.label.to_numpy(dtype=int)
        auc = roc_auc_score(group.label, group.score) if len(np.unique(group.label)) == 2 else math.nan
        rows.append(
            {
                "impression_id": impression_id,
                "auc": auc,
                "mrr": reciprocal_rank(labels),
                "ndcg_at_5": ndcg_at_k(labels, 5),
                "ndcg_at_10": ndcg_at_k(labels, 10),
                "recall_at_10": recall_at_k(labels, 10),
            }
        )
        recommended_items.update(ranked.head(10).item_id)
    per_impression = pd.DataFrame(rows)
    metrics = {
        "impressions": len(per_impression),
        **{
            name: float(per_impression[name].mean())
            for name in ["auc", "mrr", "ndcg_at_5", "ndcg_at_10", "recall_at_10"]
        },
        "catalog_coverage_at_10": len(recommended_items) / max(test.item_id.nunique(), 1),
    }
    return metrics, per_impression


def paired_bootstrap_interval(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    metric: str = "ndcg_at_10",
    samples: int = 1000,
    seed: int = 17,
) -> dict[str, float | int]:
    """Bootstrap the paired relative improvement using identical impressions."""
    merged = candidate[["impression_id", metric]].merge(
        baseline[["impression_id", metric]], on="impression_id", suffixes=("_candidate", "_baseline")
    ).dropna()
    left = merged[f"{metric}_candidate"].to_numpy()
    right = merged[f"{metric}_baseline"].to_numpy()
    rng = np.random.default_rng(seed)
    deltas = np.empty(samples, dtype=float)
    for idx in range(samples):
        chosen = rng.integers(0, len(merged), size=len(merged))
        baseline_mean = right[chosen].mean()
        deltas[idx] = (left[chosen].mean() - baseline_mean) / max(abs(baseline_mean), 1e-12)
    point = (left.mean() - right.mean()) / max(abs(right.mean()), 1e-12)
    return {
        "metric": metric,
        "samples": samples,
        "relative_improvement": float(point),
        "ci_95_low": float(np.quantile(deltas, 0.025)),
        "ci_95_high": float(np.quantile(deltas, 0.975)),
    }


def benchmark_latency(
    model: HybridRecommender,
    candidates: pd.DataFrame,
    requests: int = 200,
    warmup: int = 10,
) -> dict[str, dict[str, float | int]]:
    """Measure candidate-ranking latency for warm and forced-cold user contexts."""
    groups = [group.copy() for _, group in candidates.groupby("impression_id", sort=False)]
    groups = groups[: min(requests, len(groups))]
    for group in groups[:warmup]:
        model.score(group)

    def measure(cold: bool) -> dict[str, float | int]:
        values = []
        for idx, group in enumerate(groups):
            request = group.copy()
            if cold:
                request["user_id"] = f"cold-benchmark-{idx}"
                if "history_items" in request.columns:
                    request["history_items"] = [tuple()] * len(request)
            started = perf_counter()
            model.score(request)
            values.append((perf_counter() - started) * 1000)
        return {
            "requests": len(values),
            "p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95)),
            "mean_candidates": float(np.mean([len(group) for group in groups])),
        }

    return {"warm_user": measure(False), "cold_user": measure(True)}


def evaluate_history_cohorts(
    test: pd.DataFrame, scores: np.ndarray
) -> dict[str, dict[str, float | int]]:
    """Report ranking quality by the amount of pre-impression user history."""
    if "history_items" not in test.columns:
        return {}
    lengths = test.history_items.map(lambda value: len(value) if isinstance(value, (tuple, list)) else 0)
    masks = {
        "cold_0": lengths == 0,
        "sparse_1_5": lengths.between(1, 5),
        "established_6_plus": lengths >= 6,
    }
    output = {}
    for name, mask in masks.items():
        if mask.any():
            output[name] = evaluate_scores(test.loc[mask], scores[mask.to_numpy()])[0]
    return output

