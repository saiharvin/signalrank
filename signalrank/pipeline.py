"""End-to-end training, evaluation, and model-card generation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .data import Dataset, temporal_split
from .evaluation import (
    benchmark_latency,
    evaluate,
    evaluate_history_cohorts,
    evaluate_scores,
    paired_bootstrap_interval,
)
from .model import FEATURE_NAMES, HybridRecommender


def train_pipeline(dataset: Dataset, output: str | Path, dataset_name: str) -> dict[str, object]:
    dataset.validate()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    train, test = temporal_split(dataset)
    train_histories = {
        user: tuple(group.sort_values("timestamp").item_id)
        for user, group in train[train.label == 1].groupby("user_id")
    }
    model = HybridRecommender().fit(dataset.articles, train, train_histories)
    metrics = evaluate(model, test)
    model.save(output / "model.joblib")
    report = {
        "model": "hybrid-tfidf-svd-logistic-ranker",
        "dataset": dataset_name,
        "warning": (
            "Synthetic smoke-test metrics are not evidence of real-world accuracy."
            if dataset_name == "synthetic-demo" else None
        ),
        "split": "latest 25% of impressions per user",
        "train_rows": len(train),
        "test_rows": len(test),
        "metrics": metrics,
        "responsible_use": {
            "allowed": "aggregate issue discovery and evidence review",
            "prohibited": "protected-trait inference or individual political persuasion",
        },
    }
    (output / "model_card.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def benchmark_mind(
    train_data: Dataset,
    dev_data: Dataset,
    output: str | Path,
    bootstrap_samples: int = 1000,
) -> dict[str, object]:
    """Run the official train/dev boundary with baseline and systems evaluation."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    articles = (
        pd.concat([train_data.articles, dev_data.articles])
        .drop_duplicates("item_id", keep="last")
        .reset_index(drop=True)
    )
    model = HybridRecommender().fit(
        articles,
        train_data.interactions,
        train_data.user_histories,
    )
    model.add_user_histories(dev_data.user_histories or {})
    component_scores = model.score_components(dev_data.interactions)
    model_reports = {}
    detail = {}
    for name, scores in component_scores.items():
        model_reports[name], detail[name] = evaluate_scores(dev_data.interactions, scores)
    baseline_names = [name for name in model_reports if name != "hybrid"]
    strongest = max(baseline_names, key=lambda name: model_reports[name]["ndcg_at_10"])
    confidence = paired_bootstrap_interval(
        detail["hybrid"],
        detail[strongest],
        samples=bootstrap_samples,
    )
    latency = benchmark_latency(model, dev_data.interactions)
    model.save(output / "model.joblib")
    report = {
        "model": "hybrid-tfidf-sparse-svd-logistic-ranker",
        "dataset": "MIND",
        "protocol": "train on MIND train and evaluate on MIND dev impressions",
        "train_impressions": int(train_data.interactions.impression_id.nunique()),
        "dev_impressions": int(dev_data.interactions.impression_id.nunique()),
        "train_rows": len(train_data.interactions),
        "dev_rows": len(dev_data.interactions),
        "models": model_reports,
        "feature_coefficients": {
            name: float(value)
            for name, value in zip(FEATURE_NAMES, model.ranker.coef_[0])
        },
        "train_calibrated_learned_ranker_weight": model.blend_weight,
        "hybrid_vs_baselines_ndcg_at_10": {
            name: float(
                (model_reports["hybrid"]["ndcg_at_10"] - model_reports[name]["ndcg_at_10"])
                / max(abs(model_reports[name]["ndcg_at_10"]), 1e-12)
            )
            for name in baseline_names
        },
        "hybrid_history_cohorts": evaluate_history_cohorts(
            dev_data.interactions, component_scores["hybrid"]
        ),
        "strongest_baseline": strongest,
        "paired_bootstrap": confidence,
        "latency": {
            "scope": "end-to-end scoring of candidates in one observed impression",
            **latency,
        },
        "resume_ready": bool(confidence["ci_95_low"] > 0),
        "responsible_use": {
            "allowed": "aggregate issue discovery and evidence review",
            "prohibited": "protected-trait inference or individual political persuasion",
        },
    }
    (output / "benchmark_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report

