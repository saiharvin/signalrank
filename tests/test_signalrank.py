from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from signalrank.data import make_demo_dataset, temporal_split
from signalrank.evaluation import (
    benchmark_latency,
    evaluate,
    evaluate_history_cohorts,
    evaluate_scores,
    ndcg_at_k,
    paired_bootstrap_interval,
    reciprocal_rank,
    recall_at_k,
)
from signalrank.explain import explain
from signalrank.model import HybridRecommender


class MetricTests(unittest.TestCase):
    def test_hand_computed_metrics(self):
        labels = np.array([0, 1, 0, 1])
        self.assertEqual(reciprocal_rank(labels), 0.5)
        self.assertAlmostEqual(recall_at_k(labels, 2), 0.5)
        self.assertGreater(ndcg_at_k(labels, 4), 0)
        self.assertLessEqual(ndcg_at_k(labels, 4), 1)


class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = make_demo_dataset(users=24, items=72, impressions_per_user=12)
        cls.train, cls.test = temporal_split(cls.dataset)
        train_histories = {
            user: tuple(group.sort_values("timestamp").item_id)
            for user, group in cls.train.query("label == 1").groupby("user_id")
        }
        cls.model = HybridRecommender(latent_dim=12).fit(
            cls.dataset.articles, cls.train, train_histories
        )

    def test_temporal_split_has_no_future_leakage(self):
        train_max = self.train.groupby("user_id").timestamp.max()
        test_min = self.test.groupby("user_id").timestamp.min()
        self.assertTrue((train_max < test_min).all())

    def test_pre_impression_histories_survive_dataset_construction(self):
        self.assertIn("history_items", self.dataset.interactions.columns)
        user = self.dataset.interactions.query("user_id == 'u0001'")
        first = user.sort_values("timestamp").iloc[0]
        last = user.sort_values("timestamp").iloc[-1]
        self.assertEqual(first.history_items, tuple())
        self.assertGreater(len(last.history_items), 0)

    def test_evaluation_contract(self):
        metrics = evaluate(self.model, self.test)
        for name in ["auc", "mrr", "ndcg_at_5", "recall_at_10"]:
            self.assertGreaterEqual(metrics[name], 0)
            self.assertLessEqual(metrics[name], 1)

    def test_ranking_is_deterministic_and_valid(self):
        first = self.model.recommend("u0001", k=5)
        second = self.model.recommend("u0001", k=5)
        self.assertEqual([x.item_id for x in first], [x.item_id for x in second])
        self.assertEqual([x.rank for x in first], list(range(1, 6)))

    def test_cold_start_fallback(self):
        results = self.model.recommend("never-seen-user", k=5)
        self.assertEqual(len(results), 5)
        self.assertTrue(all(0 <= result.score <= 1 for result in results))

    def test_explanation_is_grounded(self):
        recommendation = self.model.recommend("u0001", k=1)[0]
        payload = explain(self.model, "u0001", recommendation)
        evidence = payload["evidence"][0]
        self.assertEqual(evidence["citation_id"], recommendation.item_id)
        article = self.model.article_lookup.loc[recommendation.item_id]
        self.assertEqual(evidence["passage"], article.abstract)
        self.assertIn(f"#{recommendation.rank}", payload["summary"])

    def test_model_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            self.model.save(path)
            loaded = HybridRecommender.load(path)
            before = self.model.recommend("u0002", k=3)
            after = loaded.recommend("u0002", k=3)
            self.assertEqual([x.item_id for x in before], [x.item_id for x in after])

    def test_sparse_history_and_component_baselines(self):
        self.assertTrue(hasattr(self.model.svd, "components_"))
        scores = self.model.score_components(self.test.head(20))
        self.assertEqual(
            set(scores),
            {"hybrid", "content", "collaborative", "category_affinity", "popularity"},
        )
        self.assertTrue(all(len(values) == 20 for values in scores.values()))
        self.assertGreaterEqual(self.model.blend_weight, 0)
        self.assertLessEqual(self.model.blend_weight, 1)

    def test_paired_bootstrap_and_latency(self):
        scores = self.model.score_components(self.test)
        _, hybrid = evaluate_scores(self.test, scores["hybrid"])
        _, popularity = evaluate_scores(self.test, scores["popularity"])
        interval = paired_bootstrap_interval(hybrid, popularity, samples=40)
        self.assertLessEqual(interval["ci_95_low"], interval["ci_95_high"])
        latency = benchmark_latency(self.model, self.test, requests=8, warmup=2)
        self.assertGreaterEqual(latency["warm_user"]["p95_ms"], latency["warm_user"]["p50_ms"])

    def test_exact_impression_history_and_cohorts(self):
        sample = self.test.head(20).copy()
        sample["history_items"] = [tuple() if idx < 10 else ("i0001", "i0002") for idx in range(20)]
        scores = self.model.score(sample)
        cohorts = evaluate_history_cohorts(sample, scores)
        self.assertIn("cold_0", cohorts)
        self.assertIn("sparse_1_5", cohorts)


if __name__ == "__main__":
    unittest.main()

