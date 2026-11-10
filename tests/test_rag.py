from __future__ import annotations

import os
import unittest

from signalrank.chat import RecommendationAssistant, validate_claims
from signalrank.data import make_demo_dataset, temporal_split
from signalrank.model import HybridRecommender
from signalrank.rag import ArticleRetriever


class RagTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = make_demo_dataset(users=18, items=60, impressions_per_user=10)
        train, _ = temporal_split(data)
        histories = {
            user: tuple(group.sort_values("timestamp").item_id)
            for user, group in train.query("label == 1").groupby("user_id")
        }
        cls.model = HybridRecommender(latent_dim=8).fit(data.articles, train, histories)

    def test_retrieval_returns_real_article_ids(self):
        evidence = ArticleRetriever(self.model.articles).retrieve("housing zoning development", k=4)
        self.assertEqual(len(evidence), 4)
        self.assertTrue(all(item.citation_id in self.model.item_to_idx for item in evidence))

    def test_invalid_model_citation_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_claims(
                {"claims": [{"text": "Unsupported", "sources": ["invented"]}]},
                {"i0001"},
            )

    def test_extractive_chat_is_grounded(self):
        os.environ["SIGNALRANK_CHAT_PROVIDER"] = "extractive"
        assistant = RecommendationAssistant(self.model)
        recommendations = self.model.recommend("u0001", k=3)
        response = assistant.chat("u0001", "Why was the top item recommended?", recommendations)
        valid = {source["citation_id"] for source in response["sources"]}
        self.assertTrue(response["claims"])
        self.assertTrue(set(response["claims"][0]["sources"]) <= valid)


if __name__ == "__main__":
    unittest.main()

