"""Cached hybrid retrieval over MIND titles and abstracts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass(frozen=True)
class Evidence:
    citation_id: str
    title: str
    passage: str
    category: str
    score: float


class ArticleRetriever:
    """Fuse word and character retrieval while keeping the corpus in memory."""

    def __init__(self, articles: pd.DataFrame):
        self.articles = articles.reset_index(drop=True).copy()
        text = (
            self.articles.title.fillna("") + " "
            + self.articles.abstract.fillna("") + " "
            + self.articles.category.fillna("") + " "
            + self.articles.subcategory.fillna("")
        )
        self.word_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, max_features=40_000, sublinear_tf=True
        )
        self.character_vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=3,
            max_features=30_000, sublinear_tf=True,
        )
        self.word_matrix = self.word_vectorizer.fit_transform(text)
        self.character_matrix = self.character_vectorizer.fit_transform(text)
        self.item_to_index = {
            item_id: index for index, item_id in enumerate(self.articles.item_id)
        }

    @staticmethod
    def _top_indices(scores: np.ndarray, limit: int) -> list[int]:
        if not len(scores):
            return []
        limit = min(limit, len(scores))
        selected = np.argpartition(scores, -limit)[-limit:]
        return selected[np.argsort(scores[selected])[::-1]].tolist()

    def retrieve(
        self,
        question: str,
        k: int = 6,
        preferred_item_ids: list[str] | None = None,
    ) -> list[Evidence]:
        if not question.strip():
            return []
        word_query = self.word_vectorizer.transform([question])
        character_query = self.character_vectorizer.transform([question])
        word_scores = (self.word_matrix @ word_query.T).toarray().ravel()
        character_scores = (self.character_matrix @ character_query.T).toarray().ravel()
        fused: dict[int, float] = {}
        for scores in (word_scores, character_scores):
            for rank, index in enumerate(self._top_indices(scores, 60), start=1):
                if scores[index] > 0:
                    fused[index] = fused.get(index, 0.0) + 1 / (60 + rank)
        for item_id in preferred_item_ids or []:
            index = self.item_to_index.get(item_id)
            if index is not None:
                fused[index] = fused.get(index, 0.0) + 0.04
        evidence = []
        for index, score in sorted(fused.items(), key=lambda pair: pair[1], reverse=True)[:k]:
            article = self.articles.iloc[index]
            evidence.append(Evidence(
                citation_id=str(article.item_id),
                title=str(article.title),
                passage=str(article.abstract).strip() or str(article.title).strip(),
                category=str(article.category),
                score=float(score),
            ))
        return evidence

