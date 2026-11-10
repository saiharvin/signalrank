"""Hybrid implicit-feedback ranker with cold-start and diversity controls."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


FEATURE_NAMES = [
    "content_similarity", "collaborative_similarity", "category_affinity",
    "popularity", "history_size",
]


@dataclass(frozen=True)
class Recommendation:
    item_id: str
    score: float
    rank: int
    reasons: tuple[str, ...]


class HybridRecommender:
    def __init__(self, latent_dim: int = 24, random_state: int = 17):
        self.latent_dim = latent_dim
        self.random_state = random_state
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=12_000)
        self.scaler = StandardScaler()
        self.ranker = LogisticRegression(max_iter=500, class_weight="balanced", random_state=random_state)
        self.svd: TruncatedSVD | None = None
        self.fitted = False

    def fit(
        self,
        articles: pd.DataFrame,
        interactions: pd.DataFrame,
        user_histories: dict[str, tuple[str, ...]] | None = None,
    ) -> "HybridRecommender":
        self.articles = articles.reset_index(drop=True).copy()
        self.item_ids = self.articles.item_id.tolist()
        self.item_to_idx = {item: idx for idx, item in enumerate(self.item_ids)}
        self.article_lookup = self.articles.set_index("item_id")
        text = (
            self.articles.title.fillna("") + " " + self.articles.abstract.fillna("")
            + " " + self.articles.category.fillna("") + " " + self.articles.entities.fillna("")
        )
        self.item_content = self.vectorizer.fit_transform(text)

        positive = interactions[interactions.label == 1].copy()
        histories = {user: list(items) for user, items in (user_histories or {}).items()}
        for user, group in positive.groupby("user_id"):
            current = histories.setdefault(user, [])
            current.extend(item for item in group.item_id if item not in current)
        self.user_ids = sorted(set(interactions.user_id.unique()) | set(histories))
        self.user_to_idx = {user: idx for idx, user in enumerate(self.user_ids)}
        matrix = self._history_matrix(histories)
        max_dim = max(1, min(self.latent_dim, min(matrix.shape) - 1))
        self.svd = TruncatedSVD(n_components=max_dim, random_state=self.random_state)
        self.user_latent = self.svd.fit_transform(matrix)
        self.item_latent = self.svd.components_.T
        self.user_content = {}
        self.category_counts = {}
        self.history = {}
        for user, items in histories.items():
            known = [item for item in items if item in self.item_to_idx]
            ids = [self.item_to_idx[item] for item in known]
            self.history[user] = set(known)
            self.user_content[user] = self.item_content[ids].mean(axis=0) if ids else None
            categories = self.article_lookup.loc[known, "category"] if known else pd.Series(dtype=str)
            self.category_counts[user] = categories.value_counts(normalize=True).to_dict()
        self.popularity = positive.item_id.value_counts().reindex(self.item_ids, fill_value=0)
        self.popularity = np.log1p(self.popularity) / max(np.log1p(self.popularity).max(), 1.0)

        features = self._feature_matrix(interactions)
        labels = interactions.label.to_numpy(dtype=int)
        if len(np.unique(labels)) < 2:
            raise ValueError("training requires both clicked and displayed-negative examples")
        scaled = self.scaler.fit_transform(features)
        self.blend_weight = self._calibrate_blend(interactions, scaled, features, labels)
        self.ranker.fit(scaled, labels)
        self.fitted = True
        return self

    def _calibrate_blend(
        self,
        interactions: pd.DataFrame,
        scaled: np.ndarray,
        raw: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """Choose a ranker/category blend on a train-only chronological holdout."""
        impressions = (
            interactions[["impression_id", "timestamp"]]
            .drop_duplicates("impression_id")
            .sort_values(["timestamp", "impression_id"])
        )
        if len(impressions) < 10:
            return 1.0
        boundary = max(1, int(len(impressions) * 0.8))
        fit_ids = set(impressions.iloc[:boundary].impression_id)
        calibration_ids = set(impressions.iloc[boundary:].impression_id)
        fit_mask = interactions.impression_id.isin(fit_ids).to_numpy()
        calibration_mask = interactions.impression_id.isin(calibration_ids).to_numpy()
        if len(np.unique(labels[fit_mask])) < 2 or not calibration_mask.any():
            return 1.0
        temporary = LogisticRegression(
            max_iter=500,
            class_weight="balanced",
            random_state=self.random_state,
        ).fit(scaled[fit_mask], labels[fit_mask])
        learned = temporary.predict_proba(scaled[calibration_mask])[:, 1]
        category = raw[calibration_mask, 2]
        calibration = interactions.loc[calibration_mask, ["impression_id", "label"]].copy()

        def mean_ndcg(scores: np.ndarray) -> float:
            calibration["score"] = scores
            values = []
            for _, group in calibration.groupby("impression_id", sort=False):
                ranked = group.sort_values("score", ascending=False).label.to_numpy(dtype=float)[:10]
                discounts = 1 / np.log2(np.arange(2, len(ranked) + 2))
                ideal = np.sort(ranked)[::-1]
                denominator = float(np.sum(ideal * discounts))
                values.append(float(np.sum(ranked * discounts)) / denominator if denominator else 0.0)
            return float(np.mean(values))

        candidates = np.linspace(0, 1, 21)
        quality = [mean_ndcg(alpha * learned + (1 - alpha) * category) for alpha in candidates]
        return float(candidates[int(np.argmax(quality))])

    def _history_matrix(self, histories: dict[str, list[str] | tuple[str, ...]]) -> csr_matrix:
        rows, cols = [], []
        for user, items in histories.items():
            user_idx = self.user_to_idx.get(user)
            if user_idx is None:
                continue
            for item in set(items):
                item_idx = self.item_to_idx.get(item)
                if item_idx is not None:
                    rows.append(user_idx)
                    cols.append(item_idx)
        values = np.ones(len(rows), dtype=np.float32)
        return coo_matrix(
            (values, (rows, cols)),
            shape=(len(self.user_ids), len(self.item_ids)),
            dtype=np.float32,
        ).tocsr()

    def add_user_histories(self, user_histories: dict[str, tuple[str, ...]]) -> None:
        """Attach pre-impression histories without using evaluation labels."""
        self._check_fitted()
        for user, items in user_histories.items():
            known = [item for item in items if item in self.item_to_idx]
            if not known:
                continue
            ids = [self.item_to_idx[item] for item in known]
            self.history[user] = set(known)
            self.user_content[user] = self.item_content[ids].mean(axis=0)
            categories = self.article_lookup.loc[known, "category"]
            self.category_counts[user] = categories.value_counts(normalize=True).to_dict()
            row = csr_matrix(
                (
                    np.ones(len(ids), dtype=np.float32),
                    (np.zeros(len(ids), dtype=int), np.asarray(ids)),
                ),
                shape=(1, len(self.item_ids)),
            )
            latent = self.svd.transform(row)[0]
            if user in self.user_to_idx:
                self.user_latent[self.user_to_idx[user]] = latent
            else:
                self.user_to_idx[user] = len(self.user_ids)
                self.user_ids.append(user)
                self.user_latent = np.vstack([self.user_latent, latent])

    def _feature_matrix(self, rows: pd.DataFrame) -> np.ndarray:
        output = np.zeros((len(rows), len(FEATURE_NAMES)), dtype=float)
        impression_contexts = self._impression_contexts(rows)
        for out_idx, row in enumerate(rows.itertuples(index=False)):
            item_idx = self.item_to_idx.get(row.item_id)
            if item_idx is None:
                continue
            user = row.user_id
            context = impression_contexts.get(getattr(row, "impression_id", None))
            profile = context[0] if context else self.user_content.get(user)
            if profile is not None:
                dot = float(np.asarray(profile @ self.item_content[item_idx].T).ravel()[0])
            else:
                dot = 0.0
            if context:
                collaborative = float(context[1] @ self.item_latent[item_idx])
            elif user in self.user_to_idx and self.svd is not None:
                collaborative = float(self.user_latent[self.user_to_idx[user]] @ self.item_latent[item_idx])
            else:
                collaborative = 0.0
            category = self.article_lookup.loc[row.item_id, "category"]
            category_counts = context[2] if context else self.category_counts.get(user, {})
            history_size = context[3] if context else len(self.history.get(user, set()))
            affinity = category_counts.get(category, 0.0)
            output[out_idx] = [
                dot,
                collaborative,
                affinity,
                float(self.popularity.loc[row.item_id]),
                np.log1p(history_size),
            ]
        return output

    def _impression_contexts(
        self, rows: pd.DataFrame
    ) -> dict[str, tuple[object, np.ndarray, dict[str, float], int]]:
        """Build exact pre-impression contexts in one sparse projection batch."""
        if "history_items" not in rows.columns or self.svd is None:
            return {}
        unique = rows.drop_duplicates("impression_id")[["impression_id", "history_items"]]
        histories = []
        row_indices, col_indices = [], []
        for row_idx, record in enumerate(unique.itertuples(index=False)):
            raw = record.history_items
            items = list(raw) if isinstance(raw, (tuple, list)) else []
            known = [item for item in items if item in self.item_to_idx]
            histories.append(known)
            for item in set(known):
                row_indices.append(row_idx)
                col_indices.append(self.item_to_idx[item])
        matrix = coo_matrix(
            (
                np.ones(len(row_indices), dtype=np.float32),
                (row_indices, col_indices),
            ),
            shape=(len(unique), len(self.item_ids)),
            dtype=np.float32,
        ).tocsr()
        latent = self.svd.transform(matrix)
        contexts = {}
        for row_idx, (record, known) in enumerate(zip(unique.itertuples(index=False), histories)):
            ids = [self.item_to_idx[item] for item in known]
            profile = self.item_content[ids].mean(axis=0) if ids else None
            categories = self.article_lookup.loc[known, "category"] if known else pd.Series(dtype=str)
            contexts[record.impression_id] = (
                profile,
                latent[row_idx],
                categories.value_counts(normalize=True).to_dict(),
                len(known),
            )
        return contexts

    def score_components(self, rows: pd.DataFrame) -> dict[str, np.ndarray]:
        """Return hybrid scores and auditable component baselines."""
        self._check_fitted()
        features = self._feature_matrix(rows)
        learned = self.ranker.predict_proba(self.scaler.transform(features))[:, 1]
        return {
            "hybrid": self.blend_weight * learned + (1 - self.blend_weight) * features[:, 2],
            "content": features[:, 0],
            "collaborative": features[:, 1],
            "category_affinity": features[:, 2],
            "popularity": features[:, 3],
        }

    def score(self, rows: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        features = self._feature_matrix(rows)
        learned = self.ranker.predict_proba(self.scaler.transform(features))[:, 1]
        return self.blend_weight * learned + (1 - self.blend_weight) * features[:, 2]

    def recommend(self, user_id: str, k: int = 10, diversity: float = 0.15) -> list[Recommendation]:
        self._check_fitted()
        unseen = [item for item in self.item_ids if item not in self.history.get(user_id, set())]
        candidates = pd.DataFrame(
            {
                "user_id": user_id,
                "item_id": unseen,
                "position": 0,
                "label": 0,
                "impression_id": "inference",
                "timestamp": pd.Timestamp.now(tz="UTC"),
            }
        )
        scores = self.score(candidates)
        pool = np.argsort(-scores)[: max(k * 8, k)]
        selected: list[int] = []
        while len(selected) < min(k, len(pool)):
            best_idx = None
            best_value = -np.inf
            for candidate_idx in pool:
                if int(candidate_idx) in selected:
                    continue
                redundancy = 0.0
                if selected:
                    sims = self.item_content[[self.item_to_idx[unseen[i]] for i in selected]] @ self.item_content[self.item_to_idx[unseen[int(candidate_idx)]]].T
                    redundancy = float(np.asarray(sims.max()).ravel()[0])
                value = (1 - diversity) * scores[candidate_idx] - diversity * redundancy
                if value > best_value:
                    best_value = value
                    best_idx = int(candidate_idx)
            if best_idx is None:
                break
            selected.append(best_idx)
        result = []
        feature_rows = self._feature_matrix(candidates.iloc[selected])
        for rank, (candidate_idx, features) in enumerate(zip(selected, feature_rows), start=1):
            reasons = self._reasons(features)
            result.append(Recommendation(unseen[candidate_idx], float(scores[candidate_idx]), rank, reasons))
        return result

    def _reasons(self, features: np.ndarray) -> tuple[str, ...]:
        weighted = features * self.ranker.coef_[0]
        indices = np.argsort(-weighted)[:2]
        labels = {
            0: "matches previously engaged content",
            1: "similar users engaged with it",
            2: "matches an established topic interest",
            3: "has broad recent engagement",
            4: "supported by sufficient behavioral history",
        }
        return tuple(labels[int(idx)] for idx in indices if weighted[idx] > 0) or ("cold-start relevance prior",)

    def save(self, path: str | Path) -> None:
        self._check_fitted()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "HybridRecommender":
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError("artifact is not a HybridRecommender")
        return model

    def _check_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("fit or load a model before inference")

