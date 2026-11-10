"""Dataset contracts, MIND ingestion, and a deterministic smoke-test fixture."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ARTICLE_COLUMNS = ["item_id", "category", "subcategory", "title", "abstract", "entities"]
INTERACTION_COLUMNS = [
    "impression_id", "user_id", "timestamp", "item_id", "label", "position",
    "history_items",
]


@dataclass(frozen=True)
class Dataset:
    articles: pd.DataFrame
    interactions: pd.DataFrame
    user_histories: dict[str, tuple[str, ...]] | None = None

    def validate(self) -> None:
        missing_articles = set(ARTICLE_COLUMNS) - set(self.articles.columns)
        missing_interactions = set(INTERACTION_COLUMNS) - set(self.interactions.columns)
        if missing_articles or missing_interactions:
            raise ValueError(
                f"schema error: article={sorted(missing_articles)}, "
                f"interaction={sorted(missing_interactions)}"
            )
        if self.articles.item_id.duplicated().any():
            raise ValueError("item_id must be unique")
        if not set(self.interactions.label.unique()).issubset({0, 1}):
            raise ValueError("labels must be binary")
        unknown = set(self.interactions.item_id) - set(self.articles.item_id)
        if unknown:
            raise ValueError(f"interactions reference {len(unknown)} unknown articles")


def load_mind(
    directory: str | Path,
    max_impressions: int | None = None,
    split_name: str | None = None,
) -> Dataset:
    """Load a MIND directory while preserving each displayed candidate impression."""
    directory = Path(directory)
    news_path = directory / "news.tsv"
    behavior_path = directory / "behaviors.tsv"
    if not news_path.exists() or not behavior_path.exists():
        raise FileNotFoundError("expected news.tsv and behaviors.tsv in the data directory")

    news_names = [
        "item_id", "category", "subcategory", "title", "abstract",
        "url", "title_entities", "abstract_entities",
    ]
    articles = pd.read_csv(news_path, sep="\t", names=news_names, dtype=str).fillna("")
    articles["entities"] = articles.title_entities + " " + articles.abstract_entities
    articles = articles[ARTICLE_COLUMNS]

    behavior_names = ["impression_id", "user_id", "timestamp", "history", "impressions"]
    rows: list[dict[str, object]] = []
    histories: dict[str, list[str]] = {}
    seen_impressions = 0
    prefix = f"{split_name}:" if split_name else ""
    for chunk in (
        pd.read_csv(behavior_path, sep="\t", names=behavior_names, dtype=str, chunksize=10_000)
    ):
        for record in chunk.itertuples(index=False):
            if max_impressions is not None and seen_impressions >= max_impressions:
                break
            seen_impressions += 1
            user_id = str(record.user_id)
            history_tokens = [] if pd.isna(record.history) else str(record.history).split()
            if history_tokens:
                current = histories.setdefault(user_id, [])
                current.extend(item for item in history_tokens if item not in current)
            for position, token in enumerate(str(record.impressions).split()):
                item_id, label = token.rsplit("-", 1)
                rows.append(
                    {
                        "impression_id": prefix + str(record.impression_id),
                        "user_id": user_id,
                        "timestamp": pd.to_datetime(record.timestamp, utc=True),
                        "item_id": item_id,
                        "label": int(label),
                        "position": position,
                        "history_items": tuple(history_tokens),
                    }
                )
        if max_impressions is not None and seen_impressions >= max_impressions:
            break
    interactions = pd.DataFrame(rows, columns=INTERACTION_COLUMNS)
    dataset = Dataset(
        articles.reset_index(drop=True),
        interactions.reset_index(drop=True),
        {user: tuple(items) for user, items in histories.items()},
    )
    dataset.validate()
    return dataset


def make_demo_dataset(
    users: int = 80,
    items: int = 160,
    impressions_per_user: int = 18,
    candidates: int = 8,
    seed: int = 17,
) -> Dataset:
    """Create learnable synthetic civic behavior for tests, never for accuracy claims."""
    rng = np.random.default_rng(seed)
    topics = {
        "transportation": ["transit", "traffic", "bus", "rail", "street"],
        "housing": ["housing", "rent", "zoning", "homes", "development"],
        "energy": ["solar", "grid", "energy", "battery", "utility"],
        "education": ["school", "students", "teachers", "campus", "bond"],
        "water": ["water", "flood", "drainage", "river", "infrastructure"],
        "health": ["clinic", "health", "care", "hospital", "community"],
    }
    topic_names = list(topics)
    article_rows = []
    for idx in range(items):
        topic = topic_names[idx % len(topic_names)]
        words = topics[topic]
        article_rows.append(
            {
                "item_id": f"i{idx:04d}",
                "category": topic,
                "subcategory": "local-policy",
                "title": f"{words[0].title()} proposal {idx}",
                "abstract": f"Community evidence about {' '.join(words)} and public priorities.",
                "entities": f"district-{idx % 9} council",
            }
        )
    articles = pd.DataFrame(article_rows, columns=ARTICLE_COLUMNS)

    interactions = []
    base = pd.Timestamp("2025-01-01", tz="UTC")
    for user_idx in range(users):
        preferred = {topic_names[user_idx % len(topic_names)], topic_names[(user_idx + 2) % len(topic_names)]}
        clicked_history: list[str] = []
        for step in range(impressions_per_user):
            chosen = rng.choice(items, size=candidates, replace=False)
            scores = np.array([
                2.8 * (articles.iloc[item].category in preferred)
                + 0.35 * rng.normal()
                - 0.002 * item
                for item in chosen
            ])
            clicked_position = int(np.argmax(scores))
            impression_id = f"p{user_idx:04d}-{step:03d}"
            for position, item_idx in enumerate(chosen):
                interactions.append(
                    {
                        "impression_id": impression_id,
                        "user_id": f"u{user_idx:04d}",
                        "timestamp": base + pd.Timedelta(days=step, minutes=user_idx),
                        "item_id": f"i{item_idx:04d}",
                        "label": int(position == clicked_position),
                        "position": position,
                        "history_items": tuple(clicked_history),
                    }
                )
            clicked_history.append(f"i{chosen[clicked_position]:04d}")
    histories = {}
    positive = pd.DataFrame(interactions).query("label == 1")
    for user, group in positive.groupby("user_id"):
        histories[user] = tuple(group.sort_values("timestamp").item_id.tolist())
    dataset = Dataset(
        articles,
        pd.DataFrame(interactions, columns=INTERACTION_COLUMNS),
        histories,
    )
    dataset.validate()
    return dataset


def temporal_split(dataset: Dataset, test_fraction: float = 0.25) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out the latest impressions per user to prevent future leakage."""
    frame = dataset.interactions.sort_values(["user_id", "timestamp", "impression_id"])
    impression_times = frame.groupby(["user_id", "impression_id"], as_index=False).timestamp.min()
    impression_times["order"] = impression_times.groupby("user_id").cumcount()
    counts = impression_times.groupby("user_id").impression_id.transform("count")
    impression_times["is_test"] = impression_times.order >= np.floor(counts * (1 - test_fraction))
    flags = impression_times.set_index(["user_id", "impression_id"]).is_test
    index = pd.MultiIndex.from_frame(frame[["user_id", "impression_id"]])
    mask = flags.reindex(index).to_numpy(dtype=bool)
    return frame.loc[~mask].reset_index(drop=True), frame.loc[mask].reset_index(drop=True)

