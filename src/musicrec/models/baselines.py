"""Non-personalized baselines. Any real model must beat these to be worth anything."""
from __future__ import annotations

import numpy as np

from musicrec.preprocess import Split, days


class Random:
    """Sanity floor: if a model can't beat this, there is a bug."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.name = "Random"

    def fit(self, split: Split) -> "Random":
        self.n_items = split.n_items
        self.rng = np.random.default_rng(self.seed)
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        return self.rng.random((len(users), self.n_items), dtype=np.float32)


class MostPopular:
    """Everyone gets the same list: the most popular tracks they haven't played yet.

    by="listeners": popularity = number of distinct users (robust to heavy re-listeners)
    by="plays"    : popularity = total plays
    recent_days   : only count the last N days of train ("what's trending now")
    """

    def __init__(self, by: str = "listeners", recent_days: int | None = None):
        self.by, self.recent_days = by, recent_days
        self.name = f"MostPopular({by}" + (f", last {recent_days}d)" if recent_days else ")")

    def fit(self, split: Split) -> "MostPopular":
        ev = split.train_events
        if self.recent_days:
            ev = ev[ev["ts"] >= ev["ts"].max() - days(ev["ts"], self.recent_days)]
        if self.by == "listeners":
            counts = ev.groupby("item_idx")["user_idx"].nunique()
        else:
            counts = ev.groupby("item_idx").size()
        self.popularity_ = np.zeros(split.n_items, dtype=np.float32)
        self.popularity_[counts.index.to_numpy()] = counts.to_numpy()
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        # Same row for every user; broadcast_to avoids allocating n_users copies
        return np.broadcast_to(self.popularity_, (len(users), len(self.popularity_)))
