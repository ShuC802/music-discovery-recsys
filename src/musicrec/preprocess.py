"""Stage 2: cleaning, temporal split, filtering, and ID encoding. Works for both datasets.

Vocabulary used throughout the project
--------------------------------------
track_key : the dataset's own track identifier
            (Last.fm: integer ID of an (artist, title) pair; Yambda: item_id)
item_idx  : contiguous 0..n_items-1 index of a track kept in the model vocabulary
user_idx  : contiguous 0..n_users-1 index of a user kept for training
Models only ever see user_idx / item_idx, which is why the same model code runs on
any dataset.

Time
----
`ts` is a timezone-aware datetime for Last.fm and integer seconds for Yambda. Cut points
follow the same type: "2009-01-01" for Last.fm, 270 * DAY for Yambda.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ time helpers

def to_cutoff(x):
    """'2009-01-01' -> UTC Timestamp; an int (seconds) stays an int."""
    return pd.Timestamp(x, tz="UTC") if isinstance(x, str) else x


def days(ts: pd.Series, n: float):
    """A span of n days in the same unit as `ts`."""
    return pd.Timedelta(days=n) if pd.api.types.is_datetime64_any_dtype(ts) else n * 86_400


def age_days(cutoff, ts: pd.Series) -> np.ndarray:
    """How many days before `cutoff` each event happened."""
    diff = to_cutoff(cutoff) - ts
    secs = diff.dt.total_seconds() if pd.api.types.is_timedelta64_dtype(diff) else diff
    return np.asarray(secs, dtype=np.float64) / 86_400


# ------------------------------------------------------------------ Last.fm cleaning

LASTFM_START = pd.Timestamp("2005-01-01", tz="UTC")
LASTFM_END = pd.Timestamp("2009-05-01", tz="UTC")


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Last.fm only. Return (events, tracks).

    events: user_id, ts, track_key   (sorted by user, then time)
    tracks: track_key, artist_name, track_name
    """
    n0 = len(df)
    df = df.dropna(subset=["user_id", "artist_name", "track_name"])
    df = df[(df["ts"] >= LASTFM_START) & (df["ts"] < LASTFM_END)].copy()
    n_time = len(df)

    df["track_key"] = df.groupby(["artist_name", "track_name"], observed=True, sort=False).ngroup().astype("int32")
    df = df.drop_duplicates(subset=["user_id", "ts", "track_key"])

    tracks = (df.drop_duplicates("track_key")[["track_key", "artist_name", "track_name"]]
                .sort_values("track_key").reset_index(drop=True))
    events = (df[["user_id", "ts", "track_key"]]
                .sort_values(["user_id", "ts"], kind="stable").reset_index(drop=True))

    print(f"clean: {n0:,} rows -> {n_time:,} after time window -> {len(events):,} after dedup "
          f"| {events['user_id'].nunique()} users | {len(tracks):,} tracks")
    return events, tracks


# ------------------------------------------------------------------ split building blocks
#
# Speed trick used below: a (user, track) pair is packed into ONE 64-bit integer,
#     key = user << 32 | track
# so "which pairs did this user already hear?" becomes set operations on a single
# integer array (np.unique, np.searchsorted) instead of slow two-column pandas
# drop_duplicates / merge. Both IDs fit in 32 bits, so the packing is lossless.

def _user_ints(s: pd.Series) -> tuple[np.ndarray, np.ndarray | None]:
    """User IDs as non-negative ints (+ the categories to translate back, if categorical)."""
    if isinstance(s.dtype, pd.CategoricalDtype):
        return s.cat.codes.to_numpy().astype(np.int64), np.asarray(s.cat.categories.astype(str))
    return s.to_numpy().astype(np.int64), None


def _pack(users: np.ndarray, tracks: np.ndarray) -> np.ndarray:
    return (users.astype(np.int64) << 32) | tracks.astype(np.int64)


def _unpack(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return keys >> 32, keys & 0xFFFFFFFF


def _in_sorted(values: np.ndarray, sorted_set: np.ndarray) -> np.ndarray:
    """Boolean mask: is each value in `sorted_set`? Binary search, O(n log m)."""
    if len(sorted_set) == 0:
        return np.zeros(len(values), dtype=bool)
    pos = np.minimum(np.searchsorted(sorted_set, values), len(sorted_set) - 1)
    return sorted_set[pos] == values


class _Prepared:
    """Arrays shared by every step of one split, computed once."""

    def __init__(self, events: pd.DataFrame, train_end, eval_end,
                 positive: str | None, target: str | None):
        t0, t1 = to_cutoff(train_end), to_cutoff(eval_end)
        ts = events["ts"]
        self.in_train = (ts < t0).to_numpy()
        self.in_window = ((ts >= t0) & (ts < t1)).to_numpy()
        self.users, self.user_cats = _user_ints(events["user_id"])
        self.tracks = events["track_key"].to_numpy().astype(np.int64)
        self.keys = _pack(self.users, self.tracks)
        self.positive = np.ones(len(events), bool) if positive is None else events[positive].to_numpy()
        self.target = self.positive if target is None else self.positive & events[target].to_numpy()
        # every (user, track) pair heard in train, skips included: "already heard"
        self.seen = np.unique(self.keys[self.in_train])

    def new_pairs(self, target_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(users, tracks, plays) for pairs in the eval window never heard in train."""
        keys, plays = np.unique(self.keys[self.in_window & target_mask], return_counts=True)
        new = ~_in_sorted(keys, self.seen)
        users, tracks = _unpack(keys[new])
        return users, tracks, plays[new]

    def item_listeners(self) -> pd.Series:
        """Distinct users with a positive train interaction, per track."""
        pairs = np.unique(self.keys[self.in_train & self.positive])
        tracks, n_users = np.unique(_unpack(pairs)[1], return_counts=True)
        return pd.Series(n_users, index=tracks)


def item_threshold_table(events: pd.DataFrame, train_end, eval_end,
                         thresholds=(1, 2, 3, 5, 10, 20, 50),
                         positive: str | None = None, target: str | None = None) -> pd.DataFrame:
    """Trade-off table for the 'minimum distinct listeners per track' filter.

    Higher threshold -> smaller, denser vocabulary (easier for CF), but fewer of the
    future discoveries are even recommendable. Computed on the validation split only.
    """
    p = _Prepared(events, train_end, eval_end, positive, target)
    item_users = p.item_listeners()
    _, new_tracks, _ = p.new_pairs(p.target)
    train_tracks = p.tracks[p.in_train & p.positive]

    rows = []
    for k in thresholds:
        keep = np.sort(item_users.index[item_users >= k].to_numpy())
        rows.append({
            "min_listeners": k,
            "n_items": len(keep),
            "train interactions kept": _in_sorted(train_tracks, keep).mean(),
            "future discoveries recommendable": _in_sorted(new_tracks, keep).mean(),
        })
    cold = 1 - _in_sorted(new_tracks, np.sort(item_users.index.to_numpy())).mean()
    print(f"{len(new_tracks):,} (user, new track) pairs in the eval window; "
          f"{cold:.1%} are tracks nobody interacted with during training (pure cold start)")
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ the split object

@dataclass
class Split:
    """Everything a model needs for one train/eval experiment."""
    train_end: object
    eval_end: object
    train_events: pd.DataFrame     # user_idx, item_idx, ts      (for sequential models)
    train_counts: pd.DataFrame     # user_idx, item_idx, plays   (for CF / matrix factorization)
    targets: pd.DataFrame          # user_idx, item_idx, plays   new tracks in the eval window
    user_ids: np.ndarray           # user_idx  -> original user id
    item_keys: np.ndarray          # item_idx  -> track_key
    stats: dict = field(default_factory=dict)

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    @property
    def n_items(self) -> int:
        return len(self.item_keys)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.train_events.to_parquet(path / "train_events.parquet", index=False)
        self.train_counts.to_parquet(path / "train_counts.parquet", index=False)
        self.targets.to_parquet(path / "targets.parquet", index=False)
        pd.DataFrame({"user_id": self.user_ids}).to_parquet(path / "users.parquet", index=False)
        pd.DataFrame({"track_key": self.item_keys}).to_parquet(path / "items.parquet", index=False)
        meta = {"train_end": self.train_end, "eval_end": self.eval_end, "stats": self.stats}
        (path / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

    @classmethod
    def load(cls, path: Path) -> "Split":
        meta = json.loads((path / "meta.json").read_text())
        return cls(
            train_end=meta["train_end"], eval_end=meta["eval_end"],
            train_events=pd.read_parquet(path / "train_events.parquet"),
            train_counts=pd.read_parquet(path / "train_counts.parquet"),
            targets=pd.read_parquet(path / "targets.parquet"),
            user_ids=pd.read_parquet(path / "users.parquet")["user_id"].to_numpy(),
            item_keys=pd.read_parquet(path / "items.parquet")["track_key"].to_numpy(),
            stats=meta["stats"],
        )


def make_split(events: pd.DataFrame, train_end, eval_end,
               min_item_users: int = 10, min_user_items: int = 10,
               positive: str | None = None, target: str | None = None) -> Split:
    """Global temporal split: train on [start, train_end), evaluate on [train_end, eval_end).

    positive : boolean column marking events that count as a real interaction, used as
               training signal and as eval targets (Yambda: "is_full_listen", i.e. not a
               skip). None = every event counts (Last.fm).
    target   : extra boolean column an eval event must satisfy to be a target
               (Yambda: "is_organic", i.e. the user found it themself).
    "Already heard" is always checked against ALL train events, skips included.
    All filtering statistics come from the train part only (no peeking at the future).
    """
    p = _Prepared(events, train_end, eval_end, positive, target)

    # 1) item vocabulary: tracks with enough distinct listeners in train
    item_users = p.item_listeners()
    item_keys = np.sort(item_users.index[item_users >= min_item_users].to_numpy())
    item_index = pd.Index(item_keys)
    item_idx = item_index.get_indexer(p.tracks)              # -1 = not in vocabulary
    rows = p.in_train & p.positive & (item_idx >= 0)

    # 2) users: enough distinct vocabulary tracks in train to learn a taste from
    pairs = np.unique(p.keys[rows])
    users_u, n_items = np.unique(_unpack(pairs)[0], return_counts=True)
    user_ints = users_u[n_items >= min_user_items]            # sorted
    user_index = pd.Index(user_ints)
    user_idx = user_index.get_indexer(p.users)
    rows &= user_idx >= 0

    # 3) encoded training data
    train_events = pd.DataFrame({
        "user_idx": user_idx[rows].astype("int32"),
        "item_idx": item_idx[rows].astype("int32"),
        "ts": events["ts"].to_numpy()[rows],
    })
    pair_keys, plays = np.unique(_pack(user_idx[rows], item_idx[rows]), return_counts=True)
    u, i = _unpack(pair_keys)
    train_counts = pd.DataFrame({"user_idx": u.astype("int32"), "item_idx": i.astype("int32"),
                                 "plays": plays.astype("int32")})

    # 4) targets: discoveries in the eval window, restricted to known users and items
    targets, n_new = _encode_targets(p, p.target, user_index, item_index)

    user_ids = user_ints if p.user_cats is None else p.user_cats[user_ints]
    stats = {
        "n_users": len(user_ids),
        "n_items": len(item_keys),
        "train_interactions": len(train_events),
        "train_pairs": len(train_counts),
        "density": len(train_counts) / (len(user_ids) * len(item_keys)),
        "eval_new_pairs_all": n_new,
        "eval_new_pairs_reachable": len(targets),
        "eval_users_with_targets": int(targets["user_idx"].nunique()),
    }
    split = Split(train_end, eval_end, train_events, train_counts, targets, user_ids, item_keys, stats)
    split._prepared = (p, user_index, item_index)      # reused by extra_targets, not saved
    return split


def _encode_targets(p: _Prepared, mask: np.ndarray, user_index: pd.Index, item_index: pd.Index):
    users, tracks, plays = p.new_pairs(mask)
    u, i = user_index.get_indexer(users), item_index.get_indexer(tracks)
    keep = (u >= 0) & (i >= 0)
    df = pd.DataFrame({"user_idx": u[keep], "item_idx": i[keep], "plays": plays[keep]}).astype("int32")
    return df, len(users)


def extra_targets(events: pd.DataFrame, split: Split,
                  positive: str | None = None, target: str | None = None) -> pd.DataFrame:
    """Alternative eval targets for a split just built by make_split, in its index space.

    Same rules (new = never heard in train, known users/items only), different event
    filter. Used for secondary metrics, e.g. Yambda discoveries that came through the
    platform's recommendations as well as organic ones.
    """
    p, user_index, item_index = split._prepared
    mask = np.ones(len(events), bool) if positive is None else events[positive].to_numpy()
    if target is not None:
        mask &= events[target].to_numpy()
    return _encode_targets(p, mask, user_index, item_index)[0]
