"""Two-stage recommendation: candidates from several models, re-ranked by LightGBM.

Stage 1 (retrieval): each source model (iALS, SASRec, popularity) proposes its top-K unheard
tracks per user. The union is the candidate set.
Stage 2 (ranking): a LightGBM LambdaRank model scores every (user, candidate) pair from
features: each source's score and rank, recent popularity and trend, and how familiar the
user is with the track's artist. Labels = the candidate is one of the user's discoveries.

Why: iALS and SASRec hit different users (oracle hit rate 15% -> 21% on test). A ranker can
learn when to trust which source.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp

from musicrec.evaluation import interaction_matrix, top_k
from musicrec.preprocess import Split, to_cutoff

DAY = 86_400


def _pack(u, i):
    return (np.asarray(u, np.int64) << 32) | np.asarray(i, np.int64)


# ------------------------------------------------------------------ stage 1: candidates

def eval_users(split: Split) -> np.ndarray:
    """Users with at least one discovery: the only ones that matter for training and scoring."""
    return np.unique(split.targets["user_idx"].to_numpy())


def candidates_from_model(model, split: Split, k: int, users: np.ndarray | None = None,
                          batch_size: int = 256) -> pd.DataFrame:
    """Top-k unheard tracks per user from any model with .score(users). Columns: user_idx, item_idx, score, rank."""
    users = eval_users(split) if users is None else users
    seen = interaction_matrix(split, binary=True)
    parts = []
    for start in range(0, len(users), batch_size):
        b = users[start:start + batch_size]
        s = np.array(model.score(b), dtype=np.float32)
        r, c = seen[b].nonzero()
        s[r, c] = -np.inf
        top = top_k(s, k)
        parts.append(pd.DataFrame({
            "user_idx": np.repeat(b, k).astype(np.int32),
            "item_idx": top.ravel().astype(np.int32),
            "score": np.take_along_axis(s, top, axis=1).ravel(),
            "rank": np.tile(np.arange(1, k + 1, dtype=np.int16), len(b)),
        }))
    return pd.concat(parts, ignore_index=True)


def merge_candidates(sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Outer-join sources on (user, item). A track missing from a source gets NaN score/rank there."""
    out = None
    for name, df in sources.items():
        d = df[["user_idx", "item_idx", "score", "rank"]].rename(
            columns={"score": f"score_{name}", "rank": f"rank_{name}"})
        out = d if out is None else out.merge(d, on=["user_idx", "item_idx"], how="outer")
    for name in sources:
        out[f"rank_{name}"] = out[f"rank_{name}"].astype("float32")
    return out.sort_values(["user_idx", "item_idx"]).reset_index(drop=True)


def add_labels(cands: pd.DataFrame, split: Split, targets: pd.DataFrame | None = None) -> pd.DataFrame:
    t = split.targets if targets is None else targets
    tk = np.unique(_pack(t["user_idx"], t["item_idx"]))
    cands["label"] = np.isin(_pack(cands["user_idx"], cands["item_idx"]), tk).astype(np.int8)
    return cands


def candidate_recall(cands: pd.DataFrame, split: Split, sources: list[str]) -> pd.Series:
    """Share of all reachable discoveries that made it into each source's list / the union.
    This is the ceiling for any re-ranking."""
    n_targets = len(split.targets)
    out = {s: cands.loc[cands[f"rank_{s}"].notna(), "label"].sum() / n_targets for s in sources}
    out["union"] = cands["label"].sum() / n_targets
    out["candidates per user"] = len(cands) / cands["user_idx"].nunique()
    return pd.Series(out)


# ------------------------------------------------------------------ stage 2: features

def add_features(cands: pd.DataFrame, split: Split, item_artist: pd.Series, sources: list[str]) -> pd.DataFrame:
    ev = split.train_events
    u_all, i_all, ts = ev["user_idx"].to_numpy(), ev["item_idx"].to_numpy(), ev["ts"].to_numpy()
    cut = to_cutoff(split.train_end)
    n_items, n_users = split.n_items, split.n_users

    def listeners(mask):                                  # distinct users per item
        pairs = np.unique(_pack(u_all[mask], i_all[mask]))
        return np.bincount((pairs & 0xFFFFFFFF).astype(np.int64), minlength=n_items)

    recent7, recent30 = ts >= cut - 7 * DAY, ts >= cut - 30 * DAY
    pop_all = listeners(np.ones(len(ts), bool))
    pop_7d, pop_30d = listeners(recent7), listeners(recent30)
    first_seen = pd.Series(ts).groupby(i_all).min().reindex(range(n_items)).to_numpy()

    # artists
    codes = pd.factorize(item_artist.reindex(split.item_keys))[0]
    n_art = codes.max() + 1
    a_all = codes[i_all]
    ok = a_all >= 0
    UA = sp.csr_matrix((np.ones(ok.sum(), np.float32), (u_all[ok], a_all[ok])), shape=(n_users, n_art))
    ok7 = ok & recent7
    UA7 = sp.csr_matrix((np.ones(ok7.sum(), np.float32), (u_all[ok7], a_all[ok7])), shape=(n_users, n_art))
    art_pairs7 = np.unique(_pack(u_all[ok7], a_all[ok7]))
    artist_pop_7d = np.bincount((art_pairs7 & 0xFFFFFFFF).astype(np.int64), minlength=n_art)

    # users
    user_events = np.bincount(u_all, minlength=n_users)
    user_events_7d = np.bincount(u_all[recent7], minlength=n_users)
    user_last = pd.Series(ts).groupby(u_all).max().reindex(range(n_users)).to_numpy()

    u, i = cands["user_idx"].to_numpy(), cands["item_idx"].to_numpy()
    a = codes[i]
    has_a = a >= 0
    a_safe = np.where(has_a, a, 0)

    f = cands
    f["item_pop_all"] = np.log1p(pop_all[i])
    f["item_pop_7d"] = np.log1p(pop_7d[i])
    f["item_pop_30d"] = np.log1p(pop_30d[i])
    f["item_trend"] = (pop_7d[i] + 1) / (pop_30d[i] + 1)          # share of last-month listeners that are from the last week
    f["item_age_days"] = (cut - first_seen[i]) / DAY
    f["artist_pop_7d"] = np.where(has_a, np.log1p(artist_pop_7d[a_safe]), np.nan)
    f["user_artist_plays"] = np.where(has_a, np.log1p(np.asarray(UA[u, a_safe]).ravel()), 0.0)
    f["user_artist_plays_7d"] = np.where(has_a, np.log1p(np.asarray(UA7[u, a_safe]).ravel()), 0.0)
    f["user_events_log"] = np.log1p(user_events[u])
    f["user_events_7d_log"] = np.log1p(user_events_7d[u])
    f["user_days_since_last"] = (cut - user_last[u]) / DAY
    f["n_sources"] = sum(f[f"rank_{s}"].notna().astype(np.int8) for s in sources)
    return f


ORGANIC_FEATURES = ["user_organic_share", "item_organic_share", "item_organic_listeners_7d", "item_skip_rate"]


def add_organic_features(cands: pd.DataFrame, split: Split, events: pd.DataFrame,
                         smoothing: float = 20.0) -> pd.DataFrame:
    """Features from signals the split's train events don't keep: organic flag and skips.

    events: the full Yambda event table (musicrec.yambda.build_events), raw IDs.
    user_organic_share       share of the user's listens they chose themselves
    item_organic_share       share of the track's listens that were organic (shrunk to the global mean)
    item_organic_listeners_7d  distinct users who found the track themselves and played it, last 7 days
    item_skip_rate           share of the track's listens played < 50% (shrunk to the global mean)
    Only events before the split's train_end are used.
    """
    cut = to_cutoff(split.train_end)
    ev = events[events["ts"] < cut]
    u = pd.Index(split.user_ids).get_indexer(ev["user_id"].to_numpy())
    i = pd.Index(split.item_keys).get_indexer(ev["track_key"].to_numpy())
    org = ev["is_organic"].to_numpy()
    skip = ~ev["is_full_listen"].to_numpy()

    ku = u >= 0
    n_u = np.bincount(u[ku], minlength=split.n_users)
    user_org = np.bincount(u[ku], weights=org[ku], minlength=split.n_users) / np.maximum(n_u, 1)

    ki = i >= 0
    n_i = np.bincount(i[ki], minlength=split.n_items)
    g_org, g_skip = org.mean(), skip.mean()
    item_org = (np.bincount(i[ki], weights=org[ki], minlength=split.n_items) + smoothing * g_org) / (n_i + smoothing)
    item_skip = (np.bincount(i[ki], weights=skip[ki], minlength=split.n_items) + smoothing * g_skip) / (n_i + smoothing)

    rec = ki & org & ~skip & (ev["ts"].to_numpy() >= cut - 7 * DAY) & (u >= 0)
    pairs = np.unique(_pack(u[rec], i[rec]))
    org_listeners_7d = np.bincount((pairs & 0xFFFFFFFF).astype(np.int64), minlength=split.n_items)

    cu, ci = cands["user_idx"].to_numpy(), cands["item_idx"].to_numpy()
    cands["user_organic_share"] = user_org[cu]
    cands["item_organic_share"] = item_org[ci]
    cands["item_organic_listeners_7d"] = np.log1p(org_listeners_7d[ci])
    cands["item_skip_rate"] = item_skip[ci]
    return cands


def feature_columns(sources: list[str], organic: bool = False) -> list[str]:
    cols = ([f"score_{s}" for s in sources] + [f"rank_{s}" for s in sources] +
            ["item_pop_all", "item_pop_7d", "item_pop_30d", "item_trend", "item_age_days", "artist_pop_7d",
             "user_artist_plays", "user_artist_plays_7d", "user_events_log", "user_events_7d_log",
             "user_days_since_last", "n_sources"])
    return cols + (ORGANIC_FEATURES if organic else [])


def graded_relevance(cands: pd.DataFrame) -> pd.Series:
    """0 = not a discovery, 1 = discovery by an artist the user knows, 2 = discovery by a NEW artist.
    With LambdaRank's label_gain = [0, 1, w], a new-artist hit counts w times a familiar one."""
    new_artist = cands["user_artist_plays"].fillna(0) == 0
    return (cands["label"] * np.where(new_artist, 2, 1)).astype(np.int8)


# ------------------------------------------------------------------ scoring wrappers

class PrecomputedScores:
    """Turn a (user_idx, item_idx, score) table into a model the standard evaluator accepts.
    Tracks outside the table get a very low score, so only candidates can be recommended."""

    def __init__(self, df: pd.DataFrame, n_items: int, name: str, col: str = "score"):
        df = df.sort_values("user_idx")
        self.users = df["user_idx"].to_numpy()
        self.items = df["item_idx"].to_numpy()
        self.vals = df[col].to_numpy(np.float32)
        self.n_items, self.name = n_items, name

    def score(self, users: np.ndarray) -> np.ndarray:
        out = np.full((len(users), self.n_items), -1e9, dtype=np.float32)
        lo = np.searchsorted(self.users, users, side="left")
        hi = np.searchsorted(self.users, users, side="right")
        for row, (a, b) in enumerate(zip(lo, hi)):
            out[row, self.items[a:b]] = self.vals[a:b]
        return out


def rrf_scores(cands: pd.DataFrame, sources: list[str], k: int = 60) -> pd.Series:
    """Reciprocal rank fusion: sum of 1 / (k + rank) over sources. A strong, tuning-free blend."""
    return sum((1.0 / (k + cands[f"rank_{s}"])).fillna(0.0) for s in sources)


# ------------------------------------------------------------------ LightGBM ranker

def fit_ranker(df: pd.DataFrame, features: list[str], seed: int = 42, label_col: str = "label", **params):
    """LambdaRank on users with >= 1 positive candidate; 10% of those users held out for early stopping.
    label_col can be graded (see graded_relevance) together with params label_gain=[0, 1, w]."""
    from lightgbm import LGBMRanker, early_stopping

    pos_users = df.groupby("user_idx")[label_col].max()
    users = pos_users.index[pos_users > 0].to_numpy()
    rng = np.random.default_rng(seed)
    hold = set(rng.choice(users, size=max(1, len(users) // 10), replace=False))
    d = df[df["user_idx"].isin(users)].sort_values("user_idx")
    is_hold = d["user_idx"].isin(hold).to_numpy()
    tr, va = d[~is_hold], d[is_hold]

    cfg = dict(objective="lambdarank", n_estimators=1000, learning_rate=0.05, num_leaves=63,
               min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
               random_state=seed, verbose=-1)
    cfg.update(params)
    model = LGBMRanker(**cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                  # lightgbm API deprecation notices
        model.fit(tr[features], tr[label_col], group=tr.groupby("user_idx", sort=True).size().to_numpy(),
                  eval_set=[(va[features], va[label_col])],
                  eval_group=[va.groupby("user_idx", sort=True).size().to_numpy()],
                  eval_at=[10], callbacks=[early_stopping(50, verbose=False)])
    return model


def cross_val_ranker(df: pd.DataFrame, features: list[str], n_folds: int = 2, seed: int = 42, **params) -> pd.Series:
    """Out-of-fold ranker scores, folds split by USER (same time window, disjoint users), so no
    user's own labels are used to score that user."""
    users = df["user_idx"].unique()
    rng = np.random.default_rng(seed)
    fold_of = pd.Series(rng.integers(0, n_folds, len(users)), index=users)
    fold = fold_of.reindex(df["user_idx"]).to_numpy()
    pred = np.zeros(len(df), dtype=np.float64)
    for k in range(n_folds):
        model = fit_ranker(df[fold != k], features, seed=seed + k, **params)
        pred[fold == k] = model.predict(df.loc[fold == k, features])
    return pd.Series(pred, index=df.index)


# ------------------------------------------------------------------ per-user new-artist weight

def user_exploration(split: Split, events: pd.DataFrame, item_artist: pd.Series,
                     days: int = 60, smoothing: float = 20.0) -> np.ndarray:
    """Per user (indexed by user_idx): share of recent discoveries that were by a new artist.

    A discovery = the user's first ever play of a track, organic and played >= 50%, in the
    `days` before train_end. It is a new-artist discovery if that play was also the user's
    first play of the artist. Shrunk toward the global share, so users with few discoveries
    get roughly the average. Only events before train_end are used.
    """
    cut = to_cutoff(split.train_end)
    ev = events[events["ts"] < cut]                        # sorted by user, then time
    u = pd.Index(split.user_ids).get_indexer(ev["user_id"].to_numpy())
    keep = u >= 0
    ev, u = ev[keep], u[keep].astype(np.int64)
    t = pd.factorize(ev["track_key"])[0].astype(np.int64)
    a = pd.factorize(item_artist.reindex(ev["track_key"].to_numpy()).to_numpy())[0].astype(np.int64)

    # np.unique returns the first occurrence, which is the earliest play since events are time-sorted
    first_track = np.zeros(len(ev), bool)
    first_track[np.unique(_pack(u, t), return_index=True)[1]] = True
    first_artist = np.zeros(len(ev), bool)
    has_a = a >= 0
    idx_a = np.flatnonzero(has_a)
    first_artist[idx_a[np.unique(_pack(u[has_a], a[has_a]), return_index=True)[1]]] = True

    disc = (first_track & has_a & ev["is_organic"].to_numpy() & ev["is_full_listen"].to_numpy()
            & (ev["ts"].to_numpy() >= cut - days * DAY))
    n_disc = np.bincount(u[disc], minlength=split.n_users)
    n_new = np.bincount(u[disc & first_artist], minlength=split.n_users)
    g = n_new.sum() / max(n_disc.sum(), 1)
    return (n_new + smoothing * g) / (n_disc + smoothing)


def per_user_blend(cands: pd.DataFrame, col_a: str, col_b: str, lam: np.ndarray) -> np.ndarray:
    """(1 - lam[u]) * pct_rank(col_a) + lam[u] * pct_rank(col_b), ranks taken within each user.

    lam: one value per user_idx. lam = 0 reproduces col_a's order, lam = 1 col_b's.
    """
    g = cands.groupby("user_idx")
    ra, rb = g[col_a].rank(pct=True).to_numpy(), g[col_b].rank(pct=True).to_numpy()
    l = lam[cands["user_idx"].to_numpy()]
    return (1 - l) * ra + l * rb


def spread_lambda(explore: np.ndarray, users: np.ndarray, mean: float) -> np.ndarray:
    """Per-user blend weight with a fixed average: low explorers get less than `mean`, high
    explorers more. lam = mean + (pct - 0.5) * 2 * min(mean, 1 - mean), pct = the user's
    exploration percentile among `users`. Users outside `users` get `mean`."""
    lam = np.full(len(explore), mean)
    pct = (pd.Series(explore[users]).rank().to_numpy() - 0.5) / len(users)   # mean exactly 0.5
    lam[users] = mean + (pct - 0.5) * 2 * min(mean, 1 - mean)
    return lam
