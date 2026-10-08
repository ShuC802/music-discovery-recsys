"""Offline evaluation shared by every model in the project (baselines, CF, PyTorch).

Model contract (duck typing, nothing to inherit):
    model.fit(split)            -> model
    model.score(user_idx_array) -> array of shape (n_users_in_batch, n_items), higher = better

The evaluator, not the model, removes items the user already played in train,
so every model is judged on exactly the same task: discover new tracks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from musicrec.preprocess import Split


def interaction_matrix(split: Split, binary: bool = False) -> sp.csr_matrix:
    """users x items sparse matrix of train play counts (or 1/0 if binary)."""
    c = split.train_counts
    data = np.ones(len(c), np.float32) if binary else c["plays"].to_numpy(np.float32)
    return sp.csr_matrix((data, (c["user_idx"], c["item_idx"])), shape=(split.n_users, split.n_items))


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k highest scores per row, best first.

    argpartition finds the top k in O(n) without fully sorting 70k items;
    then we only sort those k.
    """
    part = np.argpartition(-scores, k, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1)
    return np.take_along_axis(part, order, axis=1)


def evaluate(model, split: Split, ks=(10, 50), min_plays: int = 1, batch_size: int = 256,
             targets: pd.DataFrame | None = None, return_per_user: bool = False) -> dict:
    """Ranking metrics averaged over users who have at least one target.

    recall@k    : hits / min(k, n_targets)   (normalized so a perfect model scores 1)
    ndcg@k      : like recall, but a hit at rank 1 is worth more than a hit at rank k
    hit_rate@k  : share of users with at least one hit (easiest to explain to a PM)
    coverage@k  : share of the catalog recommended to at least one user
    avg_pop@k   : mean number of train listeners of recommended items (popularity bias)
    min_plays   : 2 = only count discoveries the user played at least twice ("stuck")
    targets     : evaluate on a subset of split.targets instead (e.g. one user segment)
    return_per_user : also return result["per_user"], one row per evaluated user, for
                  significance tests (see paired_bootstrap)
    """
    tgt = split.targets if targets is None else targets
    tgt = tgt[tgt["plays"] >= min_plays]
    targets = tgt.groupby("user_idx")["item_idx"].apply(np.asarray)
    users = targets.index.to_numpy()

    seen = interaction_matrix(split, binary=True)
    item_listeners = np.asarray(seen.sum(axis=0)).ravel()
    k_max = max(ks)
    discounts = 1.0 / np.log2(np.arange(2, k_max + 2))

    per_user = {f"{m}@{k}": [] for k in ks for m in ("recall", "ndcg", "hit_rate")}
    recommended = {k: np.zeros(split.n_items, dtype=bool) for k in ks}
    pop = {k: [] for k in ks}

    for start in range(0, len(users), batch_size):
        batch = users[start:start + batch_size]
        scores = np.array(model.score(batch), dtype=np.float32)   # copy: we modify it
        rows, cols = seen[batch].nonzero()
        scores[rows, cols] = -np.inf                                # mask already-played items
        top = top_k(scores, k_max)

        for i, u in enumerate(batch):
            hits = np.isin(top[i], targets[u])
            for k in ks:
                h, n_rel = hits[:k], min(k, len(targets[u]))
                per_user[f"recall@{k}"].append(h.sum() / n_rel)
                per_user[f"hit_rate@{k}"].append(float(h.any()))
                per_user[f"ndcg@{k}"].append((h * discounts[:k]).sum() / discounts[:n_rel].sum())
        for k in ks:
            recommended[k][top[:, :k].ravel()] = True
            pop[k].append(item_listeners[top[:, :k]].mean(axis=1))

    result = {name: float(np.mean(v)) for name, v in per_user.items()}
    for k in ks:
        result[f"coverage@{k}"] = float(recommended[k].mean())
        result[f"avg_pop@{k}"] = float(np.concatenate(pop[k]).mean())
    result["n_eval_users"] = len(users)
    if return_per_user:
        result["per_user"] = pd.DataFrame({"user_idx": users, **{n: np.asarray(v) for n, v in per_user.items()}})
    return result


def paired_bootstrap(per_user_a: pd.DataFrame, per_user_b: pd.DataFrame, metric: str = "recall@10",
                     n_boot: int = 2000, seed: int = 42) -> dict:
    """Is model B really better than model A, or is the gap noise?

    Both models are scored on the SAME users, so we look at the per-user difference
    b - a and resample users with replacement many times. If the 95% interval of the
    mean difference excludes 0, the gap is unlikely to come from which users happened
    to be in the evaluation window.
    """
    m = per_user_a[["user_idx", metric]].merge(per_user_b[["user_idx", metric]], on="user_idx",
                                               suffixes=("_a", "_b"))
    d = (m[f"{metric}_b"] - m[f"{metric}_a"]).to_numpy()
    rng = np.random.default_rng(seed)
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    base = m[f"{metric}_a"].mean()
    return {
        "a": base, "b": m[f"{metric}_b"].mean(),
        "diff": d.mean(), "rel_diff": d.mean() / base,
        "ci_low": np.percentile(boots, 2.5), "ci_high": np.percentile(boots, 97.5),
        "p(b<=a)": float((boots <= 0).mean()),
    }


def artist_novelty(split: Split, item_artist: pd.Series) -> pd.Series:
    """For each row of split.targets: True if the track's artist is NEW to the user.

    item_artist: Series mapping track_key -> artist (name or ID), any dataset.
    "Known" = the user played some other track by that artist during train (within the
    model vocabulary, so this slightly under-counts known artists).
    """
    artist_of_item = item_artist.reindex(split.item_keys).to_numpy()
    heard = pd.DataFrame({"user_idx": split.train_counts["user_idx"].to_numpy(),
                          "artist": artist_of_item[split.train_counts["item_idx"].to_numpy()]}).drop_duplicates()
    t = pd.DataFrame({"user_idx": split.targets["user_idx"].to_numpy(),
                      "artist": artist_of_item[split.targets["item_idx"].to_numpy()]})
    t = t.merge(heard, on=["user_idx", "artist"], how="left", indicator=True)
    return pd.Series((t["_merge"] == "left_only").to_numpy(), index=split.targets.index, name="new_artist")


class ArtistLens:
    """Which tracks belong to an artist the user already knows? Computed batch by batch.

    known(users) returns a sparse users x items matrix with a 1 where the track's artist
    appears in the user's train history. Built as (user -> artist) @ (artist -> item);
    never materialised for all users at once.
    """

    def __init__(self, split: Split, item_artist: pd.Series):
        artists = item_artist.reindex(split.item_keys)
        codes, _ = pd.factorize(artists)                 # artist -> 0..n_artists-1 (-1 = unknown)
        has = codes >= 0
        n_art = codes.max() + 1
        self.item_to_artist = sp.csr_matrix(
            (np.ones(has.sum(), np.float32), (np.flatnonzero(has), codes[has])),
            shape=(split.n_items, n_art))
        c = split.train_counts
        a = codes[c["item_idx"].to_numpy()]
        ok = a >= 0
        ua = sp.csr_matrix((np.ones(ok.sum(), np.float32), (c["user_idx"].to_numpy()[ok], a[ok])),
                           shape=(split.n_users, n_art))
        ua.data[:] = 1.0                                   # duplicates summed -> back to 0/1
        self.user_to_artist = ua
        self.artist_to_item = self.item_to_artist.T.tocsr()

    def known(self, users: np.ndarray) -> sp.csr_matrix:
        return (self.user_to_artist[users] @ self.artist_to_item).tocsr()


class ExcludeKnownArtists:
    """Wraps a model so it may only recommend tracks by artists NEW to the user.

    Used for a fair "new artist" comparison: every model gets all k slots for new artists.
    """

    def __init__(self, model, lens: ArtistLens):
        self.model, self.lens = model, lens
        self.name = f"{model.name} [new artists only]"

    def score(self, users: np.ndarray) -> np.ndarray:
        scores = np.array(self.model.score(users), dtype=np.float32)
        r, c = self.lens.known(users).nonzero()
        scores[r, c] = -np.inf
        return scores


def known_artist_share(model, split: Split, lens: ArtistLens, k: int = 10, batch_size: int = 256) -> float:
    """Share of a model's top-k slots that go to artists the user already knows."""
    users = np.unique(split.targets["user_idx"].to_numpy())
    seen = interaction_matrix(split, binary=True)
    shares = []
    for start in range(0, len(users), batch_size):
        batch = users[start:start + batch_size]
        scores = np.array(model.score(batch), dtype=np.float32)
        r, c = seen[batch].nonzero()
        scores[r, c] = -np.inf
        top = top_k(scores, k)
        known = lens.known(batch)
        shares.append(np.asarray(known[np.arange(len(batch))[:, None], top].todense()).mean(axis=1))
    return float(np.concatenate(shares).mean())
