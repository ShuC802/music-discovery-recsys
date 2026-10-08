"""Item-based collaborative filtering ("物以类聚").

Two tracks are similar if the same users listen to them. Each track is a vector over
users; similarity is the cosine between those vectors. For each track we keep only its
k most similar tracks (the "neighbors"), then a user's score for track j is

    score(u, j) = sum over tracks i the user played of  weight(u, i) * sim(i, j)

i.e. "tracks that are similar to many things you already like".

Shrinkage: plain cosine gives two obscure tracks that share a single listener a
similarity close to 1. We add a constant to the denominator,
    sim = dot / (|a| |b| + shrink * typical_norm_product)
so similarities backed by little evidence are damped. `shrink` is expressed relative to
the median norm product, which makes it comparable across weighting schemes and time
decay (decayed weights are much smaller, so an absolute constant would mean something
different every time).

Implementation: sparse x sparse products, one chunk of items at a time. Only users who
actually played a track contribute to its row, so we never materialise the full
users x items matrix densely (8.6k x 160k floats would be 5.5 GB on Yambda).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from musicrec.models.weights import interaction_weights
from musicrec.preprocess import Split


class ItemKNN:
    def __init__(self, k: int = 100, shrink: float = 0.0, weighting: str = "log",
                 half_life_days: float | None = None, chunk: int = 256):
        self.k, self.shrink, self.weighting = k, shrink, weighting
        self.half_life_days, self.chunk = half_life_days, chunk
        hl = f", half-life {half_life_days}d" if half_life_days else ""
        self.name = f"ItemKNN(k={k}, shrink={shrink}, {weighting}{hl})"

    def fit(self, split: Split) -> "ItemKNN":
        X = interaction_weights(split, self.weighting, self.half_life_days).tocsr()
        self.X_ = X
        XT = X.T.tocsr()                                              # items x users
        norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=0)).ravel()).astype(np.float32)
        typical = float(np.median(norms[norms > 0])) ** 2
        shrink_abs = np.float32(self.shrink * typical + 1e-9)

        n, k = split.n_items, self.k
        rows, cols, vals = [], [], []
        for start in range(0, n, self.chunk):
            stop = min(start + self.chunk, n)
            sims = (XT[start:stop] @ X).toarray()                    # (chunk, n_items) dot products
            sims /= np.outer(norms[start:stop], norms) + shrink_abs  # -> (shrunk) cosine, in place
            local = np.arange(stop - start)
            sims[local, local + start] = 0.0                         # an item is not its own neighbor
            nb = np.argpartition(-sims, k, axis=1)[:, :k]           # top-k neighbors, unsorted
            rows.append(np.repeat(np.arange(start, stop), k))
            cols.append(nb.ravel())
            vals.append(np.take_along_axis(sims, nb, axis=1).ravel())

        self.S_ = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                                shape=(n, n), dtype=np.float32)
        self.S_.eliminate_zeros()
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        return (self.X_[users] @ self.S_).toarray()
