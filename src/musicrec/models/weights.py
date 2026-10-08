"""How strongly does a user like an item? Turning raw listening logs into a weight matrix.

Every CF model consumes a users x items sparse matrix. What goes in each cell is a
modeling decision:
  "binary": 1 if played at all
  "plays" : raw play count  (one obsessive replay can dominate)
  "log"   : log(1 + plays)  (more plays = more confidence, with diminishing returns)
Optional time decay: each listen is weighted by 0.5 ** (age_in_days / half_life_days),
so last month counts more than three years ago. Baselines showed recency matters a lot.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from musicrec.preprocess import Split, age_days


def interaction_weights(split: Split, weighting: str = "log",
                        half_life_days: float | None = None) -> sp.csr_matrix:
    ev = split.train_events
    if half_life_days is None:
        w = np.ones(len(ev), dtype=np.float32)
    else:
        age = age_days(split.train_end, ev["ts"])
        w = np.power(0.5, age / half_life_days).astype(np.float32)

    # Summing duplicate (user, item) entries is exactly what csr_matrix does on construction
    X = sp.csr_matrix((w, (ev["user_idx"].to_numpy(), ev["item_idx"].to_numpy())),
                      shape=(split.n_users, split.n_items), dtype=np.float32)
    X.sum_duplicates()

    if weighting == "binary":
        X.data[:] = 1.0
    elif weighting == "log":
        X.data = np.log1p(X.data)
    elif weighting != "plays":
        raise ValueError(f"unknown weighting: {weighting}")
    return X
