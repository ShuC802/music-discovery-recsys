"""iALS: matrix factorization for implicit feedback (Hu, Koren & Volinsky 2008).

Idea: give every user a vector p_u and every track a vector q_i (e.g. 128 numbers each),
such that p_u . q_i is high for tracks the user listens to. "Taste" becomes coordinates
in a shared latent space, so two tracks can be close even if no single user played both,
as long as they are linked through other users. That's the advantage over ItemKNN.

Implicit feedback: we never observe dislikes, only listens. Every cell of the
users x items matrix takes part in training, with preference 1 if played else 0 and
confidence 1 + alpha * weight(u, i). Unplayed cells get low confidence rather than being
ignored (ignoring them is what plain ALS for explicit star ratings does).

"Alternating": fix item vectors -> each user vector is a small regularised least-squares
problem; then fix users and solve items; repeat. The `implicit` library solves these
with conjugate gradient, which is much faster than exact solves.

Hyperparameters that matter (Rendle et al., "Revisiting the Performance of iALS on Item
Recommendation Benchmarks", RecSys 2022): factors, regularization, alpha, and here also
the time-decay half-life of the weights.
"""
from __future__ import annotations

import numpy as np
from implicit.als import AlternatingLeastSquares
from threadpoolctl import threadpool_limits

from musicrec.models.weights import interaction_weights
from musicrec.preprocess import Split


class IALS:
    def __init__(self, factors: int = 64, regularization: float = 0.05, alpha: float = 10.0,
                 iterations: int = 15, weighting: str = "log",
                 half_life_days: float | None = None, seed: int = 42):
        self.factors, self.regularization, self.alpha = factors, regularization, alpha
        self.iterations, self.weighting, self.half_life_days, self.seed = iterations, weighting, half_life_days, seed
        hl = f", half-life {half_life_days}d" if half_life_days else ""
        self.name = f"iALS(f={factors}, reg={regularization}, alpha={alpha}{hl})"

    def params(self) -> dict:
        return {"factors": self.factors, "regularization": self.regularization, "alpha": self.alpha,
                "iterations": self.iterations, "weighting": self.weighting,
                "half_life_days": self.half_life_days}

    def fit(self, split: Split) -> "IALS":
        X = interaction_weights(split, self.weighting, self.half_life_days)
        # implicit runs its own parallel threads. numpy's BLAS spawning threads on top of
        # that oversubscribes the CPU, so limit BLAS to 1 thread while creating AND fitting.
        with threadpool_limits(limits=1, user_api="blas"):
            model = AlternatingLeastSquares(
                factors=self.factors, regularization=self.regularization, alpha=self.alpha,
                iterations=self.iterations, random_state=self.seed,
            )
            model.fit(X, show_progress=False)
        self.user_factors_ = np.asarray(model.user_factors)
        self.item_factors_ = np.asarray(model.item_factors)
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        return self.user_factors_[users] @ self.item_factors_.T


ALS = IALS   # old name, kept so earlier notebooks still run
