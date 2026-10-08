"""BPR matrix factorization, written from scratch in PyTorch (Rendle et al. 2009).

Same model family as iALS (a user vector and an item vector, score = dot product), but a
different way of learning it:

  iALS : regress every cell of the users x items matrix towards 1 (played) or 0 (not),
         solved in closed form with alternating least squares.
  BPR  : a *ranking* loss. Take a (user, track they played, random track) triple and push
         score(played) above score(random):   loss = -log sigmoid(s_pos - s_neg).
         Learned with stochastic gradient descent on mini-batches.

Pieces of a PyTorch training setup you will see here, and again in SASRec later:
  nn.Module + nn.Embedding  (the parameters), forward()  (how a score is computed),
  a loss, an optimizer (Adam), a mini-batch loop, evaluation + early stopping.

Recency: positives are sampled with probability proportional to their time-decayed weight
(same `interaction_weights` as iALS), so a half-life of 3 days means the model mostly
trains on what users played in the last week or two.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from musicrec.models.weights import interaction_weights
from musicrec.preprocess import Split


class MFModule(nn.Module):
    """The parameters: one embedding row per user, one per item, plus an item bias."""

    def __init__(self, n_users: int, n_items: int, dim: int):
        super().__init__()
        self.user = nn.Embedding(n_users, dim)
        self.item = nn.Embedding(n_items, dim)
        self.item_bias = nn.Embedding(n_items, 1)      # lets popular items score higher overall
        nn.init.normal_(self.user.weight, std=0.01)
        nn.init.normal_(self.item.weight, std=0.01)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """users: (B,)  items: (B, K)  ->  scores (B, K)"""
        u = self.user(users).unsqueeze(1)               # (B, 1, d)
        return (u * self.item(items)).sum(-1) + self.item_bias(items).squeeze(-1)


class BPR:
    def __init__(self, dim: int = 128, n_neg: int = 4, lr: float = 1e-3, reg: float = 1e-5,
                 batch_size: int = 8192, epochs: int = 15, eval_every: int = 3, patience: int = 2,
                 samples_per_epoch: int | None = None,
                 weighting: str = "log", half_life_days: float | None = None,
                 early_stopping: bool = True, seed: int = 42, device: str | None = None):
        self.dim, self.n_neg, self.lr, self.reg = dim, n_neg, lr, reg
        self.batch_size, self.epochs, self.eval_every, self.patience = batch_size, epochs, eval_every, patience
        self.samples_per_epoch = samples_per_epoch
        self.weighting, self.half_life_days = weighting, half_life_days
        self.early_stopping, self.seed = early_stopping, seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        hl = f", half-life {half_life_days}d" if half_life_days else ""
        self.name = f"BPR(d={dim}, neg={n_neg}, lr={lr}, reg={reg}{hl})"

    def params(self) -> dict:
        return {"dim": self.dim, "n_neg": self.n_neg, "lr": self.lr, "reg": self.reg,
                "batch_size": self.batch_size, "samples_per_epoch": self.samples_per_epoch,
                "weighting": self.weighting,
                "half_life_days": self.half_life_days}

    # ------------------------------------------------------------------ training
    def fit(self, split: Split, verbose: bool = True) -> "BPR":
        from musicrec.evaluation import evaluate          # local import avoids a cycle

        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)

        # positives: every (user, item) pair in train, weighted by (decayed) listening
        X = interaction_weights(split, self.weighting, self.half_life_days).tocoo()
        pos_users, pos_items = X.row.astype(np.int64), X.col.astype(np.int64)
        p = X.data.astype(np.float64)
        p /= p.sum()
        # With weighted sampling an "epoch" is just a budget of sampled triples. Default:
        # one per train pair. A smaller budget = more frequent evaluation, faster feedback.
        n_samples = self.samples_per_epoch or len(pos_users)

        self.module_ = MFModule(split.n_users, split.n_items, self.dim).to(self.device)
        opt = torch.optim.Adam(self.module_.parameters(), lr=self.lr)

        self.history_, best, best_state, bad = [], -1.0, None, 0
        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            self.module_.train()
            idx = rng.choice(len(pos_users), size=n_samples, p=p)     # weighted sampling of positives
            total = 0.0
            for start in range(0, n_samples, self.batch_size):
                b = idx[start:start + self.batch_size]
                u = torch.from_numpy(pos_users[b]).to(self.device)
                i_pos = torch.from_numpy(pos_items[b]).to(self.device).unsqueeze(1)          # (B, 1)
                i_neg = torch.randint(0, split.n_items, (len(b), self.n_neg), device=self.device)  # (B, K)

                s_pos = self.module_(u, i_pos)                  # (B, 1)
                s_neg = self.module_(u, i_neg)                  # (B, K)
                loss = -F.logsigmoid(s_pos - s_neg).mean()      # BPR: played should outrank random
                # L2 penalty on the embeddings used in this batch (keeps vectors small)
                l2 = (self.module_.user(u).pow(2).sum() + self.module_.item(i_pos).pow(2).sum()
                      + self.module_.item(i_neg).pow(2).sum()) / len(b)
                loss = loss + self.reg * l2

                opt.zero_grad()
                loss.backward()                                 # gradients via autograd
                opt.step()                                      # Adam update
                total += loss.item() * len(b)

            row = {"epoch": epoch, "loss": total / n_samples, "secs": time.time() - t0}
            if self.early_stopping and (epoch % self.eval_every == 0 or epoch == self.epochs):
                row["val_recall@10"] = evaluate(self, split)["recall@10"]
                if row["val_recall@10"] > best:
                    best, best_state, bad = row["val_recall@10"], copy.deepcopy(self.module_.state_dict()), 0
                    self.best_epoch_ = epoch
                else:
                    bad += 1
            self.history_.append(row)
            if verbose:
                extra = f"  val recall@10={row['val_recall@10']:.4f}" if "val_recall@10" in row else ""
                print(f"epoch {epoch:>2}  loss={row['loss']:.4f}  ({row['secs']:.0f}s){extra}")
            if self.early_stopping and bad >= self.patience:
                print(f"early stop: no improvement in {self.patience} evaluations")
                break

        if best_state is not None:                              # keep the best checkpoint
            self.module_.load_state_dict(best_state)
        else:
            self.best_epoch_ = self.epochs
        return self

    # ------------------------------------------------------------------ scoring
    @torch.no_grad()
    def score(self, users: np.ndarray) -> np.ndarray:
        self.module_.eval()
        u = self.module_.user.weight[torch.tensor(users, dtype=torch.long, device=self.device)]
        s = u @ self.module_.item.weight.T + self.module_.item_bias.weight.T
        return s.float().cpu().numpy()
