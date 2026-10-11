"""SASRec: self-attentive sequential recommendation (Kang & McAuley 2018), in PyTorch.

iALS and BPR see a user as a *bag* of tracks (with recency only through decay weights).
SASRec sees the user as an *ordered sequence*: given the last L tracks, predict the next one.
A causal Transformer (each position may only look at earlier positions) turns the sequence
into a hidden state; the score of a candidate track is hidden_state . track_embedding.

Design choices for this project
-------------------------------
* Input: each user's train listens in time order (valid listens, vocabulary tracks only),
  with consecutive replays of the same track collapsed, truncated to the last `max_len`.
* Loss: sampled softmax cross-entropy. For every position, the true next track competes
  against `n_neg` random tracks shared across the batch. Cross-entropy trains SASRec much
  better than the original one-negative BCE (Klenitskiy & Vasilev 2023); a full softmax over
  160k tracks would not fit in GPU memory, so we sample.
* Training windows: each step picks users and a window end position in their history.
  With probability `p_last` the window is the user's most recent one; recency matters on
  this data, so we deliberately train a lot on the latest behaviour.
* Scoring: run the user's last `max_len` tracks through the model and score all tracks with
  the final position's hidden state. The evaluator then removes already-heard tracks.
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from musicrec.preprocess import Split


def build_sequences(split: Split, dedupe_consecutive: bool = True):
    """Per user (index = user_idx), oldest first: (items + 1, timestamps, is_first_listen).

    Items are shifted by +1 so that 0 can mean padding. is_first_listen marks the first time
    the user plays that track in train; those positions are "discoveries", like our targets.
    """
    ev = split.train_events.sort_values(["user_idx", "ts"], kind="stable")
    users = ev["user_idx"].to_numpy()
    items = ev["item_idx"].to_numpy().astype(np.int64) + 1
    ts = ev["ts"].to_numpy()
    if dedupe_consecutive:
        keep = np.ones(len(items), bool)
        keep[1:] = (items[1:] != items[:-1]) | (users[1:] != users[:-1])
        users, items, ts = users[keep], items[keep], ts[keep]
    first = ~pd.DataFrame({"u": users, "i": items}).duplicated().to_numpy()
    bounds = np.searchsorted(users, np.arange(split.n_users + 1))
    return [(items[a:b], ts[a:b], first[a:b]) for a, b in zip(bounds[:-1], bounds[1:])]


class SASRecModule(nn.Module):
    def __init__(self, n_items: int, dim: int, max_len: int, n_layers: int, n_heads: int, dropout: float):
        super().__init__()
        self.item = nn.Embedding(n_items + 1, dim, padding_idx=0)      # row 0 = padding
        self.pos = nn.Embedding(max_len, dim)
        layer = nn.TransformerEncoderLayer(dim, n_heads, dim_feedforward=4 * dim, dropout=dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        self.n_heads, self.dim = n_heads, dim
        self.register_buffer("causal", torch.triu(torch.ones(max_len, max_len, dtype=torch.bool), 1))
        nn.init.normal_(self.item.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """seq: (B, L) left-padded item ids  ->  hidden states (B, L, d)"""
        B, L = seq.shape
        x = self.item(seq) * math.sqrt(self.dim) + self.pos.weight[:L]
        x = self.drop(x)
        # blocked[b, i, j] = True: position i may not look at j (future, or padding).
        # Each position may always look at itself, so no row is fully blocked (avoids NaN).
        blocked = self.causal[:L, :L].unsqueeze(0) | (seq == 0).unsqueeze(1)
        blocked = blocked & ~torch.eye(L, dtype=torch.bool, device=seq.device)
        mask = blocked.repeat_interleave(self.n_heads, dim=0)            # (B * heads, L, L)
        h = self.encoder(x, mask=mask)
        return self.norm(h)


class SASRec:
    def __init__(self, dim: int = 128, max_len: int = 200, n_layers: int = 2, n_heads: int = 2,
                 dropout: float = 0.2, lr: float = 1e-3, batch_size: int = 128, n_neg: int = 2048,
                 p_last: float = 0.5, steps_per_epoch: int = 400, epochs: int = 30, patience: int = 3,
                 loss_on: str = "all", recent_days: float | None = None,
                 early_stopping: bool = True, seed: int = 42, device: str | None = None):
        self.dim, self.max_len, self.n_layers, self.n_heads, self.dropout = dim, max_len, n_layers, n_heads, dropout
        self.lr, self.batch_size, self.n_neg, self.p_last = lr, batch_size, n_neg, p_last
        self.steps_per_epoch, self.epochs, self.patience = steps_per_epoch, epochs, patience
        self.early_stopping, self.seed = early_stopping, seed
        self.loss_on, self.recent_days = loss_on, recent_days
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        extra = (", loss=first-listens" if loss_on == "first" else "") + (f", recent {recent_days}d" if recent_days else "")
        self.name = f"SASRec(d={dim}, L={max_len}, layers={n_layers}, neg={n_neg}, p_last={p_last}{extra})"

    def params(self) -> dict:
        return {k: getattr(self, k) for k in ["dim", "max_len", "n_layers", "n_heads", "dropout", "lr",
                                              "batch_size", "n_neg", "p_last", "steps_per_epoch",
                                              "loss_on", "recent_days"]}

    # ------------------------------------------------------------------ batches
    def _window(self, arr: np.ndarray, end: int, shift: int) -> np.ndarray:
        """arr[end-L+shift : end+shift], left-padded to length L."""
        L = self.max_len
        start = max(0, end - L)
        out = arr[start + shift:end + shift]
        return np.pad(out, (L - len(out), 0))

    def _batch(self, rng):
        users = rng.choice(self.train_users_, size=self.batch_size)
        inps, tgts, firsts = [], [], []
        for u in users:
            items, ts, first = self.seqs_[u]
            n = len(items)
            if rng.random() < self.p_last:
                end = n - 1                                   # most recent window
            else:
                lo = 1
                if self.recent_days:                          # only windows ending recently
                    lo = max(1, int(np.searchsorted(ts, self.recent_cut_)))
                end = n - 1 if lo >= n - 1 else int(rng.integers(lo, n))
            inps.append(self._window(items, end, 0))
            tgts.append(self._window(items, end, 1))
            firsts.append(self._window(first, end, 1))
        to = lambda a: torch.from_numpy(np.stack(a)).to(self.device)
        return to(inps), to(tgts), to(firsts)

    # ------------------------------------------------------------------ training
    def fit(self, split: Split, verbose: bool = True) -> "SASRec":
        from musicrec.evaluation import evaluate

        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.n_items_ = split.n_items
        self.seqs_ = build_sequences(split)
        self.train_users_ = np.array([u for u, s in enumerate(self.seqs_) if len(s[0]) >= 2])
        if self.recent_days:
            self.recent_cut_ = split.train_end - self.recent_days * 86_400

        self.module_ = SASRecModule(split.n_items, self.dim, self.max_len, self.n_layers,
                                    self.n_heads, self.dropout).to(self.device)
        opt = torch.optim.Adam(self.module_.parameters(), lr=self.lr)

        self.history_, best, best_state, bad = [], -1.0, None, 0
        self.best_epoch_ = self.epochs
        for epoch in range(1, self.epochs + 1):
            t0, total = time.time(), 0.0
            self.module_.train()
            for _ in range(self.steps_per_epoch):
                inp, tgt, first = self._batch(rng)
                h = self.module_(inp)                                    # (B, L, d)
                keep = tgt > 0                                           # real positions only
                if self.loss_on == "first":                              # learn to predict discoveries,
                    keep &= first                                        # not replays (matches our targets)
                if not keep.any():
                    continue
                h, tgt = h[keep], tgt[keep]                              # (N, d), (N,)
                emb = self.module_.item.weight
                pos_logit = (h * emb[tgt]).sum(-1, keepdim=True)         # (N, 1)
                neg = torch.randint(1, self.n_items_ + 1, (self.n_neg,), device=self.device)
                neg_logit = h @ emb[neg].T                               # (N, n_neg)
                neg_logit = neg_logit.masked_fill(neg.unsqueeze(0) == tgt.unsqueeze(1), -1e9)
                logits = torch.cat([pos_logit, neg_logit], dim=1)
                loss = F.cross_entropy(logits, torch.zeros(len(tgt), dtype=torch.long, device=self.device))
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item()

            row = {"epoch": epoch, "loss": total / self.steps_per_epoch, "secs": time.time() - t0}
            if self.early_stopping:
                row["val_recall@10"] = evaluate(self, split)["recall@10"]
                if row["val_recall@10"] > best:
                    best, best_state, bad, self.best_epoch_ = row["val_recall@10"], copy.deepcopy(self.module_.state_dict()), 0, epoch
                else:
                    bad += 1
            self.history_.append(row)
            if verbose:
                extra = f"  val recall@10={row['val_recall@10']:.4f}" if "val_recall@10" in row else ""
                print(f"epoch {epoch:>2}  loss={row['loss']:.4f}  ({row['secs']:.0f}s){extra}")
            if self.early_stopping and bad >= self.patience:
                print(f"early stop: no improvement in {self.patience} epochs")
                break
        if best_state is not None:
            self.module_.load_state_dict(best_state)
        return self

    @classmethod
    def from_checkpoint(cls, path, split: Split, device: str | None = None) -> "SASRec":
        """Rebuild a trained model (saved as {"state_dict", "params", ...}) for scoring on `split`."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        m = cls(**ckpt["params"], device=device)
        m.n_items_ = split.n_items
        m.seqs_ = build_sequences(split)
        m.module_ = SASRecModule(split.n_items, m.dim, m.max_len, m.n_layers, m.n_heads, m.dropout).to(m.device)
        m.module_.load_state_dict(ckpt["state_dict"])
        m.best_epoch_ = ckpt.get("best_epoch")
        return m

    # ------------------------------------------------------------------ scoring
    @torch.no_grad()
    def score(self, users: np.ndarray) -> np.ndarray:
        self.module_.eval()
        rows = []
        for u in users:
            inp = self.seqs_[u][0][-self.max_len:]
            rows.append(np.pad(inp, (self.max_len - len(inp), 0)))
        seq = torch.from_numpy(np.stack(rows)).to(self.device)
        h = self.module_(seq)[:, -1]                                     # last position
        scores = h @ self.module_.item.weight[1:].T                      # (b, n_items), drop padding row
        return scores.float().cpu().numpy()
