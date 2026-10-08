"""Yambda-50M (Yandex Music, released 2025): loading and building the event table.

Raw files in data/raw/yambda/ (flat/50m/ on Hugging Face):
  listens.parquet   uid, timestamp, item_id, is_organic, played_ratio_pct, track_length_seconds
  likes / dislikes / unlikes / undislikes.parquet   uid, timestamp, item_id, is_organic
  artist_item_mapping.parquet, album_item_mapping.parquet   (repo root, optional)

Facts that shape this module (checked before writing it):
  * The files are already compact Parquet with unsigned-int columns. Unlike Last.fm,
    there is no text to parse, so no "stage 0" conversion is needed.
  * `timestamp` = seconds since the start of observation, binned to 5 s; max = 26,000,000
    (about 301 days). There are no calendar dates.
  * `is_organic` = 1 if the user chose the track themself, 0 if the listen came from the
    service's recommendations (e.g. an endless personalised radio).
  * `played_ratio_pct` = share of the track played; values above 100 exist (seeking or
    replaying within one event), so we clip at 100.
"""
from __future__ import annotations

import pandas as pd

from musicrec import config

DAY = config.DAY


def load_listens(columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(config.YAMBDA_RAW_DIR / "listens.parquet", columns=columns)


def load_feedback(kind: str) -> pd.DataFrame:
    """kind in {"likes", "dislikes", "unlikes", "undislikes"}"""
    return pd.read_parquet(config.YAMBDA_RAW_DIR / f"{kind}.parquet")


def load_item_artist() -> pd.Series | None:
    """item_id -> artist_id, or None if the mapping file was not downloaded."""
    path = config.YAMBDA_RAW_DIR / "artist_item_mapping.parquet"
    if not path.exists():
        return None
    m = pd.read_parquet(path)
    # An item can be credited to several artists; keep the first as its main artist.
    return m.drop_duplicates("item_id").set_index("item_id")["artist_id"]


def build_events(min_played_ratio: int = 50) -> pd.DataFrame:
    """The standard event table that preprocess.make_split expects.

    Columns: user_id, ts (int seconds), track_key, is_organic (bool),
             played_ratio (0-100), is_full_listen (bool)

    ALL listens are kept, including skips: a skipped track still counts as "already
    heard" when we decide what is a new discovery later. Whether a skip counts as a
    positive signal is decided in make_split via the `positive` column.
    """
    df = load_listens(["uid", "timestamp", "item_id", "is_organic", "played_ratio_pct"])
    df = df.rename(columns={"uid": "user_id", "timestamp": "ts", "item_id": "track_key"})
    df["ts"] = df["ts"].astype("int64")                  # int64: safe to subtract later
    df["is_organic"] = df["is_organic"].astype(bool)
    df["played_ratio"] = df.pop("played_ratio_pct").clip(upper=100).astype("uint8")
    df["is_full_listen"] = df["played_ratio"] >= min_played_ratio
    return df.sort_values(["user_id", "ts"], kind="stable").reset_index(drop=True)
