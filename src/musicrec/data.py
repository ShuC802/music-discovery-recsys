"""Stage 0: turn the raw Last.fm-1K archive into compact, typed Parquet files.

Run once:
    uv run python -m musicrec.data

Why this step exists
--------------------
The raw TSV is 2.5 GB of text. Re-parsing it in every notebook would be slow
and memory hungry. We parse it once, fix types, and store it as Parquet:
columnar, compressed, and it remembers dtypes (timestamps stay timestamps,
categories stay categories).
"""
from __future__ import annotations

import csv
import tarfile
import time

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from musicrec import config

LISTEN_COLUMNS = ["user_id", "ts", "artist_mbid", "artist_name", "track_mbid", "track_name"]
STRING_COLUMNS = ["user_id", "artist_mbid", "artist_name", "track_mbid", "track_name"]
CHUNK_ROWS = 1_000_000

# Every chunk must be written with the same schema. Strings are stored as
# dictionary-encoded columns: each distinct value is stored once and rows keep
# a small integer code, which is exactly what pandas "category" dtype is.
LISTEN_SCHEMA = pa.schema(
    [("user_id", pa.dictionary(pa.int32(), pa.string())),
     ("ts", pa.timestamp("s", tz="UTC"))]
    + [(c, pa.dictionary(pa.int32(), pa.string())) for c in STRING_COLUMNS[1:]]
)


def _iter_listen_chunks(archive_path=config.RAW_ARCHIVE):
    """Yield DataFrame chunks straight out of the .tar.gz, no extraction to disk."""
    with tarfile.open(archive_path, mode="r:gz") as tar:
        fileobj = tar.extractfile(config.LISTENS_MEMBER)
        reader = pd.read_csv(
            fileobj,
            sep="\t",
            header=None,
            names=LISTEN_COLUMNS,
            dtype=str,                 # parse types ourselves, explicitly
            keep_default_na=False,     # otherwise a band called "NA" or "None" becomes missing
            na_values=[""],            # only a truly empty field means "missing"
            quoting=csv.QUOTE_NONE,    # track names contain stray quotes
            on_bad_lines="skip",       # a handful of lines have extra tabs
            encoding="utf-8",
            encoding_errors="replace",
            chunksize=CHUNK_ROWS,
        )
        yield from reader


def _clean_chunk(df: pd.DataFrame) -> pd.DataFrame:
    df["ts"] = pd.to_datetime(df["ts"], format="%Y-%m-%dT%H:%M:%SZ", utc=True, errors="coerce")
    return df


def build_listens(out_path=config.LISTENS_PARQUET) -> dict:
    """Stream the raw TSV into a single Parquet file, chunk by chunk."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats = {"rows": 0, "bad_ts": 0, "chunks": 0}
    t0 = time.time()

    with pq.ParquetWriter(out_path, LISTEN_SCHEMA, compression="zstd") as writer:
        for chunk in _iter_listen_chunks():
            chunk = _clean_chunk(chunk)
            stats["bad_ts"] += int(chunk["ts"].isna().sum())
            chunk = chunk.dropna(subset=["user_id", "ts"])
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            table = pa.table(
                {name: (table[name].dictionary_encode() if name in STRING_COLUMNS else table[name])
                 for name in LISTEN_COLUMNS}
            ).cast(LISTEN_SCHEMA)
            writer.write_table(table)
            stats["rows"] += len(chunk)
            stats["chunks"] += 1
            print(f"  chunk {stats['chunks']:>2}: {stats['rows']:>11,} rows  ({time.time() - t0:5.0f}s)")

    stats["seconds"] = round(time.time() - t0, 1)
    return stats


def build_users(out_path=config.USERS_PARQUET) -> pd.DataFrame:
    with tarfile.open(config.RAW_ARCHIVE, mode="r:gz") as tar:
        users = pd.read_csv(tar.extractfile(config.PROFILES_MEMBER), sep="\t")
    users = users.rename(columns={"#id": "user_id"})
    users["age"] = pd.to_numeric(users["age"], errors="coerce").astype("Int16")
    users["registered"] = pd.to_datetime(users["registered"], format="%b %d, %Y", errors="coerce")
    for col in ["user_id", "gender", "country"]:
        users[col] = users[col].astype("category")
    users.to_parquet(out_path, index=False)
    return users


def load_listens(columns: list[str] | None = None) -> pd.DataFrame:
    """The function every later stage uses to get the data."""
    return pd.read_parquet(config.LISTENS_PARQUET, columns=columns)


def load_users() -> pd.DataFrame:
    return pd.read_parquet(config.USERS_PARQUET)


if __name__ == "__main__":
    print("Building users.parquet ...")
    users = build_users()
    print(f"  {len(users):,} user profiles")
    print("Building listens.parquet (takes a few minutes) ...")
    print(build_listens())
