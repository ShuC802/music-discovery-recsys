"""Central place for paths and constants, so no other file hard-codes a path."""
from pathlib import Path

# src/musicrec/config.py -> parents[2] is the project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

RAW_ARCHIVE = RAW_DIR / "lastfm-dataset-1K.tar.gz"
LISTENS_MEMBER = "lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv"
PROFILES_MEMBER = "lastfm-dataset-1K/userid-profile.tsv"

LISTENS_PARQUET = PROCESSED_DIR / "listens.parquet"
USERS_PARQUET = PROCESSED_DIR / "users.parquet"

SEED = 42

# Stage 2 outputs
TRACKS_PARQUET = PROCESSED_DIR / "tracks.parquet"
SPLIT_VAL_DIR = PROCESSED_DIR / "split_val"     # train < 2009-01, eval Jan-Feb 2009 (tuning)
SPLIT_TEST_DIR = PROCESSED_DIR / "split_test"   # train < 2009-03, eval Mar-Apr 2009 (final, touch once)

VAL_TRAIN_END, VAL_EVAL_END = "2009-01-01", "2009-03-01"
TEST_TRAIN_END, TEST_EVAL_END = "2009-03-01", "2009-05-01"

# ---------------------------------------------------------------- Yambda-50M
# huggingface.co/datasets/yandex/yambda  (flat/50m/*.parquet + item mappings at repo root)
# Timestamps are seconds since the start of observation (~301 days), so cut points are ints.
YAMBDA_RAW_DIR = RAW_DIR / "yambda"
YAMBDA_DIR = PROCESSED_DIR / "yambda"
YAMBDA_FIGURES_DIR = FIGURES_DIR / "yambda"
YAMBDA_SPLIT_VAL_DIR = YAMBDA_DIR / "split_val"
YAMBDA_SPLIT_TEST_DIR = YAMBDA_DIR / "split_test"
DAY = 86_400
# Cut points (seconds). Two-week windows = two full weekly cycles.
YAMBDA_VAL_TRAIN_END, YAMBDA_VAL_EVAL_END = 273 * DAY, 287 * DAY
YAMBDA_TEST_TRAIN_END, YAMBDA_TEST_EVAL_END = 287 * DAY, 301 * DAY
