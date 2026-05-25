"""Single source of truth for runtime configuration.

Loads ``.env`` once at import time. All modules should import constants from
this module instead of reading ``os.environ`` directly.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "datasets"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "indexes"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
EDA_PLOTS_DIR = OUTPUT_DIR / "eda_plots"
TABLES_DIR = OUTPUT_DIR / "tables"
PER_QUESTION_DIR = OUTPUT_DIR / "per_question"

PIPELINE_KEYS: tuple[str, ...] = (
    "bm25",
    "dense",
    "sentwin",
    "hybrid",
    "pc",
)


def pipeline_output_dir(pipeline_key: str) -> Path:
    """Return the output directory for a pipeline.

    The directory is created automatically if it does not already exist.

    Args:
        pipeline_key: Identifier for the retrieval pipeline.

    Returns:
        Path to the pipeline-specific output directory.

    Raises:
        ValueError: If the pipeline key is unknown.

    """
    if pipeline_key not in PIPELINE_KEYS:
        raise ValueError(f"unknown pipeline key {pipeline_key!r}")

    path = OUTPUT_DIR / pipeline_key
    path.mkdir(parents=True, exist_ok=True)

    return path


# Create required directories at import time.
for directory in (
    RAW_DIR,
    PROCESSED_DIR,
    INDEX_DIR,
    OUTPUT_DIR,
    EDA_PLOTS_DIR,
    TABLES_DIR,
    PER_QUESTION_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)

for pipeline_key in PIPELINE_KEYS:
    (OUTPUT_DIR / pipeline_key).mkdir(parents=True, exist_ok=True)

RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))

# Models
GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "llama-3.3-70b-versatile",
)

GEMINI_JUDGE_MODEL = os.getenv(
    "GEMINI_JUDGE_MODEL",
    "gemini-2.5-flash",
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

EMBEDDING_DIM = 384

# Generation
GENERATION_TEMPERATURE = 0.0
GENERATION_MAX_TOKENS = 200

# Qdrant
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

QDRANT_COLLECTIONS = {
    "passages": "passages",
    "sentences": "sentences",
    "child_chunks": "child_chunks",
}

BM25_PICKLE_PATH = INDEX_DIR / "bm25.pkl"

# Cache
LLM_CACHE_DIR = Path(os.getenv("LLM_CACHE_DIR", ".cache/llm"))
LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Sampling / EDA
SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE", "200"))
TRAIN_SAMPLE = int(os.getenv("TRAIN_SAMPLE", "120"))
VAL_SAMPLE = int(os.getenv("VAL_SAMPLE", "40"))
TEST_SAMPLE = int(os.getenv("TEST_SAMPLE", "40"))

assert TRAIN_SAMPLE + VAL_SAMPLE + TEST_SAMPLE == SAMPLE_SIZE, (
    "TRAIN_SAMPLE + VAL_SAMPLE + TEST_SAMPLE "
    f"({TRAIN_SAMPLE} + {VAL_SAMPLE} + {TEST_SAMPLE}) "
    f"!= SAMPLE_SIZE ({SAMPLE_SIZE})"
)

# Quota table for quota_stratified_sample (used by step05).
#
# Designed to give the answerability and yes/no analyses enough power
# without leaving descriptive thin. Totals:
#   yes/no       = 55  (35 answerable + 20 unanswerable)
#   descriptive  = 145 (90 answerable + 55 unanswerable)
#   answerable   = 125, unanswerable = 75
#   train/val/test = 120/40/40
SAMPLE_QUOTAS: dict[tuple[str, int], dict[str, int]] = {
    ("yesno", 1):       {"train": 21, "val":  7, "test":  7},
    ("yesno", 0):       {"train": 12, "val":  4, "test":  4},
    ("descriptive", 1): {"train": 54, "val": 18, "test": 18},
    ("descriptive", 0): {"train": 33, "val": 11, "test": 11},
}

_quota_total = sum(
    count
    for split_counts in SAMPLE_QUOTAS.values()
    for count in split_counts.values()
)
assert _quota_total == SAMPLE_SIZE, (
    f"SAMPLE_QUOTAS sums to {_quota_total}, expected SAMPLE_SIZE={SAMPLE_SIZE}"
)
del _quota_total

WILSON_VOTE_THRESHOLD = 5

# short <= 5, medium 6-12, long >= 13
QUESTION_LENGTH_BUCKETS = (5, 12)

# Cells with n below this threshold are flagged.
INDICATIVE_THRESHOLD = 10

# Named categories 
NAMED_CATEGORIES: tuple[str, ...] = (
    "Electronics",
    "Toys_and_Games",
    "Health_and_Personal_Care",
    "Home_and_Kitchen",
)

MIN_PER_NAMED_CATEGORY = int(
    os.getenv("MIN_PER_NAMED_CATEGORY", "30")
)

# Correct-answer definition 
CORRECT_F1_THRESHOLD = float(
    os.getenv("CORRECT_F1_THRESHOLD", "0.5")
)

CORRECT_F1_SENSITIVITY = (0.3, 0.5, 0.7)

# Retrieval

K_VALUES = (1, 3, 5, 10)
RRF_K = 60

# RAGAS evaluation
RAGAS_BATCH_SIZE = int(os.getenv("RAGAS_BATCH_SIZE", "10"))
RAGAS_SLEEP_BETWEEN_BATCHES = float(
    os.getenv("RAGAS_SLEEP_BETWEEN_BATCHES", "15")
)
RAGAS_MAX_RETRIES = int(os.getenv("RAGAS_MAX_RETRIES", "5"))
RAGAS_BACKOFF_SECONDS = float(os.getenv("RAGAS_BACKOFF_SECONDS", "30"))
RAGAS_BACKOFF_MULTIPLIER = float(
    os.getenv("RAGAS_BACKOFF_MULTIPLIER", "2")
)

# Chunking
PASSAGE_CHUNK_TOKENS = 200
PASSAGE_CHUNK_OVERLAP = 20
CHILD_CHUNK_TOKENS = 100

# Dataset URLs
DATASET_URLS = {
    "train": (
        "https://amazon-qa.s3-us-west-2.amazonaws.com/"
        "train-qar.jsonl"
    ),
    "val": (
        "https://amazon-qa.s3-us-west-2.amazonaws.com/"
        "val-qar.jsonl"
    ),
    "test_drive_id": "1A_gaYbyBUOfwi8CQ7d5OO_b91lEvSnwr",
}

DATASET_FILES = {
    "train": RAW_DIR / "train-qar.jsonl",
    "val": RAW_DIR / "val-qar.jsonl",
    "test": RAW_DIR / "test-qar.jsonl",
}
