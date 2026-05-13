"""Load AmazonQA JSONL splits as DataFrames."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_jsonl(path: Path) -> pd.DataFrame:
    """Read a JSONL file line-by-line, skipping malformed lines with a warning."""
    rows: list[dict] = []
    skipped = 0
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                skipped += 1
                logger.warning("Skipping malformed line %d in %s: %s", lineno, path, error)
    if skipped:
        logger.warning("Skipped %d malformed lines in %s", skipped, path)
    return pd.DataFrame(rows)
