"""IO helpers shared across data prep / experiments."""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

_NUMPY_ARRAY_RE = re.compile(r"\barray\(\s*(\[[^\]]*\])\s*(?:,\s*dtype=[^)]+)?\)")


def _strip_numpy_repr(text: str) -> str:
    """Replace `array([1, 2])` and `array([1, 2], dtype=int64)` with bare `[1, 2]`.

    CSV round-trips of dataframes containing numpy arrays produce these reprs;
    they trip up json.loads and ast.literal_eval because `array` is not a literal.
    """
    return _NUMPY_ARRAY_RE.sub(r"\1", text)


def parse_list_field(value: Any) -> list[Any]:
    """Best-effort parse a field that may be a list, JSON string, or Python literal."""
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return list(value)
    if isinstance(value, list):
        return value
    if isinstance(value, float) and pd.isna(value):
        return []
    if not isinstance(value, str):
        return [value]
    text = value.strip()
    if not text or text in ("[]", "nan", "NaN", "None"):
        return []
    candidates = (text, _strip_numpy_repr(text))
    for candidate in candidates:
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(candidate)
                return parsed if isinstance(parsed, list) else [parsed]
            except (json.JSONDecodeError, ValueError, SyntaxError):
                continue
    return [text]


def write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_per_question(
    per_question_dir: Path,
    pipelines: Iterable[str] | None = None,
    ks: Iterable[int] | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Load every per-question JSONL under ``per_question_dir`` into one DataFrame.

    Optionally filters by pipeline / k / seed via filename match (filename schema:
    ``{pipeline}_k{k}_seed{seed}.jsonl``).
    """
    if not per_question_dir.exists():
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    pipeline_set = set(pipelines) if pipelines is not None else None
    k_set = {int(k) for k in ks} if ks is not None else None
    for path in sorted(per_question_dir.glob("*.jsonl")):
        stem = path.stem  # pipeline_k{k}_seed{seed}
        parts = stem.split("_")
        if len(parts) < 3:
            continue
        seed_part = parts[-1]
        k_part = parts[-2]
        pipeline_part = "_".join(parts[:-2])
        if not seed_part.startswith("seed") or not k_part.startswith("k"):
            continue
        try:
            file_seed = int(seed_part[len("seed"):])
            file_k = int(k_part[len("k"):])
        except ValueError:
            continue
        if seed is not None and file_seed != seed:
            continue
        if k_set is not None and file_k not in k_set:
            continue
        if pipeline_set is not None and pipeline_part not in pipeline_set:
            continue
        rows.extend(read_jsonl(path))
    return pd.DataFrame(rows)
