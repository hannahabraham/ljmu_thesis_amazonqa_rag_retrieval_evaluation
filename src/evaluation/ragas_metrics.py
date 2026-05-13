"""RAGAS wiring: faithfulness, context_precision, context_recall (k=5 only)."""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.llm_clients.ragas_judge import build_ragas_judge


def run_ragas(results_df: pd.DataFrame) -> Any:
    """Evaluate a results DataFrame with the columns:
        question, answer, contexts (list[str]), ground_truth.
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import context_precision, context_recall, faithfulness
    from ragas.run_config import RunConfig

    judge_llm, embeddings, n_keys = build_ragas_judge()
    ds = Dataset.from_pandas(results_df.reset_index(drop=True))
    return evaluate(
        ds,
        metrics=[faithfulness, context_precision, context_recall],
        llm=judge_llm,
        embeddings=embeddings,
        run_config=RunConfig(max_workers=max(1, n_keys)),
    )
