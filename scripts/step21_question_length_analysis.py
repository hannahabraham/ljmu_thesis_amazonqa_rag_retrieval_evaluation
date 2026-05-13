"""Question-length bucket breakdown of F1 at k=5 (Sheet Table 4)."""
from __future__ import annotations

import logging

import pandas as pd

from config.settings import OUTPUT_DIR, PIPELINE_KEYS, pipeline_output_dir
from src.evaluation.generation_metrics import token_f1
from src.evaluation.statistics import bootstrap_ci, is_indicative
from src.sampling import assign_q_bucket
from src.utils.logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


def main() -> None:
    rows: list[dict] = []
    for pipeline in PIPELINE_KEYS:
        path = pipeline_output_dir(pipeline) / "answers_k5.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df = df[df["gold_answer"].astype(str).str.upper() != "[UNANSWERABLE]"].copy()
        # q_bucket may be NaN in older answers files (script 07 dropped it). Derive
        # from question text so the breakdown still works without re-running.
        if "q_bucket" not in df.columns or df["q_bucket"].isna().all():
            df["q_bucket"] = df["question"].apply(assign_q_bucket)
        df["f1"] = [token_f1(r["generated_answer"], r["gold_answer"]) for _, r in df.iterrows()]
        for bucket, group in df.groupby("q_bucket"):
            f1_mean, lo, hi = bootstrap_ci(group["f1"].tolist())
            rows.append({
                "pipeline": pipeline,
                "q_bucket": bucket,
                "n": len(group),
                "f1": f1_mean, "f1_lo": lo, "f1_hi": hi,
                "indicative": is_indicative(len(group)),
            })

    out = OUTPUT_DIR / "qbucket_metrics.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
