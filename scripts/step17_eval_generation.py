"""Aggregate generation-quality + faithfulness metrics across (pipeline, k).

Captures: EM, F1, ROUGE-L, BERTScore F1, Semantic Similarity, lexical
Groundedness, lexical Hallucination Rate. Bootstrap 95% CIs throughout.
"""
from __future__ import annotations

import logging
from itertools import product

import pandas as pd

from config.settings import K_VALUES, OUTPUT_DIR, PIPELINE_KEYS, pipeline_output_dir
from src.evaluation.faithfulness import groundedness, hallucination_rate_row
from src.evaluation.generation_metrics import (
    bertscore_f1,
    exact_match,
    rouge_l,
    semantic_similarity,
    token_f1,
)
from src.evaluation.statistics import bootstrap_ci
from src.utils.io import parse_list_field
from src.utils.logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


def main() -> None:
    rows: list[dict] = []
    for pipeline, k in product(PIPELINE_KEYS, K_VALUES):
        path = pipeline_output_dir(pipeline) / f"answers_k{k}.csv"
        if not path.exists():
            continue
        df_full = pd.read_csv(path)
        df_full["retrieved_context"] = df_full["retrieved_context"].apply(parse_list_field)

        df = df_full[df_full["gold_answer"].astype(str).str.upper() != "[UNANSWERABLE]"] \
            .reset_index(drop=True)
        if df.empty:
            continue

        ems = [exact_match(r["generated_answer"], r["gold_answer"]) for _, r in df.iterrows()]
        f1s = [token_f1(r["generated_answer"], r["gold_answer"]) for _, r in df.iterrows()]
        rouges = [rouge_l(r["generated_answer"], r["gold_answer"]) for _, r in df.iterrows()]
        berts = bertscore_f1(df["generated_answer"].fillna("").tolist(),
                             df["gold_answer"].fillna("").tolist())
        sims = semantic_similarity(df["generated_answer"].fillna("").tolist(),
                                   df["gold_answer"].fillna("").tolist())

        # Faithfulness/hallucination computed over *all* non-refusal rows in df_full,
        # not just answerable ones — refusals don't make claims, so they're skipped.
        refused = df_full["refused"].astype(bool).tolist() if "refused" in df_full.columns else \
            [False] * len(df_full)
        grounded_scores: list[float] = []
        halluc_scores: list[float] = []
        for (_, r), is_refused in zip(df_full.iterrows(), refused):
            if is_refused:
                continue
            g = groundedness(str(r.get("generated_answer", "")), r["retrieved_context"])
            h = hallucination_rate_row(str(r.get("generated_answer", "")), r["retrieved_context"])
            if g == g:
                grounded_scores.append(g)
            if h == h:
                halluc_scores.append(h)

        em_mean, em_lo, em_hi = bootstrap_ci([float(x) for x in ems])
        f1_mean, f1_lo, f1_hi = bootstrap_ci(f1s)
        rouge_mean, rouge_lo, rouge_hi = bootstrap_ci(rouges)
        bert_mean, bert_lo, bert_hi = bootstrap_ci(berts)
        sim_mean, sim_lo, sim_hi = bootstrap_ci(sims)
        gnd_mean, gnd_lo, gnd_hi = bootstrap_ci(grounded_scores)
        hal_mean, hal_lo, hal_hi = bootstrap_ci(halluc_scores)

        row = {
            "pipeline": pipeline, "k": k, "n": len(df),
            "em": em_mean, "em_lo": em_lo, "em_hi": em_hi, "em_correct": int(sum(ems)),
            "f1": f1_mean, "f1_lo": f1_lo, "f1_hi": f1_hi,
            "rouge_l": rouge_mean, "rouge_l_lo": rouge_lo, "rouge_l_hi": rouge_hi,
            "bertscore_f1": bert_mean, "bertscore_f1_lo": bert_lo, "bertscore_f1_hi": bert_hi,
            "semantic_similarity": sim_mean,
            "semantic_similarity_lo": sim_lo, "semantic_similarity_hi": sim_hi,
            "groundedness": gnd_mean,
            "groundedness_lo": gnd_lo, "groundedness_hi": gnd_hi,
            "hallucination_rate": hal_mean,
            "hallucination_rate_lo": hal_lo, "hallucination_rate_hi": hal_hi,
        }
        rows.append(row)

        # Mirror per-pipeline
        per_pipeline_path = pipeline_output_dir(pipeline) / "generation_metrics.csv"
        existing = (
            pd.read_csv(per_pipeline_path) if per_pipeline_path.exists()
            else pd.DataFrame()
        )
        existing = existing[~((existing.get("pipeline") == pipeline) & (existing.get("k") == k))] \
            if not existing.empty else existing
        out_df = pd.concat([existing, pd.DataFrame([row])], ignore_index=True) \
            .sort_values("k").reset_index(drop=True)
        out_df.to_csv(per_pipeline_path, index=False)

    out = OUTPUT_DIR / "generation_metrics.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
