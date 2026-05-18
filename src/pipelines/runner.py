"""Per-pipeline cell runner.

`run_pipeline_cell(pipeline, k, sample, seed)` performs the full per-cell flow:

  1. Retrieval over the golden questions
       → outputs/<pipeline>/retrieval_k<k>.csv
  2. Generation via Groq
       → outputs/<pipeline>/answers_k<k>.csv
  3. Per-question JSONL (v5 source of truth, consumed by the table builders)
       → outputs/per_question/<pipeline>_k<k>_seed<seed>.jsonl
  4. Per-cell evaluation across:
       Retrieval     : Hit@K, Recall@K, MRR, nDCG@K
       Answer quality: Exact Match, F1, ROUGE-L, Semantic Similarity
       Faithfulness  : (lexical) Groundedness, Hallucination Rate
       Efficiency    : Avg Latency, Retrieval Latency
       Robustness    : Answerability Accuracy, Long-Context Accuracy,
                       Noise-Robust F1, Clean-vs-Noisy F1 delta
  5. Upsert one row into:
       outputs/<pipeline>/summary.csv         (this pipeline's k-sweep)
       outputs/results.csv                    (legacy cross-pipeline summary)

RAGAS Faithfulness / Context Precision / Context Recall are NOT computed here;
they come from scripts/step18_eval_ragas.py, which writes per-row scores back into
the per-question JSONL. The v5 Results Sheet tables are assembled from the JSONL
by scripts/step23_build_results_tables.py.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.settings import (
    CORRECT_F1_THRESHOLD,
    GENERATION_MAX_TOKENS,
    GENERATION_TEMPERATURE,
    GROQ_MODEL,
    OUTPUT_DIR,
    PER_QUESTION_DIR,
    PROCESSED_DIR,
    RANDOM_SEED,
    pipeline_output_dir,
)
from src.evaluation.answerability import compute_answerability_table
from src.evaluation.faithfulness import (
    aggregate_groundedness,
    aggregate_hallucination_rate,
)
from src.evaluation.generation_metrics import (
    correct_answers_count,
    exact_match,
    is_correct,
    rouge_l,
    semantic_similarity,
    token_f1,
    yesno_em,
)
from src.evaluation.latency import latency_detail
from src.evaluation.retrieval_metrics import (
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from src.evaluation.robustness import long_context_metrics, noise_robustness_metrics
from src.generation.prompt import PROMPT_TEMPLATE
from src.generation.rag_generator import format_context
from src.generation.refusal import is_refusal
from src.llm_clients.loader import load_groq_keys
from src.llm_clients.parallel_groq import GroqClient
from src.retrievers.base import Retriever
from src.retrievers.bm25 import BM25Retriever
from src.retrievers.dense import DenseRetriever
from src.retrievers.hybrid import HybridRetriever
from src.retrievers.parent_child import ParentChildRetriever
from src.retrievers.sentence_window import SentenceWindowRetriever
from src.utils.caching import get_cached, set_cached
from src.utils.io import parse_list_field, write_jsonl

logger = logging.getLogger(__name__)


PIPELINE_LABEL = {
    "bm25": "BM25 Retrieval",
    "dense": "Dense Retrieval",
    "sentwin": "Sentence Window Retrieval",
    "hybrid": "Hybrid Retrieval",
    "pc": "Parent-Child Retrieval",
}

RESULTS_COLUMNS = [
    "Pipeline",
    "pipeline_key",
    "K Value",
    "Total Questions",
    "Correct Answers",
    # Retrieval
    "Recall@K",
    "MRR",
    "nDCG@K",
    # Context (filled by RAGAS step)
    "Context Precision",
    "Context Recall",
    # Answer quality
    "Yes/No EM (%)",
    "F1 Score",
    "ROUGE-L",
    "Semantic Similarity",
    # Faithfulness
    "Faithfulness Score",  # RAGAS — filled later
    "Hallucination Rate",
    "Groundedness",
    # Efficiency
    "Avg Latency / Question (s)",
    "Retrieval Latency (s)",
    "Generation Latency (s)",
    "Retrieval p95 (s)",
    # Robustness
    "Answerability Accuracy",
    "Long Context Accuracy",
    "Noise Robustness",
]


def _build_retriever(
    pipeline: str,
    passage_chunks: pd.DataFrame,
    sentence_chunks: pd.DataFrame,
    bm25_retriever: BM25Retriever | None = None,
) -> Retriever:
    if pipeline == "bm25":
        return bm25_retriever or BM25Retriever(passage_chunks, text_col="text")
    if pipeline == "dense":
        return DenseRetriever()
    if pipeline == "sentwin":
        return SentenceWindowRetriever(sentence_chunks)
    if pipeline == "hybrid":
        return HybridRetriever(
            bm25=bm25_retriever or BM25Retriever(passage_chunks, text_col="text"),
            dense=DenseRetriever(),
        )
    if pipeline == "pc":
        return ParentChildRetriever()
    raise ValueError(f"unknown pipeline {pipeline!r}")


# Module-level caches so repeated cells in one process don't re-read CSVs from
# disk or re-tokenise the BM25 corpus. Each cache key is the source CSV path so
# rebuilding after a chunk regeneration works without restarting Python.
_GOLDEN_CACHE: dict[Path, pd.DataFrame] = {}
_CHUNKS_CACHE: dict[Path, pd.DataFrame] = {}
_BM25_CACHE: dict[Path, BM25Retriever] = {}


def _load_cached_csv(path: Path, cache: dict[Path, pd.DataFrame]) -> pd.DataFrame:
    """Return a cached DataFrame for ``path``, loading once per process."""
    if path not in cache:
        cache[path] = pd.read_csv(path)
    return cache[path]


def _get_bm25_retriever(passage_path: Path) -> BM25Retriever:
    """Return a cached BM25Retriever.

    Prefers the on-disk pickle at ``BM25_PICKLE_PATH`` when it is newer than
    the passage-chunks CSV; otherwise rebuilds from the CSV and refreshes the
    pickle so future processes hit the cache.
    """
    from config.settings import BM25_PICKLE_PATH

    if passage_path in _BM25_CACHE:
        return _BM25_CACHE[passage_path]

    pickle_fresh = (
        BM25_PICKLE_PATH.exists()
        and passage_path.exists()
        and BM25_PICKLE_PATH.stat().st_mtime >= passage_path.stat().st_mtime
    )
    if pickle_fresh:
        retriever = BM25Retriever.load(BM25_PICKLE_PATH)
    else:
        passage = _load_cached_csv(passage_path, _CHUNKS_CACHE)
        retriever = BM25Retriever(passage, text_col="text")
        retriever.save(BM25_PICKLE_PATH)

    _BM25_CACHE[passage_path] = retriever
    return retriever


def _run_retrieval(pipeline: str, k: int, sample: int | None) -> pd.DataFrame:
    from src.sampling import assign_q_bucket

    golden_path = PROCESSED_DIR / "golden_dataset_200_verified.csv"
    passage_path = PROCESSED_DIR / "passage_chunks.csv"
    sentence_path = PROCESSED_DIR / "sentence_chunks.csv"

    golden = _load_cached_csv(golden_path, _GOLDEN_CACHE)
    if sample:
        golden = golden.head(sample)

    passage = _load_cached_csv(passage_path, _CHUNKS_CACHE)
    sentence = _load_cached_csv(sentence_path, _CHUNKS_CACHE)
    bm25_retriever = _get_bm25_retriever(passage_path) if pipeline in {"bm25", "hybrid"} else None
    retriever = _build_retriever(pipeline, passage, sentence, bm25_retriever)

    rows: list[dict[str, Any]] = []
    for _, row in golden.iterrows():
        t0 = time.perf_counter()
        hits = list(retriever.retrieve(row["question"], row["asin"], k))
        retrieval_ms = (time.perf_counter() - t0) * 1000.0
        # q_bucket isn't carried into golden_dataset_200_verified.csv by script 07,
        # so derive it from the question text (same logic used in stratified_sample).
        q_bucket = row.get("q_bucket")
        if q_bucket is None or (isinstance(q_bucket, float) and pd.isna(q_bucket)):
            q_bucket = assign_q_bucket(row["question"])
        rows.append(
            {
                "golden_id": row["golden_id"],
                "record_id": row["record_id"],
                "asin": row["asin"],
                "category": row.get("category", "unknown"),
                "q_bucket": q_bucket,
                "question_type": row.get("question_type", row.get("questionType", "")),
                "pipeline": pipeline,
                "k": k,
                "question": row["question"],
                "gold_answer": row["golden_answer"],
                "evidence_doc_id": row.get("evidence_doc_id"),
                "is_answerable": row.get("answerability"),
                "retrieved_doc_ids": json.dumps(
                    [h.get("doc_id") for h in hits], ensure_ascii=False
                ),
                "retrieved_context": json.dumps(
                    [h.get("text") for h in hits], ensure_ascii=False
                ),
                "retrieval_ms": retrieval_ms,
            }
        )

    out = pipeline_output_dir(pipeline) / f"retrieval_k{k}.csv"
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    logger.info("Wrote %s (%d rows)", out, len(df))
    return df


def _run_generation(retrieval_df: pd.DataFrame, pipeline: str, k: int) -> pd.DataFrame:
    prompts: list[str] = []
    for _, r in retrieval_df.iterrows():
        doc_ids = parse_list_field(r["retrieved_doc_ids"])
        contexts = parse_list_field(r["retrieved_context"])
        retrieved_docs = [{"doc_id": d, "text": t} for d, t in zip(doc_ids, contexts)]
        prompts.append(
            PROMPT_TEMPLATE.format(
                question=r["question"], context=format_context(retrieved_docs)
            )
        )

    cached_answers: list[str | None] = []
    cached_latency: list[float | None] = []
    pending_idx: list[int] = []
    pending_prompts: list[str] = []
    for i, prompt in enumerate(prompts):
        hit = get_cached("groq_rag", prompt, GROQ_MODEL)
        generated_answer = ""
        if hit is not None and isinstance(hit, dict):
            generated_answer = str(hit.get("generated_answer") or "").strip()

        if generated_answer:
            cached_answers.append(generated_answer)
            cached_latency.append(float(hit.get("generation_ms", 0.0)))
        else:
            cached_answers.append(None)
            cached_latency.append(None)
            pending_idx.append(i)
            pending_prompts.append(prompt)

    logger.info(
        "[%s k=%d] %d/%d cached, %d to call",
        pipeline,
        k,
        len(prompts) - len(pending_idx),
        len(prompts),
        len(pending_idx),
    )

    if pending_prompts:
        client = GroqClient(
            api_keys=load_groq_keys(),
            model=GROQ_MODEL,
            temperature=GENERATION_TEMPERATURE,
            max_tokens=GENERATION_MAX_TOKENS,
        )
        live_answers, live_latency = client.batch_invoke(pending_prompts)
        for i, ans, ms in zip(pending_idx, live_answers, live_latency):
            generated_answer = str(ans or "").strip()
            cached_answers[i] = generated_answer
            cached_latency[i] = ms
            if generated_answer:
                set_cached(
                    "groq_rag",
                    {"generated_answer": generated_answer, "generation_ms": ms},
                    prompts[i],
                    GROQ_MODEL,
                )

    rows: list[dict[str, Any]] = []
    for i, (_, r) in enumerate(retrieval_df.iterrows()):
        ans = cached_answers[i] or ""
        ms = float(cached_latency[i] or 0.0)
        rows.append(
            {
                **r.to_dict(),
                "generated_answer": ans,
                "refused": is_refusal(ans),
                "generation_ms": ms,
            }
        )

    out = pipeline_output_dir(pipeline) / f"answers_k{k}.csv"
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    logger.info("Wrote %s (%d rows)", out, len(df))
    return df


def _round_or_blank(value: float, digits: int = 4) -> Any:
    if value is None:
        return ""
    try:
        if np.isnan(value):
            return ""
    except TypeError:
        return ""
    return round(float(value), digits)


def compute_per_cell_metrics(answers_df: pd.DataFrame, k: int) -> dict[str, Any]:
    """Compute every per-cell metric the thesis cares about for one (pipeline, k) cell.

    Returns a dict keyed by `RESULTS_COLUMNS` headers (where applicable). RAGAS
    Faithfulness / Context Precision / Context Recall are intentionally left out
    of the dict — they come from scripts/step18_eval_ragas.py.
    """
    df = answers_df.copy()
    df["retrieved_doc_ids"] = df["retrieved_doc_ids"].apply(parse_list_field)
    df["retrieved_context"] = df["retrieved_context"].apply(parse_list_field)

    # ---------- Retrieval ----------
    answerable_with_evidence = df[df["evidence_doc_id"].notna()]
    if len(answerable_with_evidence):
        recalls = [
            recall_at_k(r["retrieved_doc_ids"], r["evidence_doc_id"], k)
            for _, r in answerable_with_evidence.iterrows()
        ]
        rrs = [
            reciprocal_rank(r["retrieved_doc_ids"], r["evidence_doc_id"])
            for _, r in answerable_with_evidence.iterrows()
        ]
        ndcgs = [
            ndcg_at_k(r["retrieved_doc_ids"], r["evidence_doc_id"], k)
            for _, r in answerable_with_evidence.iterrows()
        ]
        recall_mean = float(np.nanmean(recalls))
        mrr_mean = float(np.nanmean(rrs))
        ndcg_mean = float(np.nanmean(ndcgs))
    else:
        recall_mean = mrr_mean = ndcg_mean = float("nan")

    # ---------- Answer quality (only on answerable rows) ----------
    answerable = df[df["gold_answer"].astype(str).str.upper() != "[UNANSWERABLE]"]
    if len(answerable):
        f1s = [
            token_f1(r["generated_answer"], r["gold_answer"])
            for _, r in answerable.iterrows()
        ]
        rouges = [
            rouge_l(r["generated_answer"], r["gold_answer"])
            for _, r in answerable.iterrows()
        ]
        sims = semantic_similarity(
            answerable["generated_answer"].fillna("").tolist(),
            answerable["gold_answer"].fillna("").tolist(),
        )
        f1_mean = float(np.nanmean(f1s))
        rouge_mean = float(np.nanmean(rouges)) if rouges else float("nan")
        sim_mean = float(np.nanmean(sims)) if sims else float("nan")
    else:
        f1_mean = rouge_mean = sim_mean = float("nan")

    # Yes/No EM is computed on the yes/no slice (gold starts with yes/no).
    yesno_mask = answerable["gold_answer"].astype(str).str.strip().str.lower().str.match(
        r"^(yes|no)\b"
    )
    yesno_subset = answerable[yesno_mask.fillna(False)]
    if len(yesno_subset):
        yesno_scores = [
            yesno_em(r["generated_answer"], r["gold_answer"])
            for _, r in yesno_subset.iterrows()
        ]
        yesno_pct = float(np.mean(yesno_scores) * 100.0)
    else:
        yesno_pct = float("nan")

    correct_count = correct_answers_count(df, CORRECT_F1_THRESHOLD)

    # ---------- Faithfulness (lexical) ----------
    # Treat missing refused flags as False so a row whose generation didn't
    # produce a refusal flag (e.g. a failed call left NaN) isn't silently
    # excluded from the groundedness / hallucination aggregates.
    refused_flags = (
        df["refused"].fillna(False).astype(bool).tolist()
        if "refused" in df.columns
        else None
    )
    grounded_mean = aggregate_groundedness(
        df["generated_answer"].fillna("").tolist(),
        df["retrieved_context"].tolist(),
        refused_flags,
    )
    halluc_mean = aggregate_hallucination_rate(
        df["generated_answer"].fillna("").tolist(),
        df["retrieved_context"].tolist(),
        refused_flags,
    )

    # ---------- Robustness ----------
    ans_df = df[df["is_answerable"].notna()].copy()
    if len(ans_df):
        ans_df["is_answerable"] = ans_df["is_answerable"].astype(int)
        answerability_acc = float(
            compute_answerability_table(ans_df)["answerability_acc"].iloc[0]
        )
    else:
        answerability_acc = float("nan")

    long_ctx = long_context_metrics(df)
    noise = noise_robustness_metrics(df, k)

    # ---------- Efficiency ----------
    df_lat = df.assign(
        total_ms=df["retrieval_ms"].astype(float) + df["generation_ms"].astype(float)
    )
    avg_latency_s = float(df_lat["total_ms"].mean()) / 1000.0
    retrieval_latency_s = float(df["retrieval_ms"].astype(float).mean()) / 1000.0
    generation_latency_s = float(df["generation_ms"].astype(float).mean()) / 1000.0
    lat_detail = latency_detail(df_lat)
    retrieval_p95_s = lat_detail["retrieval_p95_ms"] / 1000.0

    return {
        "Total Questions": int(len(df)),
        "Correct Answers": correct_count,
        # Retrieval
        "Recall@K": _round_or_blank(recall_mean),
        "MRR": _round_or_blank(mrr_mean),
        "nDCG@K": _round_or_blank(ndcg_mean),
        # Answer quality
        "Yes/No EM (%)": _round_or_blank(yesno_pct, digits=2),
        "F1 Score": _round_or_blank(f1_mean),
        "ROUGE-L": _round_or_blank(rouge_mean),
        "Semantic Similarity": _round_or_blank(sim_mean),
        # Faithfulness (lexical — RAGAS Faithfulness merged later)
        "Hallucination Rate": _round_or_blank(halluc_mean),
        "Groundedness": _round_or_blank(grounded_mean),
        # Efficiency
        "Avg Latency / Question (s)": _round_or_blank(avg_latency_s, digits=3),
        "Retrieval Latency (s)": _round_or_blank(retrieval_latency_s, digits=3),
        "Generation Latency (s)": _round_or_blank(generation_latency_s, digits=3),
        "Retrieval p95 (s)": _round_or_blank(retrieval_p95_s, digits=3),
        # Robustness
        "Answerability Accuracy": _round_or_blank(answerability_acc),
        "Long Context Accuracy": _round_or_blank(
            long_ctx["long_context_answerability"]
        ),
        "Noise Robustness": _round_or_blank(noise["noise_robust_f1"]),
    }


def _upsert_pipeline_summary(pipeline: str, k: int, metrics: dict[str, Any]) -> None:
    """Maintain outputs/<pipeline>/summary.csv across k values."""
    new_row = {
        "Pipeline": PIPELINE_LABEL[pipeline],
        "pipeline_key": pipeline,
        "K Value": int(k),
        **metrics,
        "Faithfulness Score": "",  # RAGAS merge later
        "Context Precision": "",
        "Context Recall": "",
    }
    new_row = {col: new_row.get(col, "") for col in RESULTS_COLUMNS}

    path = pipeline_output_dir(pipeline) / "summary.csv"
    if path.exists():
        df = pd.read_csv(path)
        mask = (df["pipeline_key"] == pipeline) & (df["K Value"] == k)
        if mask.any():
            for ragas_col in (
                "Faithfulness Score",
                "Context Precision",
                "Context Recall",
            ):
                existing = df.loc[mask, ragas_col].iloc[0]
                if pd.notna(existing) and str(existing).strip() != "":
                    new_row[ragas_col] = existing
            df = df[~mask]
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.DataFrame([new_row])

    df = (
        df.reindex(columns=RESULTS_COLUMNS)
        .sort_values(["K Value"])
        .reset_index(drop=True)
    )
    df.to_csv(path, index=False)
    logger.info("Upserted %s/k=%d into %s", pipeline, k, path)


def upsert_results_row(pipeline: str, k: int, metrics: dict[str, Any]) -> None:
    """Upsert the legacy cross-pipeline outputs/results.csv row for (pipeline, k).

    Faithfulness / Context Precision / Context Recall are left blank here. The
    v5 Results Sheet tables come from the per-question JSONL via
    scripts/step23_build_results_tables.py; this CSV is kept as a quick-glance
    cross-pipeline snapshot only.
    """
    path = OUTPUT_DIR / "results.csv"
    new_row = {
        "Pipeline": PIPELINE_LABEL[pipeline],
        "pipeline_key": pipeline,
        "K Value": int(k),
        **metrics,
        "Faithfulness Score": "",
        "Context Precision": "",
        "Context Recall": "",
    }
    new_row = {col: new_row.get(col, "") for col in RESULTS_COLUMNS}

    if path.exists():
        df = pd.read_csv(path)
        # Tolerate older results.csv schemas: drop columns we no longer track,
        # add ones we now track.
        df = df.reindex(columns=[c for c in df.columns if c in RESULTS_COLUMNS])
        for col in RESULTS_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        mask = (df["pipeline_key"] == pipeline) & (df["K Value"] == k)
        if mask.any():
            for ragas_col in (
                "Faithfulness Score",
                "Context Precision",
                "Context Recall",
            ):
                existing = df.loc[mask, ragas_col].iloc[0]
                if pd.notna(existing) and str(existing).strip() != "":
                    new_row[ragas_col] = existing
            df = df[~mask]
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.DataFrame([new_row])

    df = df.reindex(columns=RESULTS_COLUMNS)
    df = df.sort_values(["pipeline_key", "K Value"]).reset_index(drop=True)
    df.to_csv(path, index=False)
    logger.info("Upserted %s/k=%d into %s", pipeline, k, path)


def _write_per_question_jsonl(
    answers_df: pd.DataFrame,
    pipeline: str,
    k: int,
    seed: int,
    output_dir: Path,
    f1_threshold: float = CORRECT_F1_THRESHOLD,
) -> Path:
    """Emit the v5 per-question JSONL: outputs/per_question/{run_id}.jsonl."""
    run_id = f"{pipeline}_k{k}_seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{run_id}.jsonl"

    rows: list[dict[str, Any]] = []
    for _, r in answers_df.iterrows():
        doc_ids = parse_list_field(r.get("retrieved_doc_ids"))
        contexts = parse_list_field(r.get("retrieved_context"))
        gold_answer = r.get("gold_answer", "")
        generated_answer = r.get("generated_answer", "") or ""
        em = (
            exact_match(generated_answer, gold_answer)
            if gold_answer not in (None, "", "[UNANSWERABLE]")
            else 0
        )
        f1 = (
            token_f1(generated_answer, gold_answer)
            if gold_answer not in (None, "", "[UNANSWERABLE]")
            else 0.0
        )
        gold_first = str(gold_answer or "").strip().lower().split()[:1]
        is_yesno_gold = bool(gold_first and gold_first[0] in {"yes", "no"})
        yn_em = (
            yesno_em(generated_answer, gold_answer)
            if is_yesno_gold and gold_answer != "[UNANSWERABLE]"
            else None
        )
        is_ans_raw = r.get("is_answerable")
        if isinstance(is_ans_raw, (int, np.integer, float, np.floating)):
            is_answerable = bool(int(is_ans_raw)) if not pd.isna(is_ans_raw) else False
        else:
            is_answerable = bool(is_ans_raw)
        refused = bool(r.get("refused", False))
        retrieval_ms = float(r.get("retrieval_ms", 0.0))
        generation_ms = float(r.get("generation_ms", 0.0))

        record = {
            "run_id": run_id,
            "seed": int(seed),
            "golden_id": r.get("golden_id"),
            "record_id": r.get("record_id"),
            "asin": r.get("asin"),
            "category": r.get("category", "unknown"),
            "q_bucket": r.get("q_bucket"),
            "question_type": r.get("question_type", ""),
            "pipeline": pipeline,
            "k": int(k),
            "question": r.get("question"),
            "gold_answer": gold_answer,
            "is_answerable": is_answerable,
            "evidence_doc_id": r.get("evidence_doc_id"),
            "retrieved_doc_ids": list(doc_ids),
            "retrieved_context": list(contexts),
            "generated_answer": generated_answer,
            "refused": refused,
            "em": int(em),
            "token_f1": float(f1),
            "yesno_em": yn_em,
            "is_yesno_gold": is_yesno_gold,
            "retrieval_ms": retrieval_ms,
            "generation_ms": generation_ms,
            "total_ms": retrieval_ms + generation_ms,
        }
        record["is_correct"] = bool(is_correct(record, f1_threshold))
        rows.append(record)

    write_jsonl(rows, out_path)
    logger.info("Wrote %s (%d rows)", out_path, len(rows))
    return out_path


def run_pipeline_cell(
    pipeline: str,
    k: int,
    sample: int | None = None,
    seed: int = RANDOM_SEED,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run retrieval + generation + per-cell eval for one (pipeline, k) cell.

    Per-question records are written to ``output_dir`` (defaults to
    ``outputs/per_question/``).
    """
    if pipeline not in PIPELINE_LABEL:
        raise ValueError(f"unknown pipeline {pipeline!r}")

    if output_dir is None:
        output_dir = PER_QUESTION_DIR

    logger.info("=== %s k=%d seed=%d ===", pipeline, k, seed)
    retrieval_df = _run_retrieval(pipeline, k, sample)
    answers_df = _run_generation(retrieval_df, pipeline, k)
    _write_per_question_jsonl(answers_df, pipeline, k, seed=seed, output_dir=output_dir)

    metrics = compute_per_cell_metrics(answers_df, k)
    _upsert_pipeline_summary(pipeline, k, metrics)
    upsert_results_row(pipeline, k, metrics)
    logger.info(
        "[%s k=%d] EM=%s%% F1=%s Hit=%s Recall=%s MRR=%s nDCG=%s "
        "ROUGE-L=%s Sim=%s Halluc=%s Ans=%s LongCtx=%s Noise=%s Lat=%ss",
        pipeline,
        k,
        metrics.get("Exact Match Accuracy (%)"),
        metrics.get("F1 Score"),
        metrics.get("Hit@K"),
        metrics.get("Recall@K"),
        metrics.get("MRR"),
        metrics.get("nDCG@K"),
        metrics.get("ROUGE-L"),
        metrics.get("Semantic Similarity"),
        metrics.get("Hallucination Rate"),
        metrics.get("Answerability Accuracy"),
        metrics.get("Long Context Accuracy"),
        metrics.get("Noise Robustness"),
        metrics.get("Avg Latency / Question (s)"),
    )
    return metrics


def run_pipeline_cells(
    pipeline: str,
    ks: list[int],
    sample: int | None = None,
    seed: int = RANDOM_SEED,
    output_dir: Path | None = None,
) -> None:
    """Run a pipeline across multiple k values, one cell at a time."""
    for k in ks:
        run_pipeline_cell(pipeline, k, sample=sample, seed=seed, output_dir=output_dir)
