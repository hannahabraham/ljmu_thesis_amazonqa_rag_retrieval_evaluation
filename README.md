# Review-Based RAG Comparison on AmazonQA

> A controlled comparative evaluation of five Retrieval-Augmented Generation (RAG) pipelines for review-grounded product question answering on the AmazonQA corpus, with statistical reporting (95% CIs, indicative-cell flagging) across retrieval, generation, faithfulness, efficiency, and robustness dimensions.

---

## Table of Contents

1. [Overview](#overview)
2. [Pipelines Under Test](#pipelines-under-test)
3. [Dataset](#dataset)
4. [Pipeline Architecture](#pipeline-architecture)
5. [Knowledge Base and Golden Dataset](#knowledge-base-and-golden-dataset)
6. [Evaluation Framework](#evaluation-framework)
7. [Outputs Layout](#outputs-layout)
8. [Statistical Reporting](#statistical-reporting)
9. [Project Structure](#project-structure)
10. [Installation](#installation)
11. [Configuration](#configuration)
12. [Usage](#usage)
13. [Tests](#tests)
14. [Dependencies](#dependencies)
15. [Acknowledgements](#acknowledgements)

---

## Overview

This project benchmarks five retrieval strategies for grounded answer generation over Amazon product reviews. A 200-record evaluation sample is drawn against an explicit quota table (55 yes/no + 145 descriptive; 125 answerable + 75 unanswerable; 120 train / 40 validation / 40 test) and evaluated across four retrieval depths (k ∈ {1, 3, 5, 10}). Answers are generated with `llama-3.3-70b-versatile` (Groq) and judged with `gemini-2.5-flash` (chosen as a cross-family judge to avoid self-bias).

The contribution is methodological as much as empirical: every metric cell in the result tables is reported with sample size and a 95% confidence interval; cells with `n < 10` are flagged `[indicative]` and excluded from the composite ranking, so weak sub-cells cannot dominate the recommendation.

---

## Pipelines Under Test

| # | Pipeline | What it does | Index used |
|---|---|---|---|
| 1 | **BM25** | Lexical baseline (`rank_bm25`) | BM25 pickle over passage chunks |
| 2 | **Dense** | Semantic baseline — `all-MiniLM-L6-v2` (384-d) over Qdrant | Qdrant `passages` |
| 3 | **Sentence Window** | Match sentence, expand `[prev, match, next]` for context | Qdrant `sentences` |
| 4 | **Hybrid** | BM25 + Dense fused via Reciprocal Rank Fusion (k_rrf=60) | BM25 + Qdrant `passages` |
| 5 | **Parent-Child** | Search ~100-token children, return full review parent to LLM | Qdrant `child_chunks` + parent map |

All pipelines share the same generator prompt and the same Groq client, so differences in answer quality are attributable to retrieval, not generation.

---

## Dataset

**Source:** [AmazonQA](https://github.com/amazonqa/amazonqa) — train/val/test JSONL splits with question-answer pairs grounded in product reviews.

| Split | Source URL |
|---|---|
| train | `https://amazon-qa.s3-us-west-2.amazonaws.com/train-qar.jsonl` |
| val   | `https://amazon-qa.s3-us-west-2.amazonaws.com/val-qar.jsonl` |
| test  | Google Drive (download via `gdown`) |

Raw JSONL files are gitignored (~3.9 GB) and fetched by [scripts/step01_download_dataset.py](scripts/step01_download_dataset.py).

### Sample design

The 200-record evaluation sample is **quota-stratified**, not proportional. Proportional sampling under-represents yes/no questions (~15% of the population) and unanswerable questions (~38%) below the thresholds needed for stable sub-analyses, so [config/settings.py](config/settings.py) declares an explicit quota table that the sampler hits directly.

| | answerable | unanswerable | row total |
|---|---|---|---|
| **yes/no** | 35 (21 train + 7 val + 7 test) | 20 (12 + 4 + 4) | **55** |
| **descriptive** | 90 (54 + 18 + 18) | 55 (33 + 11 + 11) | **145** |
| **column total** | **125** | **75** | **200** |
| split total | train 120 | val 40 | test 40 |

If a `(questionType, is_answerable, split)` cell runs short, the shortfall is redirected to a sibling **split** within the same questionType/answerability — never to a different questionType or answerability — so yes/no never gets backfilled with descriptive and answerable never gets backfilled with unanswerable. Implementation: [`quota_stratified_sample()`](src/sampling.py).

---

## Pipeline Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         END-TO-END PIPELINE                              │
│                                                                          │
│  download → load JSONL → standardise → EDA per split → merge & clean     │
│      → quota-stratified 200 sample → KB (review snippets) → golden draft │
│      → Gemini judge → soft consistency check → 3 chunking strategies     │
│      → BM25 + Qdrant indexes → 5 pipelines × 4 k values                  │
│      → answer generation → 4 evaluation suites → 3 analyses → export     │
└──────────────────────────────────────────────────────────────────────────┘
```

### Data preparation

| Step | Script | Output |
|---|---|---|
| 1. Download | `step01_download_dataset.py` | `datasets/raw/{train,val,test}-qar.jsonl` |
| 2. Load and standardise | `step02_load_and_standardize.py` | per-split DataFrames |
| 3. EDA per split | `step03_run_eda_per_split.py` | `eda_summary.csv` + 6 plots |
| 4. Merge and clean | `step04_merge_and_clean.py` | `combined_amazonqa.csv` |
| 5. Quota-stratified sample | `step05_stratified_sample.py` | `final_200_records.csv` |
| 6. Build KB | `step06_build_knowledge_base.py` | `knowledge_base_full_reviews.csv` |
| 7. Golden draft | `step07_build_golden_dataset_draft.py` | `golden_dataset_200_draft.csv` |
| 8a. Gemini judge | `step08a_run_gemini_judge.py` | `golden_dataset_200_verified.csv` |
| 8b. Soft consistency check | `step08b_validate_golden.py` | annotates `validation_status` column |
| 9. Chunking | `step09_create_chunks.py` | passage / sentence / parent-child CSVs |
| 10. Indexes | `step10_build_indexes.py` | 3 Qdrant collections + `bm25.pkl` |

### Retrieval, generation, evaluation

Each `step1X_run_<pipeline>.py` script runs retrieval + generation + per-cell evaluation in one call. Per-question records are written to **`outputs/per_question/<pipeline>_k<k>_seed<seed>.jsonl`** (the v5 source of truth), with per-pipeline artefacts mirrored under **`outputs/<pipeline>/`**. RAGAS scores are written back into the JSONL by step 18. Aggregate per-metric CSVs live directly under `outputs/` (`retrieval_metrics.csv`, `generation_metrics.csv`, `ragas_metrics.csv`, `answerability_metrics.csv`, etc.).

| Step | Script | Notes |
|---|---|---|
| 11. BM25 | `step11_run_bm25.py` | `--ks 1 3 5 10` (default), `--seed`, `--output-dir` |
| 12. Dense | `step12_run_dense.py` | Qdrant + MiniLM |
| 13. Sentence Window | `step13_run_sentence_window.py` | expand `[prev, match, next]` |
| 14. Hybrid | `step14_run_hybrid.py` | BM25 + Dense fused via RRF |
| 15. Parent-Child | `step15_run_parent_child.py` | full parent reviews |
| 16. Retrieval metrics | `step16_eval_retrieval.py` | Recall@K, MRR, nDCG@K (full table) |
| 17. Generation metrics | `step17_eval_generation.py` | EM, F1, ROUGE-L, Semantic Similarity, Groundedness |
| 18. RAGAS | `step18_eval_ragas.py` | Faithfulness, ContextPrecision/Recall (Gemini judge); per-row write-back |
| 19a. Answerability | `step19a_eval_answerability.py` | regex refusal detector, accuracy + Wilson CI |
| 19b. Hallucination | `step19b_eval_hallucination.py` | hallucination rate + refusal rate on answerable |
| 20. Category analysis | `step20_category_analysis.py` | per-category F1 (4 named categories), with CIs |
| 21. Length analysis | `step21_question_length_analysis.py` | per-`q_bucket` F1, with CIs |
| 22. Plot figures | `step22_plot_results.py` | renders Chapter 5 PNG figures under `outputs/figures/` |

---

## Knowledge Base and Golden Dataset

### Knowledge base

The KB is built from the review evidence the AmazonQA release ships with each question. [`src/knowledge_base_builder.py`](src/knowledge_base_builder.py) splits review fields into two tiers:

| Tier | Fields | Coverage |
|---|---|---|
| `REQUIRED_REVIEW_FIELDS` | `review_snippets` | 100% of rows |
| `OPTIONAL_REVIEW_FIELDS` | `top_sentences_IR`, `top_review_helpful`, `top_review_wilson` | populated only on the AmazonQA test split (~20%) |

Optional fields are read only when the column is present and non-empty for the record, so train/val rows do not produce silent zeros. Each row carries `doc_id` (e.g. `KB_00042`), `record_id`, `asin`, `category`, `source_field`, and the full review text. Reviews <5 words and within-record duplicates are dropped. Expected size on the 200-record sample is **~2,000–4,000 rows** (snippet-only baseline; a full per-ASIN review scrape is a known extension but is not required).

> **Honest framing for the thesis:** the retrieval corpus is built from review evidence provided alongside each question, not from every review of each product. This is the standard AmazonQA setup and the one the five pipelines compete on fairly.

### Golden dataset

A three-layer pipeline picks one canonical answer per question:

1. **Jeffreys score** — Bayesian-smoothed helpfulness `(helpful + 0.5) / (total + 1.0)`.
2. **Grounding filter** — drop candidates with token Jaccard <0.1 against the record's KB reviews.
3. **Gemini judge** — routed aggressively: every unanswerable row, every yes/no row, every ungrounded top candidate, every weakly grounded top candidate (Jaccard <0.2), every Jeffreys tie, and every zero-vote case. Uses `gemini-2.5-flash` with a constrained JSON schema validated by Pydantic; one retry on malformed JSON.

`evidence_text` is populated by KB lookup (not stored in the judge response), so the golden CSV cannot drift from the KB.

### Soft vs hard validation

[`scripts/step08b_validate_golden.py`](scripts/step08b_validate_golden.py) annotates each row with a `validation_status`:

- **Hard failures** (raise): missing answerability, answerable row with no `evidence_doc_id`, `evidence_doc_id` not in KB, `evidence_text` drifted from the KB row.
- **Soft warnings** (logged, not raised): `[UNANSWERABLE]` answer paired with `answerability=1` (or vice versa), unanswerable row carrying an evidence id.

The advisory AmazonQA `is_answerable` flag and the judge/grounding outcome can legitimately disagree — the soft warnings make those visible without blocking downstream work.

---

## Evaluation Framework

Every metric below is captured per `(pipeline, k)` cell by [`src/pipelines/runner.py`](src/pipelines/runner.py) and persisted into `outputs/<pipeline>/summary.csv`, with raw per-question outputs in `outputs/<pipeline>/answers_k{k}.csv` and `outputs/<pipeline>/retrieval_k{k}.csv`. Cross-pipeline aggregates with 95% CIs live at the top level (see [Outputs Layout](#outputs-layout)).

### Retrieval

| Metric | Definition |
|---|---|
| **Recall@K** | Fraction of gold docs in top-k (binary in single-evidence mode) |
| **MRR** | Mean reciprocal rank of `evidence_doc_id` |
| **nDCG@K** | Normalised DCG with binary relevance |

Unanswerable rows are excluded from retrieval aggregates (no defined gold doc); this is the `n` column in `outputs/retrieval_metrics.csv` — the number of answerable rows the metric was averaged over.

### Answer Quality

| Metric | Definition |
|---|---|
| **Yes/No EM (%)** | Strict yes/no exact-match on the slice where the gold answer starts with "yes" or "no" |
| **F1 Score** | Token overlap F1 |
| **ROUGE-L** | Longest common subsequence F1 |
| **Semantic Similarity** | Cosine similarity between MiniLM embeddings of prediction and gold |

For unanswerable rows the correct generation is a refusal — token F1 doesn't apply; answerability accuracy does.

### Faithfulness

| Metric | Source | Notes |
|---|---|---|
| **Faithfulness Score** | RAGAS (LLM-as-judge, Gemini) | merged in by step 23 |
| **Context Precision** | RAGAS | Retrieval-side relevance signal |
| **Context Recall** | RAGAS | Coverage of gold evidence |
| **Groundedness** | Lexical overlap of answer content tokens with retrieved context | Cheap, computed every cell |
| **Hallucination Rate** | `1 − groundedness`, mean over non-refusal rows | Cheap, computed every cell |

`answer_relevance` was dropped — multi-call, expensive, and overlaps with F1/Semantic Similarity. RAGAS is routed through Gemini explicitly via [`src/llm_clients/ragas_judge.py`](src/llm_clients/ragas_judge.py) so the judge is in a different model family from the generator.

### Efficiency

| Metric | Definition |
|---|---|
| **Avg Latency / Question** | Mean of `retrieval_ms + generation_ms` (seconds) |
| **Retrieval Latency** | Mean of `retrieval_ms` only — isolates retriever cost |

Median + p95 are also reported in the full-table aggregates because the LLM tail makes the mean unreliable.

### Robustness

| Metric | Definition |
|---|---|
| **Answerability Accuracy** | 2×2 table from the regex refusal detector |
| **Long Context Accuracy** | Answerability accuracy restricted to `q_bucket == "long"` (≥13 tokens) |
| **Noise Robustness** | F1 on the noisiest quartile of rows; clean-vs-noisy delta logged separately |

The refusal detector lives in [`src/generation/refusal.py`](src/generation/refusal.py); its patterns are unit-tested in [tests/test_refusal_detector.py](tests/test_refusal_detector.py).

|  | Model answered | Model refused |
|---|---|---|
| **Gold answerable** | Correctly Answered | Wrongly Refused |
| **Gold unanswerable** | Wrongly Answered | Correctly Refused |

---

## Outputs Layout

The layout was deliberately deduplicated: per-pipeline folders contain only raw per-question outputs and a live k-sweep summary; cross-pipeline aggregates are written **once** at the top level (no per-pipeline copies).

```
outputs/
├── bm25/
│   ├── retrieval_k{1,3,5,10}.csv     # raw retrieved docs per question (200 rows each)
│   ├── answers_k{1,3,5,10}.csv       # generated answers + retrieval payload (200 rows each)
│   └── summary.csv                   # live k-sweep across this pipeline (upserted by the runner)
├── dense/        … same layout
├── sentwin/      … same layout
├── hybrid/       … same layout
├── pc/           … same layout
│
├── per_question/                     # v5 source of truth (one JSONL per pipeline/k/seed)
│   └── <pipeline>_k<k>_seed<seed>.jsonl
│
├── results.csv                       # cross-pipeline summary, ranked
├── retrieval_metrics.csv             # cross-pipeline Recall/MRR/nDCG + 95% CIs
├── generation_metrics.csv            # cross-pipeline EM/F1/ROUGE/Sim/Groundedness + 95% CIs
├── ragas_metrics.csv                 # cross-pipeline Faithfulness / CP / CR
├── answerability_metrics.csv         # cross-pipeline Wilson CIs + long-context + noise robustness
├── category_metrics.csv              # per-category breakdown by k
├── qbucket_metrics.csv               # per-question-length breakdown by k
├── latency_detail.csv                # split retrieval / generation timings
└── eda_plots/                        # 6 plots × 3 splits
```

The `_lo` / `_hi` columns on each metric in the aggregate CSVs are bootstrap (continuous) or Wilson (proportion) 95% CI bounds — see [Statistical Reporting](#statistical-reporting).

---

## Statistical Reporting

Every cell in the cross-pipeline tables carries `n` and a 95% CI:

| Quantity | Method | Source |
|---|---|---|
| Continuous metrics (F1, ROUGE-L, latency, faithfulness, groundedness) | Percentile bootstrap, 1000 resamples | [src/evaluation/statistics.py](src/evaluation/statistics.py) |
| Proportions (accuracy, Recall@K, answerability) | Wilson score interval (`statsmodels`) | [src/evaluation/statistics.py](src/evaluation/statistics.py) |

Cells with `n < 10` are tagged `[indicative]` and **excluded from the composite ranking** to prevent thin sub-cells from steering the recommendation. The composite combines F1, Faithfulness, ContextPrecision/Recall, AnswerabilityAcc, CategoryConsistency, and a normalised-latency penalty:

```
score = 0.25·F1 + 0.20·Faithfulness + 0.15·ContextPrecision
      + 0.10·ContextRecall + 0.15·AnswerabilityAcc
      + 0.10·CategoryConsistency + 0.05·(1 − normalised_latency)
```

`CategoryConsistency = 1 / (1 + std(F1 across categories))` is bounded in (0, 1]. Sensitivity is checked under alternate weighting schemes; if the top pipeline changes, that's reported.

---

## Project Structure

```
.
├── README.md
├── requirements.txt
├── pyproject.toml                         # ruff / black / mypy / pytest config
├── .env.example
│
├── config/
│   └── settings.py                        # single source of truth (loads .env, owns SAMPLE_QUOTAS)
│
├── datasets/
│   ├── raw/                               # JSONL splits (gitignored)
│   ├── processed/                         # KB, golden, chunks
│   └── indexes/                           # bm25.pkl
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── eda.py
│   ├── sampling.py                        # quota_stratified_sample + legacy two-stage
│   ├── knowledge_base_builder.py          # REQUIRED + OPTIONAL review fields
│   ├── golden_dataset_builder.py          # Jeffreys + grounding + judge gating; soft validation
│   ├── chunking.py
│   ├── indexing.py
│   ├── retrievers/
│   │   ├── base.py
│   │   ├── bm25.py
│   │   ├── dense.py
│   │   ├── sentence_window.py
│   │   ├── hybrid.py
│   │   └── parent_child.py
│   ├── pipelines/
│   │   └── runner.py                      # per-cell runner: retrieval + gen + eval + summary upsert
│   ├── generation/
│   │   ├── prompt.py
│   │   ├── refusal.py                     # regex-based refusal detector
│   │   └── rag_generator.py
│   ├── llm_clients/
│   │   ├── loader.py                      # single active key loading / prompts
│   │   ├── parallel_groq.py               # sequential Groq batch client
│   │   ├── gemini_key_manager.py          # single-key google-genai Client
│   │   ├── ragas_judge.py
│   │   └── error_terms.py
│   ├── evaluation/
│   │   ├── retrieval_metrics.py           # Recall@K, MRR, nDCG@K
│   │   ├── generation_metrics.py          # Yes/No EM, F1, ROUGE-L, Semantic Similarity
│   │   ├── faithfulness.py                # lexical Groundedness, Hallucination Rate
│   │   ├── ragas_metrics.py               # RAGAS wiring (Gemini-routed)
│   │   ├── answerability.py
│   │   ├── robustness.py                  # long-context + noise robustness
│   │   └── statistics.py                  # bootstrap CI, Wilson CI
│   └── utils/
│       ├── logging_config.py
│       ├── caching.py                     # SHA256 prompt cache
│       └── io.py
│
├── scripts/
│   └── step01_download_dataset.py … step22_plot_results.py
│
├── outputs/
│   ├── {bm25,dense,sentwin,hybrid,pc}/    # answers + retrieval + summary (no metric duplicates)
│   ├── per_question/                      # v5 JSONL source of truth
│   └── *.csv                              # cross-pipeline aggregates + results.csv
│
└── tests/
    ├── test_*.py                          # target >85% line coverage on src/
    └── test_integration_pipeline.py       # end-to-end with stubs
```

---

## Installation

### Prerequisites

- Python 3.10+
- Docker (for Qdrant)
- Groq API key — generation
- Google Gemini API key — judge + RAGAS

### Setup

```bash
git clone https://github.com/hannahabraham/LJMU-Thesis-AmazonQA-RAG-Evaluation.git
cd LJMU-Thesis-AmazonQA-RAG-Evaluation

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .                    # makes config/ and src/ importable from scripts/

cp .env.example .env                # then edit with your real keys
docker run -d -p 6333:6333 qdrant/qdrant
```

---

## Configuration

`.env` (template in [.env.example](.env.example)):

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` | Active generator key; numbered keys are accepted only as fallback input |
| `GROQ_MODEL` | Default `llama-3.3-70b-versatile` |
| `GEMINI_API_KEY` | Active judge key; numbered keys are accepted only as fallback input |
| `GEMINI_JUDGE_MODEL` | Default `gemini-2.5-flash` |
| `QDRANT_HOST` / `QDRANT_PORT` | Default `localhost:6333` |
| `EMBEDDING_MODEL` | Default `sentence-transformers/all-MiniLM-L6-v2` |
| `RANDOM_SEED` | Default `42` (wired through every sampler / bootstrap) |
| `LLM_CACHE_DIR` | SHA256-keyed prompt cache (resumable runs) |
| `RAGAS_BATCH_SIZE` / `RAGAS_SLEEP_BETWEEN_BATCHES` / `RAGAS_MAX_RETRIES` / `RAGAS_BACKOFF_SECONDS` / `RAGAS_BACKOFF_MULTIPLIER` | RAGAS quota-pacing knobs |

The LLM clients use one active key at a time. If the active Groq or Gemini key fails with quota/authentication errors, the script asks for a replacement key in the terminal and continues with that key.

The sample-quota table (`SAMPLE_QUOTAS` in [config/settings.py](config/settings.py)) is the single source of truth for the 55/145, 125/75, 120/40/40 design — edit there if you ever want a different balance.

---

## Usage

### Data preparation (one-time)

```bash
python -m scripts.step01_download_dataset
python -m scripts.step02_load_and_standardize
python -m scripts.step03_run_eda_per_split
python -m scripts.step04_merge_and_clean
python -m scripts.step05_stratified_sample
python -m scripts.step06_build_knowledge_base
python -m scripts.step07_build_golden_dataset_draft
python -m scripts.step08a_run_gemini_judge
python -m scripts.step08b_validate_golden
python -m scripts.step09_create_chunks
python -m scripts.step10_build_indexes
```

### Smoke run (one-time)

```bash
python -m scripts.step11_run_bm25 --ks 5 --sample 50
```

### Run pipelines incrementally — one at a time, all k

Each script runs retrieval + generation + per-cell evaluation, writes per-cell artefacts to `outputs/<pipeline>/`, and upserts a row into `outputs/<pipeline>/summary.csv` and `outputs/results.csv`. Open either CSV between runs to inspect progress.

```bash
python -m scripts.step11_run_bm25             --ks 1 3 5 10
python -m scripts.step12_run_dense            --ks 1 3 5 10
python -m scripts.step13_run_sentence_window  --ks 1 3 5 10
python -m scripts.step14_run_hybrid           --ks 1 3 5 10
python -m scripts.step15_run_parent_child     --ks 1 3 5 10
```

Or one cell at a time when you want the tightest feedback loop:

```bash
python -m scripts.step11_run_bm25 --ks 5
```

### RAGAS, full-table aggregates, analysis, export

RAGAS runs after all 5 pipelines are done. The full-table eval scripts (16/17/19a) re-aggregate every per-pipeline `answers_k{k}.csv` they find and write their own metric CSVs (with bootstrap/Wilson CIs) — run them whenever you want.

```bash
python -m scripts.step16_eval_retrieval
python -m scripts.step17_eval_generation
python -m scripts.step18_eval_ragas              # Gemini-routed, per-row JSONL write-back
python -m scripts.step19a_eval_answerability
python -m scripts.step19b_eval_hallucination
python -m scripts.step20_category_analysis
python -m scripts.step21_question_length_analysis
python -m scripts.step22_plot_results          # renders Chapter 5 PNG figures
```

For quota-safe RAGAS runs, step 18 skips rows that already have RAGAS scores, restores matching rows from the on-disk cache, writes JSONL checkpoints after each batch, and retries quota/rate-limit failures with exponential backoff. Start conservatively and increase only after a cell completes without quota errors:

```bash
python -m scripts.step18_eval_ragas --pipelines bm25 --ks 1 --workers 1 --batch-size 10 --sleep-seconds 30
```

### LLM call budget

| Source | Calls |
|---|---|
| RAG generation (Groq) | 4,000 (5 pipelines × 4 k × 200 questions) |
| RAGAS judging (Gemini) | ~12,000 (5 pipelines × 4 k × 200 questions × 3 metrics) |
| Gemini golden judge | ~80–120 (one-time, aggressive gating: every yes/no + every unanswerable + every ungrounded) |
| **Subtotal** | **~16,200** |
| Re-run buffer (×2) | **~32,400** |

Single-key prompting plus the on-disk prompt cache keeps reruns resumable: completed answers are reused, and only blank/missing answers are called again.

---

## Tests

```bash
pytest                           # all tests including integration
pytest --cov=src                 # with coverage (target >85%)
pytest -m "not integration"      # unit only (fast)
pytest tests/test_refusal_detector.py -v
pytest tests/test_sampling_balance.py -v
ruff check src/ tests/
mypy src/
```

Key fast-running suites:

- [tests/test_sampling_balance.py](tests/test_sampling_balance.py) — asserts the quota sampler hits each `(questionType, is_answerable, split)` cell exactly and never backfills across questionType / answerability.
- [tests/test_golden_dataset_builder.py](tests/test_golden_dataset_builder.py) — covers Jeffreys scoring, grounding, judge gating, and the soft-validation status flags.
- [tests/test_refusal_detector.py](tests/test_refusal_detector.py) — pattern-level checks for the regex refusal detector.
- [tests/test_integration_pipeline.py](tests/test_integration_pipeline.py) — exercises the full pipeline on five fake records with stubbed Groq/Gemini clients and an in-memory retriever — no API keys, no Qdrant required.

---

## Dependencies

| Package | Role |
|---|---|
| `pandas`, `numpy`, `scipy`, `scikit-learn` | Data wrangling and stats |
| `statsmodels` | Wilson CI for proportions |
| `langchain`, `langchain-groq`, `langchain-google-genai` | LLM orchestration |
| `google-genai` | Per-instance Gemini SDK (replaces `google-generativeai`) |
| `sentence-transformers` | `all-MiniLM-L6-v2` embeddings + semantic similarity |
| `qdrant-client` | Vector store |
| `rank-bm25` | Lexical baseline |
| `nltk` | Sentence tokenisation for windowing |
| `ragas` | LLM-as-judge metrics |
| `rouge-score` | Generation metrics |
| `pytest`, `pytest-cov`, `pytest-mock` | Testing |
| `ruff`, `black`, `mypy` | Lint, format, type-check |


---

## Acknowledgements

- **Dataset:** [AmazonQA](https://github.com/amazonqa/amazonqa) — review-grounded product QA corpus.
- **Generator:** Meta `llama-3.3-70b-versatile` via Groq.
- **Judge:** Google `gemini-2.5-flash` via the `google-genai` SDK.
- **Vector store:** [Qdrant](https://qdrant.tech/).
- **Evaluation:** [RAGAS](https://docs.ragas.io) — faithfulness, context precision, context recall.
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`.
