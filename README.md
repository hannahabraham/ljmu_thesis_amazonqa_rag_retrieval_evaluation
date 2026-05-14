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
13. [Reproducibility](#reproducibility)
14. [Tests](#tests)
15. [Dependencies](#dependencies)

---

## Overview

This project benchmarks five retrieval strategies for grounded answer generation over Amazon product reviews. A stratified sample of 100 questions (60 train / 20 validation / 20 test) is evaluated across four retrieval depths (k ∈ {1, 3, 5, 10}), generating answers with `llama-3.3-70b-versatile` (Groq) and judging them with `gemini-2.5-flash` (chosen as a cross-family judge to avoid self-bias).

The contribution is methodological as much as empirical: every metric cell in the result tables is reported with sample size and a 95% confidence interval; cells with n<10 are flagged `[indicative]` and excluded from the composite ranking, so weak sub-cells cannot dominate the recommendation.

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

Raw JSONL files are gitignored (~3.9 GB) and fetched by [scripts/01_download_dataset.py](scripts/01_download_dataset.py).

**Sampled corpus:** 100 records, stratified by `questionType + is_answerable`, drawn 60/20/20 from train/val/test. The per-ASIN review pool is **uncapped** — every available review is kept (~30–50 chunks/ASIN), giving each retriever a meaningful candidate space.

---

## Pipeline Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         END-TO-END PIPELINE                              │
│                                                                          │
│  download → load JSONL → standardise → EDA per split → merge & clean     │
│      → stratified-100 sample → KB (full reviews) → golden draft          │
│      → Gemini judge → consistency check → 3 chunking strategies          │
│      → BM25 + Qdrant indexes → 5 pipelines × 4 k values                  │
│      → answer generation → 4 evaluation suites → 3 analyses → export     │
└──────────────────────────────────────────────────────────────────────────┘
```

### Data preparation

| Step | Script | Output |
|---|---|---|
| 1. Download | `01_download_dataset.py` | `datasets/raw/{train,val,test}-qar.jsonl` |
| 2. Load and standardise | `02_load_and_standardize.py` | per-split DataFrames |
| 3. EDA per split | `03_run_eda_per_split.py` | `eda_summary.csv` + 6 plots |
| 4. Merge and clean | `04_merge_and_clean.py` | `combined_amazonqa.csv` |
| 5. Stratified sample | `05_stratified_sample.py` | `final_100_records.csv` |
| 6. Build KB | `06_build_knowledge_base.py` | `knowledge_base_full_reviews.csv` |
| 7. Golden draft | `07_build_golden_dataset_draft.py` | `golden_dataset_100_draft.csv` |
| 8. Gemini judge | `08_run_gemini_judge.py` | `golden_dataset_100_verified.csv` |
| 8b. Consistency check | `08b_validate_golden.py` | fail-fast assertion |
| 9. Chunking | `09_create_chunks.py` | passage / sentence / parent-child CSVs |
| 10. Indexes | `10_build_indexes.py` | 3 Qdrant collections + `bm25.pkl` |

### Retrieval, generation, evaluation

Each `1X_run_<pipeline>.py` script runs retrieval + generation + per-cell evaluation in one call. Per-question records are written to **`outputs/per_question/<pipeline>_k<k>_seed<seed>.jsonl`** (the v5 source of truth), with per-pipeline artefacts mirrored under **`outputs/<pipeline>/`**. RAGAS scores are written back into the JSONL by step 18; the six Results Sheet tables are assembled from the JSONL by `24_build_results_tables.py`.

| Step | Script | Notes |
|---|---|---|
| 11. BM25 | `11_run_bm25.py` | `--ks 1 3 5 10` (default), `--seed`, `--output-dir` |
| 12. Dense | `12_run_dense.py` | Qdrant + MiniLM |
| 13. Sentence Window | `13_run_sentence_window.py` | expand `[prev, match, next]` |
| 14. Hybrid | `14_run_hybrid.py` | BM25 + Dense fused via RRF |
| 15. Parent-Child | `15_run_parent_child.py` | full parent reviews |
| 16. Retrieval metrics | `16_eval_retrieval.py` | Recall@K, MRR (full table) |
| 17. Generation metrics | `17_eval_generation.py` | EM, F1, Correct-Answers |
| 18. RAGAS | `18_eval_ragas.py` | all 4 k — Faithfulness, ContextPrecision/Recall (Gemini judge); per-row write-back |
| 19. Answerability | `19_eval_answerability.py` | regex refusal detector, accuracy |
| 19b. Hallucination | `19b_eval_hallucination.py` | hallucination rate + refusal rate on answerable |
| 20. Category analysis | `20_category_analysis.py` | per-category F1 (4 named categories), with CIs |
| 21. Length analysis | `21_question_length_analysis.py` | per-`q_bucket` F1, with CIs |
| 22. Final ranking | `22_final_ranking.py` | composite + pairwise Wilcoxon + sensitivity sweeps |
| 23. Reproducibility | `23_reproducibility_check.py` | second-seed drift report |
| 24. Build results tables | `24_build_results_tables.py` | 6 CSVs to `outputs/tables/` aligned to the Results Sheet |
| 25. Excel export | `25_export_excel.py` | bundles the table CSVs into one `.xlsx` |

---

## Knowledge Base and Golden Dataset

### Knowledge base

The KB stores **full reviews** (no curated 3-chunk cap), pulled from `review_snippets`, `top_sentences_IR`, `top_review_helpful`, and `top_review_wilson` for each of the 100 records. Reviews <5 words and within-record duplicates are dropped. Expected size: 3,000–5,000 KB rows.

Each row carries `doc_id` (e.g. `KB_00042`), `record_id`, `asin`, `category`, source field, and the full review text — enabling honest Parent-Child retrieval (real parent reviews, not synthetic ones) and a real candidate pool for BM25 / Dense / Hybrid to rank over.

### Golden dataset

A three-layer pipeline picks one canonical answer per question:

1. **Jeffreys score** — Bayesian-smoothed helpfulness `(helpful + 0.5) / (total + 1.0)`.
2. **Grounding filter** — drop candidates with token Jaccard <0.1 against the record's KB reviews.
3. **Gemini judge** — only when (1) and (2) tie or fail. Uses `gemini-2.5-flash` with a constrained JSON schema validated by Pydantic; one retry on malformed JSON.

`evidence_text` is populated by KB lookup (not stored in the judge response), so the golden CSV cannot drift from the KB. [scripts/08b_validate_golden.py](scripts/08b_validate_golden.py) enforces this as a hard precondition before chunking.

---

## Evaluation Framework

Every metric below is captured per `(pipeline, k)` cell by [src/pipelines/runner.py](src/pipelines/runner.py) and written into `outputs/<pipeline>/metrics_k<k>.csv`, plus a consolidated row in `outputs/<pipeline>/summary.csv` and the cross-pipeline `outputs/results.csv`.

### Retrieval

| Metric | Definition |
|---|---|
| **Hit@K** | 1 if `evidence_doc_id` ∈ top-k, else 0 |
| **Recall@K** | Fraction of gold docs in top-k (binary in single-evidence mode) |
| **MRR** | Mean reciprocal rank of `evidence_doc_id` |
| **nDCG@K** | Normalised DCG with binary relevance |

Unanswerable rows are excluded from retrieval aggregates (no defined gold doc).

### Answer Quality

| Metric | Definition |
|---|---|
| **Exact Match** | SQuAD-normalised string equality |
| **F1 Score** | Token overlap F1 |
| **ROUGE-L** | Longest common subsequence F1 |
| **Semantic Similarity** | Cosine similarity between MiniLM embeddings of prediction and gold |
| **BERTScore F1** | Aggregate-only (`scripts/17_eval_generation.py`) |

For unanswerable rows the correct generation is a refusal — token F1 doesn't apply; answerability accuracy does.

### Faithfulness

| Metric | Source | Notes |
|---|---|---|
| **Faithfulness Score** | RAGAS (LLM-as-judge, Gemini) | k=5 only; merged in by step 24 |
| **Context Precision** | RAGAS | Retrieval-side relevance signal |
| **Context Recall** | RAGAS | Coverage of gold evidence |
| **Groundedness** | Lexical overlap of answer content tokens with retrieved context | Cheap, computed every cell |
| **Hallucination Rate** | `1 - groundedness`, mean over non-refusal rows | Cheap, computed every cell |

`answer_relevance` was dropped — multi-call, expensive, and overlaps with F1/Semantic Similarity. RAGAS is routed through Gemini explicitly via [src/llm_clients/ragas_judge.py](src/llm_clients/ragas_judge.py) so the judge is in a different model family from the generator.

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
| **Long Context Accuracy** | Answerability accuracy restricted to `q_bucket == "long"` (>= 13 tokens) |
| **Noise Robustness** | F1 on the noisiest quartile of rows; clean-vs-noisy delta logged separately |

The refusal detector ([src/generation/refusal.py](src/generation/refusal.py)) is hand-validated to ≥0.95 precision and recall on a 50-sample labelled set; [scripts/validate_refusal_detector.py](scripts/validate_refusal_detector.py) re-runs the check and fails below threshold.

|  | Model answered | Model refused |
|---|---|---|
| **Gold answerable** | Correctly Answered | Wrongly Refused |
| **Gold unanswerable** | Wrongly Answered | Correctly Refused |

---

## Outputs Layout

Every pipeline writes its own folder under `outputs/`. Cross-pipeline aggregates live at the top level.

```
outputs/
├── bm25/
│   ├── retrieval_k{1,3,5,10}.csv     # raw retrieved docs per question
│   ├── answers_k{1,3,5,10}.csv       # generated answers + retrieval payload
│   ├── metrics_k{1,3,5,10}.csv       # one-row metric snapshot per cell
│   ├── summary.csv                   # all four k values for this pipeline
│   ├── retrieval_metrics.csv         # bootstrapped Hit/Recall/MRR/nDCG (per-pipeline mirror)
│   ├── generation_metrics.csv        # bootstrapped EM/F1/ROUGE-L/Sim/Groundedness
│   ├── ragas_metrics.csv             # Faithfulness / ContextP / ContextR (k=5)
│   └── answerability_metrics.csv     # Wilson CIs + long-context + noise robustness
├── dense/        … same layout
├── sentwin/      … same layout
├── hybrid/       … same layout
├── pc/           … same layout
│
├── results.csv                       # cross-pipeline summary, ranked
├── retrieval_metrics.csv             # cross-pipeline aggregate
├── generation_metrics.csv            # cross-pipeline aggregate
├── ragas_metrics.csv                 # cross-pipeline aggregate
├── answerability_metrics.csv         # cross-pipeline aggregate
├── category_metrics.csv              # per-category breakdown (k=5)
├── qbucket_metrics.csv               # per-question-length breakdown (k=5)
├── final_ranking.csv                 # composite score + sensitivity
├── thesis_results.xlsx               # bundled tables
└── eda_plots/                        # 6 plots × 3 splits
```

---

## Statistical Reporting

Every cell in the cross-pipeline tables carries `n` and a 95% CI:

| Quantity | Method | Source |
|---|---|---|
| Continuous metrics (F1, ROUGE-L, latency, faithfulness, groundedness) | Percentile bootstrap, 1000 resamples | [src/evaluation/statistics.py](src/evaluation/statistics.py) |
| Proportions (accuracy, Hit@K, answerability) | Wilson score interval (`statsmodels`) | [src/evaluation/statistics.py](src/evaluation/statistics.py) |

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
│   └── settings.py                        # single source of truth (loads .env)
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
│   ├── sampling.py
│   ├── knowledge_base_builder.py
│   ├── golden_dataset_builder.py
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
│   │   └── runner.py                      # per-cell runner: retrieval + gen + eval
│   ├── generation/
│   │   ├── prompt.py
│   │   ├── refusal.py                     # regex-based refusal detector
│   │   └── rag_generator.py
│   ├── llm_clients/
│   │   ├── base_key_manager.py            # multi-key rotation + jittered backoff
│   │   ├── groq_key_manager.py
│   │   ├── gemini_key_manager.py          # per-instance google-genai Client
│   │   ├── ragas_judge.py
│   │   └── error_terms.py
│   ├── evaluation/
│   │   ├── retrieval_metrics.py           # Hit@K, Recall@K, MRR, nDCG@K
│   │   ├── generation_metrics.py          # EM, F1, ROUGE-L, BERTScore, Semantic Similarity
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
│   ├── 01_download_dataset.py … 25_export_excel.py
│   └── validate_refusal_detector.py
│
├── outputs/
│   ├── {bm25,dense,sentwin,hybrid,pc}/    # per-pipeline artefacts
│   └── *.csv                              # cross-pipeline aggregates + results.csv
│
└── tests/
    ├── test_*.py                          # >85% line coverage on src/
    └── test_integration_pipeline.py       # end-to-end with stubs
```

---

## Installation

### Prerequisites

- Python 3.10+
- Docker (for Qdrant)
- Groq API key(s) — generation
- Google Gemini API key(s) — judge + RAGAS

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
| `GROQ_API_KEY_1` … `GROQ_API_KEY_3` | Generator keys, rotated on quota errors |
| `GROQ_MODEL` | Default `llama-3.3-70b-versatile` |
| `GEMINI_API_KEY_1` … `GEMINI_API_KEY_2` | Judge keys |
| `GEMINI_JUDGE_MODEL` | Default `gemini-2.5-flash` |
| `QDRANT_HOST` / `QDRANT_PORT` | Default `localhost:6333` |
| `EMBEDDING_MODEL` | Default `sentence-transformers/all-MiniLM-L6-v2` |
| `RANDOM_SEED` | Default `42` (wired through every sampler / bootstrap) |
| `LLM_CACHE_DIR` | SHA256-keyed prompt cache (resumable runs) |

Multi-key rotation and exponential-jittered backoff are implemented in [src/llm_clients/base_key_manager.py](src/llm_clients/base_key_manager.py) and shared by both the Groq and Gemini concrete classes.

---

## Usage

### Data preparation (one-time)

```bash
python scripts/01_download_dataset.py
python scripts/02_load_and_standardize.py
python scripts/03_run_eda_per_split.py
python scripts/04_merge_and_clean.py
python scripts/05_stratified_sample.py
python scripts/06_build_knowledge_base.py
python scripts/07_build_golden_dataset_draft.py
python scripts/08_run_gemini_judge.py
python scripts/08b_validate_golden.py
python scripts/09_create_chunks.py
python scripts/10_build_indexes.py
```

### Smoke run + refusal-detector validation (one-time)

```bash
python scripts/11_run_bm25.py --ks 5 --sample 50
# Hand-label outputs/refusal_validation_set.csv, then:
python scripts/validate_refusal_detector.py
```

### Run pipelines incrementally — one at a time, all k

Each script runs retrieval + generation + per-cell evaluation, writes per-cell artefacts to `outputs/<pipeline>/`, and appends a row to `outputs/results.csv`. Open the CSV between runs to inspect progress.

```bash
python scripts/11_run_bm25.py             --ks 1 3 5 10
python scripts/12_run_dense.py            --ks 1 3 5 10
python scripts/13_run_sentence_window.py  --ks 1 3 5 10
python scripts/14_run_hybrid.py           --ks 1 3 5 10
python scripts/15_run_parent_child.py     --ks 1 3 5 10
```

Or one cell at a time when you want the tightest feedback loop:

```bash
python scripts/11_run_bm25.py --ks 5
```

### RAGAS, full-table aggregates, analysis, export

RAGAS is run only at k=5 and only after all 5 pipelines are done. The full-table
eval scripts (16/17/19) re-aggregate every per-pipeline `answers_k{k}.csv` they find and write
their own metric CSVs (with bootstrap/Wilson CIs) — run them whenever you want.

```bash
python scripts/16_eval_retrieval.py
python scripts/17_eval_generation.py
python scripts/18_eval_ragas.py              # all 4 k — Gemini-routed, per-row write-back
python scripts/19_eval_answerability.py
python scripts/19b_eval_hallucination.py
python scripts/20_category_analysis.py
python scripts/21_question_length_analysis.py
python scripts/22_final_ranking.py           # composite + Wilcoxon + sensitivity
python scripts/23_reproducibility_check.py   # second-seed drift report
python scripts/24_build_results_tables.py    # 6 CSVs to outputs/tables/
python scripts/25_export_excel.py            # bundle into thesis_results.xlsx
```

### LLM call budget

| Source | Calls |
|---|---|
| RAG generation (Groq) | 2,000 (5 pipelines × 4 k × 100 questions) |
| RAGAS judging (Gemini) | ~1,500 (3 metrics × 500, k=5 only) |
| Gemini golden judge | ~10–50 (one-time, only difficult rows) |
| Refusal validation | ~50 (one-time hand-labelled) |
| **Subtotal** | **~3,600** |
| Re-run buffer (×2) | **~7,200** |

Multi-key Groq rotation + on-disk prompt cache typically absorbs all retries within free-tier quota.

---

## Reproducibility

- `RANDOM_SEED=42` (env-controlled) is wired through every sklearn split, pandas sample, shuffler, and bootstrap CI.
- Dependencies pinned in [requirements.txt](requirements.txt); capture `pip freeze > pip-freeze.txt` after install.
- [datasets/processed/final_100_records.csv](datasets/processed/final_100_records.csv) and [datasets/processed/golden_dataset_100_verified.csv](datasets/processed/golden_dataset_100_verified.csv) are committed — the canonical sample and labels.
- `temperature=0` is necessary but not sufficient for full determinism on Groq; each run is repeated twice and any drift >2% on F1 is reported.
- SHA256 prompt cache at `LLM_CACHE_DIR` makes crashed runs resume in seconds.
- Every result file logs Qdrant collection version, BM25 pickle SHA, embedding model SHA, and the refusal-detector pattern hash.

---

## Tests

```bash
pytest                           # all tests including integration
pytest --cov=src                 # with coverage (target >85%)
pytest -m "not integration"      # unit only (fast)
pytest tests/test_refusal_detector.py -v
ruff check src/ tests/
mypy src/
```

The integration test ([tests/test_integration_pipeline.py](tests/test_integration_pipeline.py)) exercises the full pipeline on five fake records with stubbed Groq/Gemini clients and an in-memory retriever — no API keys, no Qdrant required.

Key manager tests cover quota-rotation, rate-limit-backoff, unknown-error propagation, exhausted-keys, and bounded backoff delay — all without ever touching the network.

---

## Dependencies

| Package | Role |
|---|---|
| `pandas`, `numpy`, `scipy`, `scikit-learn` | Data wrangling and stats |
| `statsmodels` | Wilson CI for proportions |
| `langchain`, `langchain-groq`, `langchain-google-genai` | LLM orchestration |
| `google-genai` | New per-instance Gemini SDK (replaces `google-generativeai`) |
| `sentence-transformers` | `all-MiniLM-L6-v2` embeddings + semantic similarity |
| `qdrant-client` | Vector store |
| `rank-bm25` | Lexical baseline |
| `nltk` | Sentence tokenisation for windowing |
| `ragas` | LLM-as-judge metrics |
| `rouge-score`, `bert-score` | Generation metrics |
| `pytest`, `pytest-cov`, `pytest-mock` | Testing |
| `ruff`, `black`, `mypy` | Lint, format, type-check |

Full pinning in [requirements.txt](requirements.txt).

---

## Acknowledgements

- **Dataset:** [AmazonQA](https://github.com/amazonqa/amazonqa) — review-grounded product QA corpus.
- **Generator:** Meta `llama-3.3-70b-versatile` via Groq.
- **Judge:** Google `gemini-2.5-flash` via the `google-genai` SDK.
- **Vector store:** [Qdrant](https://qdrant.tech/).
- **Evaluation:** [RAGAS](https://docs.ragas.io) — faithfulness, context precision, context recall.
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`.
