# Comparative Evaluation of Retrieval Augmented Generation Pipelines for Review Based Question Answering in E-Commerce

**Thesis title:** Comparative Evaluation of Retrieval Augmented Generation Pipelines for Review Based Question Answering in E-Commerce: Assessing Retrieval Quality, Answer Faithfulness, and Practical Trade-Offs on the AmazonQA Dataset

**Student Name:** Hannah Abraham  
**Programme:** MSc Artificial Intelligence and Machine Learning, Liverpool John Moores University (LJMU)

## Overview

This repository contains the code, dataset preparation steps, retrieval pipelines, generation scripts, and evaluation utilities used for the MSc thesis experiments.

The project compares Retrieval-Augmented Generation (RAG) pipelines for review-based question answering in e-commerce. It uses the AmazonQA review-grounded QA dataset as the experimental corpus and evaluates retrieval quality, answer faithfulness, hallucination behaviour, and practical trade-offs such as latency.

No model weights are stored in this repository. LLM calls are made through external APIs, and vector indexes are built locally from the prepared dataset.

## Dataset

The experiments use the AmazonQA dataset, which contains product questions, answers, and review evidence.

The raw dataset is downloaded into `datasets/raw/` and processed into smaller reproducible files under `datasets/processed/`.

| Split | Download source |
|---|---|
| Train | `https://amazon-qa.s3-us-west-2.amazonaws.com/train-qar.jsonl` |
| Validation | `https://amazon-qa.s3-us-west-2.amazonaws.com/val-qar.jsonl` |
| Test | Google Drive file ID: `1A_gaYbyBUOfwi8CQ7d5OO_b91lEvSnwr` |

## Experiments Included

The repository includes five RAG pipelines:

- BM25 lexical retrieval
- Dense retrieval using sentence-transformer embeddings and Qdrant
- Sentence-window retrieval
- Hybrid BM25 plus dense retrieval
- Parent-child retrieval

Each pipeline is run at `k = 1, 3, 5, 10`, generates model answers, and writes retrieval and answer outputs for later evaluation.

## Evaluation Metrics

The experiments evaluate retrieval quality, generated answer quality, faithfulness, hallucination behaviour, and robustness.

Metrics include:

- Recall@K, MRR, nDCG@K
- Yes/No exact match
- F1 score
- ROUGE-L
- Semantic similarity
- Faithfulness
- Hallucination rate
- Context precision
- Context recall
- Answerability accuracy
- Latency

## Project Structure

```text
.
├── config/
│   └── settings.py              # Paths, model names, API settings, sample quotas
├── datasets/
│   ├── raw/                     # Downloaded AmazonQA files
│   ├── processed/               # Cleaned data, golden dataset, chunks
│   └── indexes/                 # Local retrieval indexes
├── outputs/
│   ├── bm25/                    # BM25 retrieval and answer files
│   ├── dense/                   # Dense retrieval and answer files
│   ├── sentwin/                 # Sentence-window retrieval and answer files
│   ├── hybrid/                  # Hybrid retrieval and answer files
│   ├── pc/                      # Parent-child retrieval and answer files
│   ├── figures/                 # Generated plots
│   └── *.csv                    # Aggregated result tables
├── scripts/
│   ├── step01_download_dataset.py
│   ├── step02_load_and_standardize.py
│   └── step03...step23_*.py     # Ordered experiment and evaluation scripts
├── src/
│   ├── evaluation/              # Metric implementations
│   ├── generation/              # Prompting, refusal detection, answer generation
│   ├── llm_clients/             # Groq and Gemini clients
│   ├── pipelines/               # End-to-end pipeline runner
│   ├── retrievers/              # BM25, dense, hybrid, sentence-window, parent-child
│   └── utils/                   # Shared utilities
├── tests/                       # Unit and integration tests
├── pyproject.toml               # Python dependencies and test config
├── uv.lock                      # Locked dependency versions
└── README.md
```

## Outputs

Important result files are written to `outputs/`:

- `results.csv` - final cross-pipeline comparison table
- `retrieval_metrics.csv` - retrieval metric results
- `generation_metrics.csv` - answer quality results
- `ragas_metrics.csv` - faithfulness, context precision, and context recall
- `hallucination_metrics.csv` - hallucination and refusal metrics
- `answerability_metrics.csv` - answerability results
- `category_metrics.csv` - category-wise analysis
- `qbucket_metrics.csv` - question-length analysis
- `figures/` - plots used for reporting

Pipeline-specific retrieval and answer files are stored under `outputs/bm25/`, `outputs/dense/`, `outputs/sentwin/`, `outputs/hybrid/`, and `outputs/pc/`.

## Dependencies

Core dependencies are defined in `pyproject.toml` and locked in `uv.lock`.

Main libraries used:

- `pandas`, `numpy`, `scipy`, `scikit-learn` - data processing and statistics
- `sentence-transformers` - embedding generation and semantic similarity
- `qdrant-client` - vector database access
- `rank-bm25` - BM25 retrieval
- `langchain`, `langchain-groq`, `langchain-google-genai` - LLM integration
- `google-genai` - Gemini judge client
- `ragas` - faithfulness and context evaluation
- `rouge-score` - ROUGE-L scoring
- `matplotlib`, `seaborn` - result plotting
- `pytest`, `pytest-cov`, `pytest-mock` - testing
- `ruff`, `black`, `mypy` - linting, formatting, and type checking

## Testing

Run the test suite:

```bash
pytest
```

Run tests with coverage:

```bash
pytest --cov=src
```

## Purpose of This Repository

This repository is intended to provide:

- A reproducible thesis experiment pipeline
- A clear record of dataset preparation and sampling
- Implementations of multiple RAG retrieval strategies
- Transparent answer generation and evaluation scripts
- Final result tables and plots for dissertation reporting
