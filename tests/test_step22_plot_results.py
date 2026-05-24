"""Tests for Chapter 5 figure rendering."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config.settings import K_VALUES
from scripts import step22_plot_results as plots


def _write_csv(path: Path, rows: list[dict]) -> None:
    """Write rows to a CSV file under a temporary output directory."""
    pd.DataFrame(rows).to_csv(path, index=False)


def _metric_rows() -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Build compact aggregate metric rows for every pipeline and k."""
    results_rows: list[dict] = []
    retrieval_rows: list[dict] = []
    generation_rows: list[dict] = []
    ragas_rows: list[dict] = []

    for pipeline_index, pipeline in enumerate(plots.PIPELINE_ORDER, start=1):
        for k in K_VALUES:
            base = pipeline_index / 10
            results_rows.append(
                {
                    "pipeline_key": pipeline,
                    "K Value": k,
                    "F1 Score": base + k / 100,
                    "Faithfulness Score": "",
                    "Context Precision": "",
                    "Context Recall": "",
                    "Answerability Accuracy": 0.55 + base,
                    "Avg Latency / Question (s)": float(pipeline_index + k / 10),
                }
            )
            retrieval_rows.append(
                {
                    "pipeline": pipeline,
                    "k": k,
                    "recall_at_k": base,
                    "mrr": base / 2,
                    "ndcg_at_k": base / 3,
                }
            )
            generation_rows.append(
                {
                    "pipeline": pipeline,
                    "k": k,
                    "f1": base + k / 100,
                }
            )
            ragas_rows.append(
                {
                    "pipeline": pipeline,
                    "k": k,
                    "faithfulness": 0.45 + base,
                    "context_precision": 0.40 + base,
                    "context_recall": 0.35 + base,
                }
            )

    return results_rows, retrieval_rows, generation_rows, ragas_rows


def _slice_rows() -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Build compact category, length, answerability, and hallucination rows."""
    category_rows: list[dict] = []
    qbucket_rows: list[dict] = []
    answerability_rows: list[dict] = []
    hallucination_rows: list[dict] = []

    for pipeline_index, pipeline in enumerate(plots.PIPELINE_ORDER, start=1):
        for k in K_VALUES:
            category_rows.append(
                {
                    "pipeline": pipeline,
                    "k": k,
                    "category": "Home_and_Kitchen",
                    "f1": 0.10 * pipeline_index,
                }
            )
            qbucket_rows.append(
                {
                    "pipeline": pipeline,
                    "k": k,
                    "q_bucket": "short",
                    "f1": 0.10 * pipeline_index,
                }
            )
            answerability_rows.append(
                {
                    "pipeline": pipeline,
                    "k": k,
                    "correctly_answered": 5 + pipeline_index,
                    "correctly_refused": 4,
                    "wrongly_refused": 2,
                    "wrongly_answered": 1,
                }
            )
            hallucination_rows.append(
                {
                    "pipeline": pipeline,
                    "k": k,
                    "hallucination_rate": 0.05 * pipeline_index,
                }
            )

    return category_rows, qbucket_rows, answerability_rows, hallucination_rows


def _write_plot_inputs(output_dir: Path) -> None:
    """Write every aggregate CSV required by the plotting script."""
    results_rows, retrieval_rows, generation_rows, ragas_rows = _metric_rows()
    category_rows, qbucket_rows, answerability_rows, hallucination_rows = _slice_rows()

    _write_csv(output_dir / "results.csv", results_rows)
    _write_csv(output_dir / "retrieval_metrics.csv", retrieval_rows)
    _write_csv(output_dir / "generation_metrics.csv", generation_rows)
    _write_csv(output_dir / "ragas_metrics.csv", ragas_rows)
    _write_csv(output_dir / "category_metrics.csv", category_rows)
    _write_csv(output_dir / "qbucket_metrics.csv", qbucket_rows)
    _write_csv(output_dir / "answerability_metrics.csv", answerability_rows)
    _write_csv(output_dir / "hallucination_metrics.csv", hallucination_rows)


def test_step22_main_writes_png_only_for_all_k(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Test plotting entrypoint writes only PNG files for every k value."""
    figures_dir = tmp_path / "figures"
    figures_dir.mkdir()
    _write_plot_inputs(tmp_path)

    monkeypatch.setattr(plots, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(plots, "FIGURES_DIR", figures_dir)

    plots.main()

    png_names = {path.name for path in figures_dir.glob("*.png")}
    pdf_names = {path.name for path in figures_dir.glob("*.pdf")}

    expected_pngs = {
        "fig_3_retrieval_vs_k.png",
        "fig_4_generation_vs_k.png",
        "fig_5_ragas_vs_k.png",
    }

    for k in K_VALUES:
        expected_pngs.update(
            {
                f"fig_1_overall_pipeline_performance_k{k}.png",
                f"fig_2_accuracy_latency_tradeoff_k{k}.png",
                f"fig_6_category_heatmap_k{k}.png",
                f"fig_7_question_length_bar_k{k}.png",
                f"fig_8_answerability_outcome_stack_k{k}.png",
                f"fig_9_hallucination_rate_k{k}.png",
                f"fig_10_final_balanced_radar_k{k}.png",
            }
        )

    assert png_names == expected_pngs
    assert pdf_names == set()
