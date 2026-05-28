"""Render Chapter 5 thesis figures as readable PNG files.

The figures follow the revised Chapter 5 plan:

1_pipeline_quality_profiles.png
    Pipeline-specific profiles across k for Recall@K, Faithfulness,
    Answerability Accuracy, and Hallucination Rate. Token F1 is shown
    only as a faint supporting line because its 0.11-0.14 band is too
    narrow to carry interpretation.
2_pipeline_latency_profiles.png
    Pipeline-specific latency profiles across k for retrieval mean,
    generation mean, and retrieval p95 latency. (Retrieval p95 is used
    rather than total p95 because retrieval p95 is what the results
    file natively records.)
3_retrieval_quality_vs_k.png
    RQ2 retrieval-depth figure for Recall@K, MRR, and nDCG@K.
4_answer_quality_faithfulness_hallucination_vs_k.png
    RQ2 answer-quality / safety figure for Faithfulness,
    Hallucination Rate, and (low-variance) Token F1.
5_latency_vs_k.png
    RQ2 latency figure for retrieval mean, generation mean,
    and retrieval p95.
6a_category_f1.png
    RQ3 category-level Token F1 with 95% CI.
6b_category_answerability.png
    RQ3 category-level Answerability Accuracy.
7a_qbucket_f1.png
    RQ4 question-length Token F1 with 95% CI.
7b_qbucket_answerability.png
    RQ4 question-length Answerability Accuracy.
8_answerability_outcomes.png
    Answerability outcome counts by pipeline and k (stacked).
9_hallucination_rate.png
    Hallucination Rate by pipeline and k (grouped bars).

The figures are saved in OUTPUT_DIR / "figures" with the
Okabe-Ito colour-blind-friendly palette and large labels for
thesis readability.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config.settings import OUTPUT_DIR
from src.utils.logging_config import configure_logging

configure_logging()
LOGGER = logging.getLogger(__name__)

FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

K_VALUES: tuple[int, ...] = (1, 3, 5, 10)
PIPELINE_ORDER: tuple[str, ...] = ("bm25", "dense", "sentwin", "hybrid", "pc")
PIPELINE_LABELS: dict[str, str] = {
    "bm25": "BM25",
    "dense": "Dense",
    "sentwin": "Sentence Window",
    "hybrid": "Hybrid",
    "pc": "Parent-Child",
}
PIPELINE_ORDER_LABELS: list[str] = [PIPELINE_LABELS[p] for p in PIPELINE_ORDER]

# Okabe-Ito colour-vision-deficiency-safe palette.
OKABE_ITO: dict[str, str] = {
    "blue": "#0072B2",
    "verm": "#D55E00",
    "green": "#009E73",
    "sky": "#56B4E9",
    "purple": "#CC79A7",
    "orange": "#E69F00",
    "yellow": "#F0E442",
    "black": "#000000",
    "grey": "#999999",
}

PIPELINE_COLORS: dict[str, str] = {
    "BM25": OKABE_ITO["blue"],
    "Dense": OKABE_ITO["verm"],
    "Sentence Window": OKABE_ITO["green"],
    "Hybrid": OKABE_ITO["sky"],
    "Parent-Child": OKABE_ITO["purple"],
}

METRIC_COLORS: dict[str, str] = {
    "Recall@K": OKABE_ITO["blue"],
    "Faithfulness": OKABE_ITO["purple"],
    "Answerability": OKABE_ITO["orange"],
    "Hallucination": OKABE_ITO["verm"],
    "Token F1 (low variance)": OKABE_ITO["grey"],
    "Token F1": OKABE_ITO["green"],
    "Retrieval mean": OKABE_ITO["blue"],
    "Generation mean": OKABE_ITO["green"],
    "Retrieval p95": OKABE_ITO["verm"],
}

OUTCOME_COLORS: dict[str, str] = {
    "Correctly Answered": OKABE_ITO["green"],
    "Correctly Refused": OKABE_ITO["sky"],
    "Wrongly Refused": OKABE_ITO["orange"],
    "Wrongly Answered": OKABE_ITO["verm"],
}

K_COLORS: list[str] = [
    OKABE_ITO["blue"],
    OKABE_ITO["green"],
    OKABE_ITO["orange"],
    OKABE_ITO["verm"],
]

CATEGORY_LABELS: dict[str, str] = {
    "electronics": "Electronics",
    "toys_and_games": "Toys & Games",
    "toys games": "Toys & Games",
    "health_personal_care": "Health & Personal Care",
    "health and personal care": "Health & Personal Care",
    "home_and_kitchen": "Home & Kitchen",
    "home and kitchen": "Home & Kitchen",
}

Q_BUCKET_LABELS: dict[str, str] = {
    "short": "Short",
    "medium": "Medium",
    "long": "Long",
}


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------
def _apply_global_style() -> None:
    """Apply a consistent thesis-friendly style across all figures."""
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "legend.title_fontsize": 10.5,
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _read_csv(path: Path) -> pd.DataFrame | None:
    """Read a CSV file, returning None when it is unavailable."""
    if not path.exists():
        LOGGER.warning("missing input: %s; skipping dependent figure(s)", path)
        return None
    return pd.read_csv(path)


def _save(fig: plt.Figure, name: str) -> None:
    """Save a figure as PNG under FIGURES_DIR."""
    png_path = FIGURES_DIR / f"{name}.png"
    fig.savefig(png_path, facecolor="white")
    plt.close(fig)
    LOGGER.info("wrote %s", png_path)


def _has_required_columns(df: pd.DataFrame, columns: Iterable[str], figure_name: str) -> bool:
    """Return whether df contains all required columns for a figure."""
    missing_columns = sorted(set(columns) - set(df.columns))
    if missing_columns:
        LOGGER.warning(
            "skipping %s; missing column(s): %s",
            figure_name,
            ", ".join(missing_columns),
        )
        return False
    return True


def _pipeline_display(df: pd.DataFrame, key_col: str = "pipeline") -> pd.DataFrame:
    """Add an ordered display label column for pipeline plots."""
    display_df = df.copy()
    display_df["pipeline_label"] = display_df[key_col].map(PIPELINE_LABELS)
    display_df["pipeline_label"] = pd.Categorical(
        display_df["pipeline_label"], categories=PIPELINE_ORDER_LABELS, ordered=True
    )
    return display_df


def _ordered_pipeline_colors() -> list[str]:
    return [PIPELINE_COLORS[label] for label in PIPELINE_ORDER_LABELS]


def _remove_legend(ax: plt.Axes) -> None:
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()


def _legend_below(fig: plt.Figure, handles: Sequence, labels: Sequence[str], ncol: int = 5) -> None:
    """Place one clean legend below a multi-panel figure."""
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=ncol,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )


def _format_axis_0_1(ax: plt.Axes, ylabel: str | None = None) -> None:
    ax.set_ylim(0, 1.02)
    ax.set_yticks(np.linspace(0, 1, 6))
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.35)
    ax.grid(axis="x", alpha=0.15)


def _set_k_axis(ax: plt.Axes) -> None:
    ax.set_xticks(list(K_VALUES))
    ax.set_xlabel("Retrieval depth (k)")


def _clean_category_labels(series: pd.Series) -> pd.Series:
    raw = series.astype(str).str.replace("_", " ").str.strip().str.lower()
    mapped = raw.map(CATEGORY_LABELS)
    return mapped.fillna(raw.str.title())


def _clean_qbucket_labels(series: pd.Series) -> pd.Series:
    raw = series.astype(str).str.strip().str.lower()
    mapped = raw.map(Q_BUCKET_LABELS)
    return mapped.fillna(raw.str.title())


def _add_manual_error_bars(
    ax: plt.Axes,
    grouped_data: pd.DataFrame,
    x_col: str,
    y_col: str,
    low_col: str,
    high_col: str,
    hue_col: str,
    x_order: Sequence[str],
    hue_order: Sequence[str],
) -> None:
    """Add CI error bars to a seaborn grouped bar chart."""
    data_lookup = {
        (str(row[x_col]), str(row[hue_col])): row
        for _, row in grouped_data.iterrows()
    }
    patches = ax.patches
    expected = len(x_order) * len(hue_order)
    if len(patches) < expected:
        return

    patch_index = 0
    for x_value in x_order:
        for hue_value in hue_order:
            if patch_index >= len(patches):
                return
            patch = patches[patch_index]
            patch_index += 1
            row = data_lookup.get((str(x_value), str(hue_value)))
            if row is None or pd.isna(row[low_col]) or pd.isna(row[high_col]):
                continue
            y = float(row[y_col])
            yerr_low = max(0.0, y - float(row[low_col]))
            yerr_high = max(0.0, float(row[high_col]) - y)
            ax.errorbar(
                patch.get_x() + patch.get_width() / 2,
                y,
                yerr=[[yerr_low], [yerr_high]],
                fmt="none",
                ecolor="#333333",
                elinewidth=0.8,
                capsize=2,
                capthick=0.8,
                zorder=5,
            )


def _read_overall_quality_inputs() -> pd.DataFrame | None:
    """Merge per-k metrics needed for pipeline profile and RQ2 figures."""
    retrieval = _read_csv(OUTPUT_DIR / "retrieval_metrics.csv")
    ragas = _read_csv(OUTPUT_DIR / "ragas_metrics.csv")
    answerability = _read_csv(OUTPUT_DIR / "answerability_metrics.csv")
    hallucination = _read_csv(OUTPUT_DIR / "hallucination_metrics.csv")
    generation = _read_csv(OUTPUT_DIR / "generation_metrics.csv")

    required_inputs = [retrieval, ragas, answerability, hallucination]
    if any(df is None for df in required_inputs):
        return None

    assert retrieval is not None and ragas is not None and answerability is not None and hallucination is not None

    # answerability_metrics.csv stores the accuracy under either
    # answerability_acc (current) or answerability_accuracy (legacy).
    ans_col = (
        "answerability_acc"
        if "answerability_acc" in answerability.columns
        else "answerability_accuracy"
    )
    checks = [
        (retrieval, {"pipeline", "k", "recall_at_k", "mrr", "ndcg_at_k"}, "overall/retrieval"),
        (ragas, {"pipeline", "k", "faithfulness"}, "overall/ragas"),
        (answerability, {"pipeline", "k", ans_col}, "overall/answerability"),
        (hallucination, {"pipeline", "k", "hallucination_rate"}, "overall/hallucination"),
    ]
    if not all(_has_required_columns(df, cols, name) for df, cols, name in checks):
        return None

    merged = retrieval[["pipeline", "k", "recall_at_k", "mrr", "ndcg_at_k"]].merge(
        ragas[["pipeline", "k", "faithfulness"]], on=["pipeline", "k"], how="inner"
    )
    merged = merged.merge(
        answerability[["pipeline", "k", ans_col]].rename(
            columns={ans_col: "answerability_accuracy"}
        ),
        on=["pipeline", "k"],
        how="inner",
    )
    merged = merged.merge(
        hallucination[["pipeline", "k", "hallucination_rate"]],
        on=["pipeline", "k"],
        how="inner",
    )

    # Prefer Token F1 from generation_metrics.csv because it already exposes
    # f1 keyed on (pipeline, k) cleanly. Fall back to results.csv otherwise.
    token_f1: pd.DataFrame | None = None
    if generation is not None and {"pipeline", "k", "f1"}.issubset(generation.columns):
        token_f1 = generation[["pipeline", "k", "f1"]].rename(columns={"f1": "token_f1"})
    else:
        results = _read_csv(OUTPUT_DIR / "results.csv")
        if results is not None:
            if {"pipeline_key", "K Value", "F1 Score"}.issubset(results.columns):
                token_f1 = results[["pipeline_key", "K Value", "F1 Score"]].rename(
                    columns={"pipeline_key": "pipeline", "K Value": "k", "F1 Score": "token_f1"}
                )
            elif {"pipeline_key", "K Value", "F1"}.issubset(results.columns):
                token_f1 = results[["pipeline_key", "K Value", "F1"]].rename(
                    columns={"pipeline_key": "pipeline", "K Value": "k", "F1": "token_f1"}
                )
            elif {"pipeline", "k", "f1"}.issubset(results.columns):
                token_f1 = results[["pipeline", "k", "f1"]].rename(columns={"f1": "token_f1"})
    if token_f1 is not None:
        merged = merged.merge(token_f1, on=["pipeline", "k"], how="left")

    if "token_f1" not in merged.columns or merged["token_f1"].isna().all():
        LOGGER.warning(
            "Token F1 not found; figures referencing Token F1 will omit it."
        )
    return _pipeline_display(merged)


# ---------------------------------------------------------------------------
# 1 - Pipeline Performance Profiles Across Retrieval Depths
# ---------------------------------------------------------------------------
def figure_1_pipeline_quality_profiles() -> None:
    """Faceted per-pipeline profiles across k.

    Token F1 is plotted as a faint grey supporting line so its
    0.11-0.14 band does not draw the eye away from the headline
    quality and safety metrics.
    """
    quality = _read_overall_quality_inputs()
    if quality is None:
        return

    main_metrics: list[tuple[str, str]] = [
        ("recall_at_k", "Recall@K"),
        ("faithfulness", "Faithfulness"),
        ("answerability_accuracy", "Answerability"),
        ("hallucination_rate", "Hallucination"),
    ]
    show_token_f1 = "token_f1" in quality.columns and not quality["token_f1"].isna().all()

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for ax, pipeline_label in zip(axes_flat, PIPELINE_ORDER_LABELS):
        subset = quality[quality["pipeline_label"] == pipeline_label]
        # Headline metrics in bold colour.
        for column, label in main_metrics:
            ax.plot(
                subset["k"],
                subset[column],
                marker="o",
                markersize=6,
                linewidth=2.2,
                color=METRIC_COLORS[label],
                label=label,
            )
        # Token F1 faint supporting line so the reader sees it but does
        # not interpret the narrow band as meaningful.
        if show_token_f1:
            ax.plot(
                subset["k"],
                subset["token_f1"],
                marker="s",
                markersize=4,
                linewidth=1.2,
                linestyle="--",
                alpha=0.55,
                color=METRIC_COLORS["Token F1 (low variance)"],
                label="Token F1 (low variance)",
            )
        ax.set_title(pipeline_label)
        _set_k_axis(ax)
        _format_axis_0_1(ax, ylabel="Score")
        ax.tick_params(axis="x", labelbottom=True)
        ax.tick_params(axis="y", labelleft=True)
        _remove_legend(ax)

    axes_flat[-1].axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    _legend_below(fig, handles, labels, ncol=len(handles))
    fig.suptitle(
        "Pipeline Performance Profiles Across Retrieval Depths",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    _save(fig, "1_pipeline_quality_profiles")


# ---------------------------------------------------------------------------
# 2 - Pipeline Latency Profiles Across Retrieval Depths
# ---------------------------------------------------------------------------
def figure_2_pipeline_latency_profiles() -> None:
    """Faceted latency profiles for each pipeline.

    The plan calls for retrieval p95 rather than total p95, because the
    results file natively records retrieval p95 only.
    """
    latency = _read_csv(OUTPUT_DIR / "latency_detail.csv")
    if latency is None:
        return

    required = {"pipeline", "k", "retrieval_ms_mean", "generation_ms_mean", "retrieval_p95_ms"}
    if not _has_required_columns(latency, required, "pipeline_latency_profiles"):
        return

    latency = _pipeline_display(latency)
    metrics = [
        ("retrieval_ms_mean", "Retrieval mean"),
        ("generation_ms_mean", "Generation mean"),
        ("retrieval_p95_ms", "Retrieval p95"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True, sharey=False)
    axes_flat = axes.flatten()

    for ax, pipeline_label in zip(axes_flat, PIPELINE_ORDER_LABELS):
        subset = latency[latency["pipeline_label"] == pipeline_label]
        for column, label in metrics:
            ax.plot(
                subset["k"],
                subset[column],
                marker="o",
                markersize=6,
                linewidth=2.2,
                color=METRIC_COLORS[label],
                label=label,
            )
        ax.set_title(pipeline_label)
        _set_k_axis(ax)
        ax.set_ylabel("Latency (ms)")
        ax.tick_params(axis="x", labelbottom=True)
        ax.tick_params(axis="y", labelleft=True)
        ax.grid(axis="y", alpha=0.35)
        ax.grid(axis="x", alpha=0.15)
        _remove_legend(ax)

    axes_flat[-1].axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    _legend_below(fig, handles, labels, ncol=3)
    fig.suptitle(
        "Pipeline Latency Profiles Across Retrieval Depths",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    _save(fig, "2_pipeline_latency_profiles")


# ---------------------------------------------------------------------------
# 3 - Retrieval Quality Across Retrieval Depths
# ---------------------------------------------------------------------------
def figure_3_retrieval_quality_vs_k() -> None:
    """RQ2 retrieval-depth figure for Recall@K, MRR, and nDCG@K."""
    retrieval = _read_csv(OUTPUT_DIR / "retrieval_metrics.csv")
    if retrieval is None:
        return

    required = {"pipeline", "k", "recall_at_k", "mrr", "ndcg_at_k"}
    if not _has_required_columns(retrieval, required, "retrieval_quality_vs_k"):
        return

    retrieval = _pipeline_display(retrieval)
    metrics: Sequence[tuple[str, str]] = (
        ("recall_at_k", "Recall@K"),
        ("mrr", "MRR"),
        ("ndcg_at_k", "nDCG@K"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5), sharex=True)

    for ax, (column, label) in zip(axes, metrics):
        sns.lineplot(
            data=retrieval,
            x="k",
            y=column,
            hue="pipeline_label",
            hue_order=PIPELINE_ORDER_LABELS,
            palette=_ordered_pipeline_colors(),
            marker="o",
            markersize=7,
            linewidth=2.2,
            ax=ax,
        )
        ax.set_title(label)
        _set_k_axis(ax)
        _format_axis_0_1(ax, ylabel=label)
        ax.tick_params(axis="x", labelbottom=True)
        ax.tick_params(axis="y", labelleft=True)
        _remove_legend(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    _legend_below(fig, handles, labels, ncol=5)
    fig.suptitle("Retrieval Quality Across Retrieval Depths", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0.10, 1, 0.93])
    _save(fig, "3_retrieval_quality_vs_k")


# ---------------------------------------------------------------------------
# 4 - Answer Quality, Faithfulness, and Hallucination Across k
# ---------------------------------------------------------------------------
def figure_4_answer_quality_faithfulness_hallucination_vs_k() -> None:
    """RQ2 figure for Faithfulness, Hallucination Rate, and (low-variance) Token F1."""
    quality = _read_overall_quality_inputs()
    if quality is None:
        return

    metrics: list[tuple[str, str]] = [
        ("faithfulness", "Faithfulness"),
        ("hallucination_rate", "Hallucination Rate"),
    ]
    if "token_f1" in quality.columns and not quality["token_f1"].isna().all():
        metrics.append(("token_f1", "Token F1 (low variance)"))

    fig, axes = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), 5), sharex=True)
    if len(metrics) == 1:
        axes = [axes]  # type: ignore[assignment]

    for ax, (column, label) in zip(axes, metrics):
        sns.lineplot(
            data=quality,
            x="k",
            y=column,
            hue="pipeline_label",
            hue_order=PIPELINE_ORDER_LABELS,
            palette=_ordered_pipeline_colors(),
            marker="o",
            markersize=7,
            linewidth=2.2,
            ax=ax,
        )
        ax.set_title(label)
        _set_k_axis(ax)
        if column == "token_f1":
            ax.set_ylim(0.0, max(0.20, float(quality["token_f1"].max()) * 1.3))
            ax.set_ylabel("Token F1")
            ax.grid(axis="y", alpha=0.35)
            ax.grid(axis="x", alpha=0.15)
        else:
            _format_axis_0_1(ax, ylabel=label)
        ax.tick_params(axis="x", labelbottom=True)
        ax.tick_params(axis="y", labelleft=True)
        _remove_legend(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    _legend_below(fig, handles, labels, ncol=5)
    fig.suptitle(
        "Answer Quality, Faithfulness, and Hallucination Across Retrieval Depths",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.10, 1, 0.93])
    _save(fig, "4_answer_quality_faithfulness_hallucination_vs_k")


# ---------------------------------------------------------------------------
# 5 - Latency Across Retrieval Depths
# ---------------------------------------------------------------------------
def figure_5_latency_vs_k() -> None:
    """RQ2 latency figure using retrieval mean, generation mean, and retrieval p95."""
    latency = _read_csv(OUTPUT_DIR / "latency_detail.csv")
    if latency is None:
        return

    required = {"pipeline", "k", "retrieval_ms_mean", "generation_ms_mean", "retrieval_p95_ms"}
    if not _has_required_columns(latency, required, "latency_vs_k"):
        return

    latency = _pipeline_display(latency)
    metrics: Sequence[tuple[str, str, str]] = (
        ("retrieval_ms_mean", "Retrieval Mean", "Latency (ms)"),
        ("generation_ms_mean", "Generation Mean", "Latency (ms)"),
        ("retrieval_p95_ms", "Retrieval p95", "Latency (ms)"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5), sharex=True)

    for ax, (column, label, ylabel) in zip(axes, metrics):
        sns.lineplot(
            data=latency,
            x="k",
            y=column,
            hue="pipeline_label",
            hue_order=PIPELINE_ORDER_LABELS,
            palette=_ordered_pipeline_colors(),
            marker="o",
            markersize=7,
            linewidth=2.2,
            ax=ax,
        )
        ax.set_title(label)
        _set_k_axis(ax)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", labelbottom=True)
        ax.tick_params(axis="y", labelleft=True)
        ax.grid(axis="y", alpha=0.35)
        ax.grid(axis="x", alpha=0.15)
        _remove_legend(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    _legend_below(fig, handles, labels, ncol=5)
    fig.suptitle("Latency Across Retrieval Depths", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0.06, 1, 0.93])
    _save(fig, "5_latency_vs_k")


# ---------------------------------------------------------------------------
# 6 - Product Category: Token F1 and Answerability (separate files)
# ---------------------------------------------------------------------------
def _load_category_metrics() -> tuple[pd.DataFrame, list[str]] | None:
    category = _read_csv(OUTPUT_DIR / "category_metrics.csv")
    if category is None:
        return None
    required = {"pipeline", "k", "category", "f1", "answerability_acc"}
    if not _has_required_columns(category, required, "category_metrics"):
        return None
    category = _pipeline_display(category)
    category["category_label"] = _clean_category_labels(category["category"])
    category_order = sorted(category["category_label"].dropna().unique())
    return category, category_order


def figure_6a_category_f1() -> None:
    """RQ3 category-level Token F1 (1x4 across k)."""
    loaded = _load_category_metrics()
    if loaded is None:
        return
    category, category_order = loaded

    fig, axes = plt.subplots(1, 4, figsize=(19, 5.2), sharey=True)
    for col_idx, k_value in enumerate(K_VALUES):
        subset = category[category["k"] == k_value].sort_values(["category_label", "pipeline_label"])
        ax = axes[col_idx]
        sns.barplot(
            data=subset,
            x="category_label",
            y="f1",
            hue="pipeline_label",
            order=category_order,
            hue_order=PIPELINE_ORDER_LABELS,
            palette=PIPELINE_COLORS,
            edgecolor="white",
            linewidth=0.4,
            ax=ax,
        )
        if {"f1_lo", "f1_hi"}.issubset(subset.columns):
            _add_manual_error_bars(
                ax,
                subset,
                x_col="category_label",
                y_col="f1",
                low_col="f1_lo",
                high_col="f1_hi",
                hue_col="pipeline_label",
                x_order=category_order,
                hue_order=PIPELINE_ORDER_LABELS,
            )
        ax.set_title(f"k = {k_value}")
        ax.set_xlabel("Product Category")
        ax.set_ylabel("Token F1 (with 95% CI)")
        ax.set_ylim(0, max(0.25, float(category["f1_hi"].max()) * 1.1))
        ax.tick_params(axis="x", rotation=25)
        ax.tick_params(axis="y", labelleft=True)
        _remove_legend(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    _legend_below(fig, handles, labels, ncol=5)
    fig.suptitle("Token F1 by Product Category and Pipeline", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0.08, 1, 0.93])
    _save(fig, "6a_category_f1")


def figure_6b_category_answerability() -> None:
    """RQ3 category-level Answerability Accuracy (1x4 across k)."""
    loaded = _load_category_metrics()
    if loaded is None:
        return
    category, category_order = loaded

    fig, axes = plt.subplots(1, 4, figsize=(19, 5.2), sharey=True)
    for col_idx, k_value in enumerate(K_VALUES):
        subset = category[category["k"] == k_value].sort_values(["category_label", "pipeline_label"])
        ax = axes[col_idx]
        sns.barplot(
            data=subset,
            x="category_label",
            y="answerability_acc",
            hue="pipeline_label",
            order=category_order,
            hue_order=PIPELINE_ORDER_LABELS,
            palette=PIPELINE_COLORS,
            edgecolor="white",
            linewidth=0.4,
            ax=ax,
        )
        ax.set_title(f"k = {k_value}")
        ax.set_xlabel("Product Category")
        ax.set_ylabel("Answerability Accuracy")
        ax.set_ylim(0, 1.02)
        ax.tick_params(axis="x", rotation=25)
        ax.tick_params(axis="y", labelleft=True)
        _remove_legend(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    _legend_below(fig, handles, labels, ncol=5)
    fig.suptitle("Answerability Accuracy by Product Category and Pipeline", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0.08, 1, 0.93])
    _save(fig, "6b_category_answerability")


# ---------------------------------------------------------------------------
# 7 - Question-Length: Token F1 and Answerability (separate files)
# ---------------------------------------------------------------------------
def _load_qbucket_metrics() -> tuple[pd.DataFrame, list[str]] | None:
    qbucket = _read_csv(OUTPUT_DIR / "qbucket_metrics.csv")
    if qbucket is None:
        return None
    required = {"pipeline", "k", "q_bucket", "f1", "answerability_acc"}
    if not _has_required_columns(qbucket, required, "qbucket_metrics"):
        return None
    qbucket = _pipeline_display(qbucket)
    qbucket["q_bucket_label"] = _clean_qbucket_labels(qbucket["q_bucket"])
    bucket_order = ["Short", "Medium", "Long"]
    return qbucket, bucket_order


def figure_7a_qbucket_f1() -> None:
    """RQ4 question-length Token F1 (1x4 across k)."""
    loaded = _load_qbucket_metrics()
    if loaded is None:
        return
    qbucket, bucket_order = loaded

    fig, axes = plt.subplots(1, 4, figsize=(18, 5.2), sharey=True)
    for col_idx, k_value in enumerate(K_VALUES):
        subset = qbucket[qbucket["k"] == k_value].sort_values(["q_bucket_label", "pipeline_label"])
        ax = axes[col_idx]
        sns.barplot(
            data=subset,
            x="q_bucket_label",
            y="f1",
            hue="pipeline_label",
            order=bucket_order,
            hue_order=PIPELINE_ORDER_LABELS,
            palette=PIPELINE_COLORS,
            edgecolor="white",
            linewidth=0.4,
            ax=ax,
        )
        if {"f1_lo", "f1_hi"}.issubset(subset.columns):
            _add_manual_error_bars(
                ax,
                subset,
                x_col="q_bucket_label",
                y_col="f1",
                low_col="f1_lo",
                high_col="f1_hi",
                hue_col="pipeline_label",
                x_order=bucket_order,
                hue_order=PIPELINE_ORDER_LABELS,
            )
        ax.set_title(f"k = {k_value}")
        ax.set_xlabel("Question Length")
        ax.set_ylabel("Token F1 (with 95% CI)")
        ax.set_ylim(0, max(0.22, float(qbucket["f1_hi"].max()) * 1.1))
        ax.tick_params(axis="y", labelleft=True)
        _remove_legend(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    _legend_below(fig, handles, labels, ncol=5)
    fig.suptitle("Token F1 by Question Length and Pipeline", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0.08, 1, 0.93])
    _save(fig, "7a_qbucket_f1")


def figure_7b_qbucket_answerability() -> None:
    """RQ4 question-length Answerability Accuracy (1x4 across k)."""
    loaded = _load_qbucket_metrics()
    if loaded is None:
        return
    qbucket, bucket_order = loaded

    fig, axes = plt.subplots(1, 4, figsize=(18, 5.2), sharey=True)
    for col_idx, k_value in enumerate(K_VALUES):
        subset = qbucket[qbucket["k"] == k_value].sort_values(["q_bucket_label", "pipeline_label"])
        ax = axes[col_idx]
        sns.barplot(
            data=subset,
            x="q_bucket_label",
            y="answerability_acc",
            hue="pipeline_label",
            order=bucket_order,
            hue_order=PIPELINE_ORDER_LABELS,
            palette=PIPELINE_COLORS,
            edgecolor="white",
            linewidth=0.4,
            ax=ax,
        )
        ax.set_title(f"k = {k_value}")
        ax.set_xlabel("Question Length")
        ax.set_ylabel("Answerability Accuracy")
        ax.set_ylim(0, 1.02)
        ax.tick_params(axis="y", labelleft=True)
        _remove_legend(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    _legend_below(fig, handles, labels, ncol=5)
    fig.suptitle("Answerability Accuracy by Question Length and Pipeline", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0.08, 1, 0.93])
    _save(fig, "7b_qbucket_answerability")


# ---------------------------------------------------------------------------
# 8 - Answerability Outcomes Across Pipelines and k
# ---------------------------------------------------------------------------
def figure_8_answerability_outcomes() -> None:
    """Stacked answerability outcome counts by pipeline and k."""
    answerability = _read_csv(OUTPUT_DIR / "answerability_metrics.csv")
    if answerability is None:
        return

    outcome_columns = ["correctly_answered", "correctly_refused", "wrongly_refused", "wrongly_answered"]
    required = {"pipeline", "k", *outcome_columns}
    if not _has_required_columns(answerability, required, "answerability_outcomes"):
        return

    pretty_labels = {
        "correctly_answered": "Correctly Answered",
        "correctly_refused": "Correctly Refused",
        "wrongly_refused": "Wrongly Refused",
        "wrongly_answered": "Wrongly Answered",
    }
    outcome_order = [
        "Correctly Answered",
        "Correctly Refused",
        "Wrongly Refused",
        "Wrongly Answered",
    ]

    fig, axes = plt.subplots(1, 4, figsize=(19, 6), sharey=True)
    for ax, k_value in zip(axes, K_VALUES):
        subset = _pipeline_display(answerability[answerability["k"] == k_value]).sort_values("pipeline_label")
        plot_data = subset.set_index("pipeline_label")[outcome_columns].rename(columns=pretty_labels)
        plot_data = plot_data[outcome_order]
        bottom = np.zeros(len(plot_data))

        for outcome in outcome_order:
            values = plot_data[outcome].to_numpy()
            bars = ax.bar(
                plot_data.index.astype(str),
                values,
                bottom=bottom,
                color=OUTCOME_COLORS[outcome],
                edgecolor="white",
                linewidth=0.6,
                label=outcome,
            )
            for bar, value in zip(bars, values):
                if value >= 8:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + bar.get_height() / 2,
                        f"{int(value)}",
                        ha="center",
                        va="center",
                        color="white" if outcome != "Wrongly Refused" else "black",
                        fontsize=9,
                        fontweight="bold",
                    )
            bottom += values

        ax.set_title(f"k = {k_value}")
        ax.set_xlabel("Pipeline")
        ax.set_ylabel("Number of Questions")
        ax.tick_params(axis="x", rotation=30)
        ax.tick_params(axis="y", labelleft=True)
        ax.set_ylim(0, max(210, int(bottom.max() * 1.05)))

    handles, labels = axes[0].get_legend_handles_labels()
    _legend_below(fig, handles, labels, ncol=4)
    fig.suptitle("Answerability Outcomes Across Pipelines and Retrieval Depths", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0.10, 1, 0.94])
    _save(fig, "8_answerability_outcomes")


# ---------------------------------------------------------------------------
# 9 - Hallucination Rate by Pipeline and k
# ---------------------------------------------------------------------------
def figure_9_hallucination_rate() -> None:
    """Grouped bar chart for Hallucination Rate by pipeline and k."""
    hallucination = _read_csv(OUTPUT_DIR / "hallucination_metrics.csv")
    if hallucination is None:
        return

    required = {"pipeline", "k", "hallucination_rate"}
    if not _has_required_columns(hallucination, required, "hallucination_rate"):
        return

    hallucination = _pipeline_display(hallucination)
    hallucination["k_label"] = "k = " + hallucination["k"].astype(str)
    hallucination = hallucination.sort_values(["pipeline_label", "k"])

    fig, ax = plt.subplots(figsize=(12, 6.5))
    sns.barplot(
        data=hallucination,
        x="pipeline_label",
        y="hallucination_rate",
        hue="k_label",
        hue_order=[f"k = {k}" for k in K_VALUES],
        palette=K_COLORS,
        edgecolor="white",
        linewidth=0.5,
        ax=ax,
    )
    ax.set_title("Hallucination Rate by Pipeline and Retrieval Depth", fontsize=15, fontweight="bold")
    ax.set_xlabel("Pipeline")
    ax.set_ylabel("Hallucination Rate (lower is better)")
    ax.set_ylim(0, max(0.30, float(hallucination["hallucination_rate"].max()) * 1.25))
    ax.tick_params(axis="x", rotation=15)
    ax.legend(title="Retrieval Depth", loc="upper right", frameon=True)

    for patch in ax.patches:
        height = patch.get_height()
        if height > 0:
            ax.annotate(
                f"{height:.3f}",
                xy=(patch.get_x() + patch.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )

    fig.tight_layout()
    _save(fig, "9_hallucination_rate")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Render all revised Chapter 5 figures."""
    _apply_global_style()
    figure_1_pipeline_quality_profiles()
    figure_2_pipeline_latency_profiles()
    figure_3_retrieval_quality_vs_k()
    figure_4_answer_quality_faithfulness_hallucination_vs_k()
    figure_5_latency_vs_k()
    figure_6a_category_f1()
    figure_6b_category_answerability()
    figure_7a_qbucket_f1()
    figure_7b_qbucket_answerability()
    figure_8_answerability_outcomes()
    figure_9_hallucination_rate()
    LOGGER.info("figures written to %s", FIGURES_DIR)


if __name__ == "__main__":
    main()
