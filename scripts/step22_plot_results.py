"""Render the 10 thesis figures from the aggregate CSVs in ``outputs/``.

Each figure maps to a section in Chapter 5 of the thesis and is saved as PNG
under ``outputs/figures/``. The colour palette is the Okabe-Ito qualitative
set (Okabe & Ito, 2008), which is robust under deuteranopia, protanopia, and
tritanopia and remains distinguishable when printed in black and white.

Data sources:
    Fig 5.1 / 5.10  : results.csv + ragas_metrics.csv (joined at k=5)
    Fig 5.2         : results.csv (k=5)
    Fig 5.3         : retrieval_metrics.csv
    Fig 5.4         : generation_metrics.csv
    Fig 5.5         : ragas_metrics.csv
    Fig 5.6         : category_metrics.csv (k=5)
    Fig 5.7         : qbucket_metrics.csv (k=5)
    Fig 5.8         : answerability_metrics.csv (k=5)
    Fig 5.9         : hallucination_metrics.csv (k=5)
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config.settings import K_VALUES, OUTPUT_DIR
from src.utils.logging_config import configure_logging

configure_logging()
LOGGER = logging.getLogger(__name__)

FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

PIPELINE_ORDER: tuple[str, ...] = ("bm25", "dense", "sentwin", "hybrid", "pc")
PIPELINE_LABELS: dict[str, str] = {
    "bm25": "BM25",
    "dense": "Dense",
    "sentwin": "Sentence Window",
    "hybrid": "Hybrid",
    "pc": "Parent-Child",
}
DEFAULT_K = 5
ALL_K_VALUES: tuple[int, ...] = tuple(K_VALUES)

# --------------------------------------------------------------------------
# Okabe-Ito colour-vision-deficiency-safe palette
# Reference: https://jfly.uni-koeln.de/color/
# Hex order:  black,   orange,  sky blue, bluish green, yellow,  blue,    vermillion, reddish purple
# --------------------------------------------------------------------------
OKABE_ITO: tuple[str, ...] = (
    "#0072B2",  # blue          — BM25
    "#D55E00",  # vermillion    — Dense
    "#009E73",  # bluish green  — Sentence Window
    "#56B4E9",  # sky blue      — Hybrid
    "#CC79A7",  # reddish purple — Parent-Child
    "#E69F00",  # orange        — extra series
    "#F0E442",  # yellow        — extra series (use sparingly; light)
    "#000000",  # black         — extra series
)

PIPELINE_COLORS: dict[str, str] = dict(zip(PIPELINE_ORDER, OKABE_ITO[: len(PIPELINE_ORDER)]))

# Metric colours used by Figure 5.1 (F1 / Faithfulness / Answerability)
METRIC_COLORS: dict[str, str] = {
    "F1 Score": "#0072B2",            # blue
    "Faithfulness Score": "#D55E00",  # vermillion
    "Answerability Accuracy": "#009E73",  # bluish green
}

# Answerability outcome colours used by Figure 5.8 (4-way stacked bar)
OUTCOME_COLORS: dict[str, str] = {
    "Correctly Answered": "#009E73",  # bluish green
    "Correctly Refused": "#56B4E9",   # sky blue
    "Wrongly Refused": "#E69F00",     # orange
    "Wrongly Answered": "#D55E00",    # vermillion
}


def _apply_global_style() -> None:
    """Apply the Okabe-Ito palette and whitegrid style across all figures."""
    sns.set_theme(style="whitegrid", palette=list(OKABE_ITO), context="notebook")
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
        }
    )


def _read_csv(path: Path) -> pd.DataFrame | None:
    """Read a CSV, returning ``None`` if the file is missing."""
    if not path.exists():
        LOGGER.warning("missing input: %s — skipping dependent figure(s)", path)
        return None
    return pd.read_csv(path)


def _save(fig: plt.Figure, name: str) -> None:
    """Persist a figure as PNG under FIGURES_DIR."""
    png = FIGURES_DIR / f"{name}.png"
    fig.savefig(png)
    plt.close(fig)
    LOGGER.info("wrote %s", png.name)


def _figure_name(base_name: str, k: int) -> str:
    """Return a k-specific output name so all k plots are retained."""
    return f"{base_name}_k{k}"


def _pipeline_display(df: pd.DataFrame, key_col: str = "pipeline") -> pd.DataFrame:
    """Add ``pipeline_label`` column and an ordered categorical for plotting."""
    df = df.copy()
    df["pipeline_label"] = df[key_col].map(PIPELINE_LABELS)
    df["pipeline_label"] = pd.Categorical(
        df["pipeline_label"],
        categories=[PIPELINE_LABELS[k] for k in PIPELINE_ORDER],
        ordered=True,
    )
    return df


def _ordered_colors() -> list[str]:
    return [PIPELINE_COLORS[k] for k in PIPELINE_ORDER]


def _load_results_with_ragas(k: int = DEFAULT_K) -> pd.DataFrame | None:
    """Merge ``results.csv`` with ``ragas_metrics.csv`` at a given k.

    ``results.csv`` ships with the RAGAS columns blank; this join fills them.
    """
    results = _read_csv(OUTPUT_DIR / "results.csv")
    ragas = _read_csv(OUTPUT_DIR / "ragas_metrics.csv")
    if results is None or ragas is None:
        return None
    results_k = results[results["K Value"] == k].copy()
    ragas_k = ragas[ragas["k"] == k].rename(
        columns={
            "pipeline": "pipeline_key",
            "faithfulness": "Faithfulness Score",
            "context_precision": "Context Precision",
            "context_recall": "Context Recall",
        }
    )[["pipeline_key", "Faithfulness Score", "Context Precision", "Context Recall"]]
    merged = results_k.drop(
        columns=["Faithfulness Score", "Context Precision", "Context Recall"],
        errors="ignore",
    ).merge(ragas_k, on="pipeline_key", how="left")
    return _pipeline_display(merged, key_col="pipeline_key")


# ---------------------------------------------------------------------------
# Figure 5.1 — Overall Pipeline Performance (grouped bar at k=5)
# ---------------------------------------------------------------------------
def figure_5_1_overall_bar(k: int = DEFAULT_K) -> None:
    merged = _load_results_with_ragas(k=k)
    if merged is None:
        return
    long_df = merged.melt(
        id_vars=["pipeline_label"],
        value_vars=["F1 Score", "Faithfulness Score", "Answerability Accuracy"],
        var_name="Metric",
        value_name="Score",
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(
        data=long_df,
        x="pipeline_label",
        y="Score",
        hue="Metric",
        ax=ax,
        palette=METRIC_COLORS,
    )
    ax.set_title(f"Overall Pipeline Performance (k={k})")
    ax.set_xlabel("Pipeline")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.legend(title="Metric", loc="upper right")
    fig.tight_layout()
    _save(fig, _figure_name("fig_1_overall_pipeline_performance", k))


# ---------------------------------------------------------------------------
# Figure 5.2 — Accuracy / Latency Trade-off (scatter at k=5)
# ---------------------------------------------------------------------------
def figure_5_2_accuracy_latency(k: int = DEFAULT_K) -> None:
    merged = _load_results_with_ragas(k=k)
    if merged is None:
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(
        data=merged,
        x="Avg Latency / Question (s)",
        y="F1 Score",
        hue="pipeline_label",
        size="Answerability Accuracy",
        sizes=(80, 400),
        palette=dict(
            zip([PIPELINE_LABELS[k] for k in PIPELINE_ORDER], _ordered_colors())
        ),
        ax=ax,
        legend="brief",
    )
    for _, row in merged.iterrows():
        ax.annotate(
            f"{row['pipeline_label']}\n(ans={row['Answerability Accuracy']:.2f})",
            xy=(row["Avg Latency / Question (s)"], row["F1 Score"]),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_title(f"Answerability Accuracy vs Latency Trade off (k={k})")
    ax.set_xlabel("Average Latency per Question (s)")
    ax.set_ylabel("F1 Score")

    # Split seaborn's combined hue+size legend into two clearly-labelled legends.
    handles, labels = ax.get_legend_handles_labels()
    pipeline_handles: list = []
    pipeline_labels: list[str] = []
    size_handles: list = []
    size_labels: list[str] = []
    in_size_section = False
    for handle, label in zip(handles, labels):
        if label in {"pipeline_label", "Pipeline"}:
            continue
        if label == "Answerability Accuracy":
            in_size_section = True
            continue
        if in_size_section:
            size_handles.append(handle)
            size_labels.append(label)
        else:
            pipeline_handles.append(handle)
            pipeline_labels.append(label)

    # Re-colour the size handles so they share the colorblind palette and look
    # stylistically consistent with the pipeline legend (seaborn renders them
    # in a default grey otherwise).
    size_color = PIPELINE_COLORS[PIPELINE_ORDER[0]]
    for handle in size_handles:
        if hasattr(handle, "set_color"):
            handle.set_color(size_color)
        if hasattr(handle, "set_markerfacecolor"):
            handle.set_markerfacecolor(size_color)
        if hasattr(handle, "set_markeredgecolor"):
            handle.set_markeredgecolor(size_color)

    legend_pipeline = ax.legend(
        pipeline_handles,
        pipeline_labels,
        title="Pipeline (colour)",
        loc="lower right",
        fontsize=8,
        title_fontsize=9,
    )
    ax.add_artist(legend_pipeline)
    ax.legend(
        size_handles,
        size_labels,
        title="Answerability Accuracy (bubble size)",
        loc="lower left",
        fontsize=8,
        title_fontsize=9,
    )
    fig.tight_layout()
    _save(fig, _figure_name("fig_2_accuracy_latency_tradeoff", k))


# ---------------------------------------------------------------------------
# Figure 5.3 — Retrieval Depth Effect on Retrieval Quality (line, 3 subplots)
# ---------------------------------------------------------------------------
def figure_5_3_retrieval_vs_k() -> None:
    retrieval = _read_csv(OUTPUT_DIR / "retrieval_metrics.csv")
    if retrieval is None:
        return
    retrieval = _pipeline_display(retrieval)
    metrics = [
        ("recall_at_k", "Recall@K"),
        ("mrr", "MRR"),
        ("ndcg_at_k", "nDCG@K"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
    for ax, (col, label) in zip(axes, metrics):
        sns.lineplot(
            data=retrieval,
            x="k",
            y=col,
            hue="pipeline_label",
            hue_order=[PIPELINE_LABELS[k] for k in PIPELINE_ORDER],
            palette=_ordered_colors(),
            marker="o",
            ax=ax,
        )
        ax.set_title(label)
        ax.set_xlabel("k")
        ax.set_ylabel(label)
        ax.set_xticks(sorted(retrieval["k"].unique()))
        if ax is not axes[-1]:
            ax.get_legend().remove()
        else:
            ax.legend(title="Pipeline", loc="best", fontsize=8)
    fig.suptitle("Retrieval Depth Effect on Retrieval Quality")
    fig.tight_layout()
    _save(fig, "fig_3_retrieval_vs_k")


# ---------------------------------------------------------------------------
# Figure 5.4 — Retrieval Depth Effect on Answer Quality (F1 vs k)
# ---------------------------------------------------------------------------
def figure_5_4_generation_vs_k() -> None:
    generation = _read_csv(OUTPUT_DIR / "generation_metrics.csv")
    if generation is None:
        return
    generation = _pipeline_display(generation)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(
        data=generation,
        x="k",
        y="f1",
        hue="pipeline_label",
        hue_order=[PIPELINE_LABELS[k] for k in PIPELINE_ORDER],
        palette=_ordered_colors(),
        marker="o",
        ax=ax,
    )
    ax.set_title("Retrieval Depth Effect on Answer Quality")
    ax.set_xlabel("k")
    ax.set_ylabel("F1 Score")
    ax.set_xticks(sorted(generation["k"].unique()))
    ax.legend(title="Pipeline", loc="best")
    fig.tight_layout()
    _save(fig, "fig_4_generation_vs_k")


# ---------------------------------------------------------------------------
# Figure 5.5 — Faithfulness / Context Quality across k (3 subplots)
# ---------------------------------------------------------------------------
def figure_5_5_ragas_vs_k() -> None:
    ragas = _read_csv(OUTPUT_DIR / "ragas_metrics.csv")
    if ragas is None:
        return
    ragas = _pipeline_display(ragas)
    metrics = [
        ("faithfulness", "Faithfulness"),
        ("context_precision", "Context Precision"),
        ("context_recall", "Context Recall"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
    for ax, (col, label) in zip(axes, metrics):
        sns.lineplot(
            data=ragas,
            x="k",
            y=col,
            hue="pipeline_label",
            hue_order=[PIPELINE_LABELS[k] for k in PIPELINE_ORDER],
            palette=_ordered_colors(),
            marker="o",
            ax=ax,
        )
        ax.set_title(label)
        ax.set_xlabel("k")
        ax.set_ylabel(label)
        ax.set_xticks(sorted(ragas["k"].unique()))
        if ax is not axes[-1]:
            ax.get_legend().remove()
        else:
            ax.legend(title="Pipeline", loc="best", fontsize=8)
    fig.suptitle("Faithfulness and Context Quality across k")
    fig.tight_layout()
    _save(fig, "fig_5_ragas_vs_k")


# ---------------------------------------------------------------------------
# Figure 5.6 — Product-Category Heatmap (F1 at k=5)
# ---------------------------------------------------------------------------
def figure_5_6_category_heatmap(k: int = DEFAULT_K) -> None:
    category = _read_csv(OUTPUT_DIR / "category_metrics.csv")
    if category is None:
        return
    cat_k = category[category["k"] == k].copy()
    if cat_k.empty:
        LOGGER.warning("no rows at k=%d in category_metrics.csv", k)
        return
    cat_k["pipeline_label"] = cat_k["pipeline"].map(PIPELINE_LABELS)
    cat_k["category_label"] = cat_k["category"].str.replace("_", " ")
    pivot = cat_k.pivot_table(
        index="category_label",
        columns="pipeline_label",
        values="f1",
        aggfunc="mean",
    )
    pivot = pivot[[PIPELINE_LABELS[k] for k in PIPELINE_ORDER if PIPELINE_LABELS[k] in pivot.columns]]
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        cbar_kws={"label": "F1 Score"},
        ax=ax,
    )
    ax.set_title(f"Product Category-Level Performance Heatmap (k={k})")
    ax.set_xlabel("Pipeline")
    ax.set_ylabel("Category")
    fig.tight_layout()
    _save(fig, _figure_name("fig_6_category_heatmap", k))


# ---------------------------------------------------------------------------
# Figure 5.7 — Question-Length Bucket performance (grouped bar at k=5)
# ---------------------------------------------------------------------------
def figure_5_7_qbucket_bar(k: int = DEFAULT_K) -> None:
    qbucket = _read_csv(OUTPUT_DIR / "qbucket_metrics.csv")
    if qbucket is None:
        return
    qb_k = qbucket[qbucket["k"] == k].copy()
    if qb_k.empty:
        LOGGER.warning("no rows at k=%d in qbucket_metrics.csv", k)
        return
    qb_k = _pipeline_display(qb_k)
    qb_k["q_bucket"] = pd.Categorical(
        qb_k["q_bucket"], categories=["short", "medium", "long"], ordered=True
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(
        data=qb_k,
        x="q_bucket",
        y="f1",
        hue="pipeline_label",
        hue_order=[PIPELINE_LABELS[k] for k in PIPELINE_ORDER],
        palette=_ordered_colors(),
        ax=ax,
    )
    ax.set_title(f"F1 by Question Length (k={k})")
    ax.set_xlabel("Question Length Bucket")
    ax.set_ylabel("F1 Score")
    ax.legend(title="Pipeline", loc="best")
    fig.tight_layout()
    _save(fig, _figure_name("fig_7_question_length_bar", k))


# ---------------------------------------------------------------------------
# Figure 5.8 — Answerability Outcome Distribution (stacked bar at k=5)
# ---------------------------------------------------------------------------
def figure_5_8_answerability_stack(k: int = DEFAULT_K) -> None:
    answerability = _read_csv(OUTPUT_DIR / "answerability_metrics.csv")
    if answerability is None:
        return
    ans_k = answerability[answerability["k"] == k].copy()
    if ans_k.empty:
        LOGGER.warning("no rows at k=%d in answerability_metrics.csv", k)
        return
    ans_k = _pipeline_display(ans_k)
    ans_k = ans_k.sort_values("pipeline_label")
    outcome_cols = [
        "correctly_answered",
        "correctly_refused",
        "wrongly_refused",
        "wrongly_answered",
    ]
    pretty_names = {
        "correctly_answered": "Correctly Answered",
        "correctly_refused": "Correctly Refused",
        "wrongly_refused": "Wrongly Refused",
        "wrongly_answered": "Wrongly Answered",
    }
    plot_df = ans_k.set_index("pipeline_label")[outcome_cols].rename(columns=pretty_names)
    fig, ax = plt.subplots(figsize=(9, 5))
    plot_df.plot(
        kind="bar",
        stacked=True,
        color=[OUTCOME_COLORS[name] for name in plot_df.columns],
        ax=ax,
    )
    ax.set_title(f"Answerability Outcome Distribution (k={k})")
    ax.set_xlabel("Pipeline")
    ax.set_ylabel("Number of Questions")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Outcome", loc="upper right", fontsize=8)
    fig.tight_layout()
    _save(fig, _figure_name("fig_8_answerability_outcome_stack", k))


# ---------------------------------------------------------------------------
# Figure 5.9 — Hallucination Rate by Pipeline (bar at k=5)
# ---------------------------------------------------------------------------
def figure_5_9_hallucination_bar(k: int = DEFAULT_K) -> None:
    hallucination = _read_csv(OUTPUT_DIR / "hallucination_metrics.csv")
    if hallucination is None:
        return
    h_k = hallucination[hallucination["k"] == k].copy()
    if h_k.empty:
        LOGGER.warning("no rows at k=%d in hallucination_metrics.csv", k)
        return
    h_k = _pipeline_display(h_k)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=h_k,
        x="pipeline_label",
        y="hallucination_rate",
        hue="pipeline_label",
        order=[PIPELINE_LABELS[k] for k in PIPELINE_ORDER],
        hue_order=[PIPELINE_LABELS[k] for k in PIPELINE_ORDER],
        palette=_ordered_colors(),
        legend=False,
        ax=ax,
    )
    ax.set_title(f"Hallucination Rate by Pipeline (k={k})")
    ax.set_xlabel("Pipeline")
    ax.set_ylabel("Hallucination Rate")
    ax.set_ylim(0, max(0.05, float(h_k["hallucination_rate"].max()) * 1.25))
    for patch, value in zip(ax.patches, h_k.sort_values("pipeline_label")["hallucination_rate"]):
        ax.annotate(
            f"{value:.3f}",
            xy=(patch.get_x() + patch.get_width() / 2, patch.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    fig.tight_layout()
    _save(fig, _figure_name("fig_9_hallucination_rate", k))


# ---------------------------------------------------------------------------
# Figure 5.10 — Final Balanced Pipeline Comparison (radar at k=5)
# ---------------------------------------------------------------------------
def figure_5_10_radar(k: int = DEFAULT_K) -> None:
    merged = _load_results_with_ragas(k=k)
    if merged is None:
        return
    metric_columns = [
        ("F1 Score", "F1"),
        ("Faithfulness Score", "Faithfulness"),
        ("Context Precision", "Context Precision"),
        ("Context Recall", "Context Recall"),
        ("Answerability Accuracy", "Answerability"),
        ("Avg Latency / Question (s)", "Speed"),
    ]
    plot_df = merged.set_index("pipeline_label")[[col for col, _ in metric_columns]].copy()
    latency = plot_df["Avg Latency / Question (s)"]
    latency_min, latency_max = float(latency.min()), float(latency.max())
    if latency_max == latency_min:
        plot_df["Avg Latency / Question (s)"] = 1.0
    else:
        plot_df["Avg Latency / Question (s)"] = 1.0 - (latency - latency_min) / (
            latency_max - latency_min
        )

    labels = [pretty for _, pretty in metric_columns]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    for key in PIPELINE_ORDER:
        pretty = PIPELINE_LABELS[key]
        if pretty not in plot_df.index:
            continue
        values = plot_df.loc[pretty].tolist()
        values += values[:1]
        ax.plot(angles, values, label=pretty, color=PIPELINE_COLORS[key], linewidth=2)
        ax.fill(angles, values, color=PIPELINE_COLORS[key], alpha=0.10)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_title(
        f"Final Balanced Pipeline Comparison (k={k})\n"
        "Higher is better on every axis (Speed = 1 − normalised latency)",
        pad=20,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.30, 1.10), fontsize=9)
    fig.tight_layout()
    _save(fig, _figure_name("fig_10_final_balanced_radar", k))


def main() -> None:
    """Render all Chapter 5 figures as PNG files."""
    _apply_global_style()
    for k in ALL_K_VALUES:
        figure_5_1_overall_bar(k)
        figure_5_2_accuracy_latency(k)
    figure_5_3_retrieval_vs_k()
    figure_5_4_generation_vs_k()
    figure_5_5_ragas_vs_k()
    for k in ALL_K_VALUES:
        figure_5_6_category_heatmap(k)
        figure_5_7_qbucket_bar(k)
        figure_5_8_answerability_stack(k)
        figure_5_9_hallucination_bar(k)
        figure_5_10_radar(k)
    LOGGER.info("figures written to %s", FIGURES_DIR)


if __name__ == "__main__":
    main()
