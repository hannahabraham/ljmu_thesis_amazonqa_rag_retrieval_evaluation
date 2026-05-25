"""Render the thesis figures as PNG files.

The figures are saved in the outputs/figures/ folder and are used in the same order as they appear in the document.

Output files:

1_retrieval_vs_k.png
2_ragas_vs_k.png
3_category_all_k.png
4_qlength_all_k.png
5_answerability_all_k.png
6_hallucination_all_k.png
7_overall_all_k.png

The charts use the Okabe-Ito colour palette because it is clear, colour-blind friendly, and suitable for greyscale printing. Each chart title includes only the chart name, without a figure number.
"""

from __future__ import annotations

import logging
from pathlib import Path

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
DEFAULT_RADAR_K: int = 5

PIPELINE_ORDER: tuple[str, ...] = ("bm25", "dense", "sentwin", "hybrid", "pc")
PIPELINE_LABELS: dict[str, str] = {
    "bm25": "BM25",
    "dense": "Dense",
    "sentwin": "Sentence Window",
    "hybrid": "Hybrid",
    "pc": "Parent-Child",
}
PIPELINE_ORDER_LABELS: list[str] = [PIPELINE_LABELS[k] for k in PIPELINE_ORDER]

# --------------------------------------------------------------------------
# Okabe-Ito colour-vision-deficiency-safe palette
# Reference: https://jfly.uni-koeln.de/color/
# Hex order:  blue, vermillion, bluish green, sky blue, reddish purple,
#             orange, yellow, black
# --------------------------------------------------------------------------
OKABE_ITO: dict[str, str] = {
    "blue": "#0072B2",
    "verm": "#D55E00",
    "green": "#009E73",
    "sky": "#56B4E9",
    "purple": "#CC79A7",
    "orange": "#E69F00",
    "yellow": "#F0E442",
    "black": "#000000",
}

PIPELINE_COLORS: dict[str, str] = {
    "BM25": OKABE_ITO["blue"],
    "Dense": OKABE_ITO["verm"],
    "Sentence Window": OKABE_ITO["green"],
    "Hybrid": OKABE_ITO["sky"],
    "Parent-Child": OKABE_ITO["purple"],
}

METRIC_COLORS: dict[str, str] = {
    "F1 Score": OKABE_ITO["blue"],
    "Faithfulness": OKABE_ITO["verm"],
    "Answerability": OKABE_ITO["green"],
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


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _apply_global_style() -> None:
    """Apply the Okabe-Ito palette and whitegrid style across all figures."""
    sns.set_theme(style="whitegrid", palette=list(OKABE_ITO.values()), context="notebook")
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "legend.fontsize": 10,
        }
    )


def _read_csv(path: Path) -> pd.DataFrame | None:
    """Read a CSV, returning ``None`` if the file is missing."""
    if not path.exists():
        LOGGER.warning("missing input: %s — skipping dependent figure(s)", path)
        return None
    return pd.read_csv(path)


def _save(fig: plt.Figure, name: str) -> None:
    """Persist a figure as PNG under ``FIGURES_DIR``."""
    png = FIGURES_DIR / f"{name}.png"
    fig.savefig(png)
    plt.close(fig)
    LOGGER.info("wrote %s", png.name)


def _pipeline_display(df: pd.DataFrame, key_col: str = "pipeline") -> pd.DataFrame:
    """Add ``pipeline_label`` column and an ordered categorical for plotting."""
    df = df.copy()
    df["pipeline_label"] = df[key_col].map(PIPELINE_LABELS)
    df["pipeline_label"] = pd.Categorical(
        df["pipeline_label"], categories=PIPELINE_ORDER_LABELS, ordered=True
    )
    return df


def _ordered_pipeline_colors() -> list[str]:
    return [PIPELINE_COLORS[label] for label in PIPELINE_ORDER_LABELS]


# --------------------------------------------------------------------------
# 1 — Retrieval Depth Effect on Retrieval Quality 
# --------------------------------------------------------------------------
def figure_1_retrieval_vs_k() -> None:
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
            hue_order=PIPELINE_ORDER_LABELS,
            palette=_ordered_pipeline_colors(),
            marker="o",
            ax=ax,
            linewidth=2,
        )
        ax.set_title(label)
        ax.set_xlabel("k")
        ax.set_ylabel(label)
        ax.set_xticks(sorted(retrieval["k"].unique()))
        if ax is not axes[-1]:
            ax.get_legend().remove()
        else:
            ax.legend(title="Pipeline", loc="best", fontsize=9)
    fig.suptitle("Retrieval Depth Effect on Retrieval Quality", fontsize=15, fontweight="bold")
    fig.tight_layout()
    _save(fig, "1_retrieval_vs_k")


# --------------------------------------------------------------------------
# 2 — Faithfulness and Context Quality across k 
# --------------------------------------------------------------------------
def figure_2_ragas_vs_k() -> None:
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
            hue_order=PIPELINE_ORDER_LABELS,
            palette=_ordered_pipeline_colors(),
            marker="o",
            ax=ax,
            linewidth=2,
        )
        ax.set_title(label)
        ax.set_xlabel("k")
        ax.set_ylabel(label)
        ax.set_xticks(sorted(ragas["k"].unique()))
        if ax is not axes[-1]:
            ax.get_legend().remove()
        else:
            ax.legend(title="Pipeline", loc="best", fontsize=9)
    fig.suptitle("Faithfulness and Context Quality Across k", fontsize=15, fontweight="bold")
    fig.tight_layout()
    _save(fig, "2_ragas_vs_k")


# --------------------------------------------------------------------------
# 3 — Token F1 by Product Category Across k Values 
# --------------------------------------------------------------------------
def figure_3_category_all_k() -> None:
    cat = _read_csv(OUTPUT_DIR / "category_metrics.csv")
    if cat is None:
        return
    cat = _pipeline_display(cat)
    cat["category_label"] = cat["category"].str.replace("_", " ")
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5), sharey=True)
    for ax, k in zip(axes, K_VALUES):
        sub = cat[cat["k"] == k].sort_values(["category_label", "pipeline_label"])
        sns.barplot(
            data=sub,
            x="category_label",
            y="f1",
            hue="pipeline_label",
            hue_order=PIPELINE_ORDER_LABELS,
            palette=PIPELINE_COLORS,
            edgecolor="white",
            linewidth=0.5,
            ax=ax,
        )
        ax.set_title(f"k = {k}")
        ax.set_xlabel("")
        ax.set_ylabel("F1 Score" if ax is axes[0] else "")
        ax.set_ylim(0, 0.18)
        ax.tick_params(axis="x", rotation=20)
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", ncol=5,
        bbox_to_anchor=(0.5, -0.02),
        frameon=False, title="Pipeline",
    )
    fig.suptitle("Token F1 by Product Category Across k Values", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    _save(fig, "3_category_all_k")


# --------------------------------------------------------------------------
# 4 — Token F1 by Question Length Across k Values
# --------------------------------------------------------------------------
def figure_4_qlength_all_k() -> None:
    qb = _read_csv(OUTPUT_DIR / "qbucket_metrics.csv")
    if qb is None:
        return
    qb = _pipeline_display(qb)
    qb["q_bucket"] = pd.Categorical(
        qb["q_bucket"], categories=["short", "medium", "long"], ordered=True
    )
    fig, axes = plt.subplots(1, 4, figsize=(18, 5.5), sharey=True)
    for ax, k in zip(axes, K_VALUES):
        sub = qb[qb["k"] == k].sort_values(["q_bucket", "pipeline_label"])
        sns.barplot(
            data=sub,
            x="q_bucket",
            y="f1",
            hue="pipeline_label",
            hue_order=PIPELINE_ORDER_LABELS,
            palette=PIPELINE_COLORS,
            edgecolor="white",
            linewidth=0.5,
            ax=ax,
        )
        ax.set_title(f"k = {k}")
        ax.set_xlabel("")
        ax.set_ylabel("F1 Score" if ax is axes[0] else "")
        ax.set_ylim(0, 0.18)
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", ncol=5,
        bbox_to_anchor=(0.5, -0.02),
        frameon=False, title="Pipeline",
    )
    fig.suptitle("Token F1 by Question Length Across k Values", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    _save(fig, "4_qlength_all_k")


# --------------------------------------------------------------------------
# 5 — Answerability Outcomes Across k Values 
# --------------------------------------------------------------------------
def figure_5_answerability_all_k() -> None:
    ans = _read_csv(OUTPUT_DIR / "answerability_metrics.csv")
    if ans is None:
        return
    cols = ["correctly_answered", "correctly_refused", "wrongly_refused", "wrongly_answered"]
    pretty = {
        "correctly_answered": "Correctly Answered",
        "correctly_refused": "Correctly Refused",
        "wrongly_refused": "Wrongly Refused",
        "wrongly_answered": "Wrongly Answered",
    }
    fig, axes = plt.subplots(1, 4, figsize=(18, 5.5), sharey=True)
    for ax, k in zip(axes, K_VALUES):
        sub = _pipeline_display(ans[ans["k"] == k]).sort_values("pipeline_label")
        data = sub.set_index("pipeline_label")[cols].rename(columns=pretty)
        bottom = np.zeros(len(data))
        for col in data.columns:
            vals = data[col].values
            bars = ax.bar(
                data.index.astype(str),
                vals,
                bottom=bottom,
                color=OUTCOME_COLORS[col],
                edgecolor="white",
                linewidth=0.6,
                label=col,
            )
            for bar, v in zip(bars, vals):
                if v > 4:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + bar.get_height() / 2,
                        f"{int(v)}",
                        ha="center", va="center",
                        color="white", fontsize=10, fontweight="bold",
                    )
            bottom += vals
        ax.set_title(f"k = {k}")
        ax.set_xlabel("")
        ax.set_ylabel("Number of Questions" if ax is axes[0] else "")
        ax.set_ylim(0, 210)
        ax.tick_params(axis="x", rotation=30)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", ncol=4,
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
    )
    fig.suptitle("Answerability Outcomes Across k Values", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    _save(fig, "5_answerability_all_k")


# --------------------------------------------------------------------------
# 6 — Hallucination Rate Across k Values 
# --------------------------------------------------------------------------
def figure_6_hallucination_all_k() -> None:
    hall = _read_csv(OUTPUT_DIR / "hallucination_metrics.csv")
    if hall is None:
        return
    hall = _pipeline_display(hall)
    hall["k_label"] = "k = " + hall["k"].astype(str)
    hall = hall.sort_values(["pipeline_label", "k"])
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(
        data=hall,
        x="pipeline_label",
        y="hallucination_rate",
        hue="k_label",
        hue_order=[f"k = {k}" for k in K_VALUES],
        palette=K_COLORS,
        edgecolor="white",
        linewidth=0.5,
        ax=ax,
    )
    ax.set_title("Hallucination Rate Across k Values", fontsize=15, fontweight="bold")
    ax.set_xlabel("Pipeline")
    ax.set_ylabel("Hallucination Rate")
    ax.set_ylim(0, float(hall["hallucination_rate"].max()) * 1.2)
    ax.legend(title="Retrieval Depth", loc="upper right")
    for patch in ax.patches:
        h = patch.get_height()
        if h > 0:
            ax.annotate(
                f"{h:.2f}",
                xy=(patch.get_x() + patch.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", fontsize=8,
            )
    fig.tight_layout()
    _save(fig, "6_hallucination_all_k")





# --------------------------------------------------------------------------
# 7 — Overall Pipeline Performance Across k Values 
# --------------------------------------------------------------------------
def figure_7_overall_all_k() -> None:
    results = _read_csv(OUTPUT_DIR / "results.csv")
    ragas = _read_csv(OUTPUT_DIR / "ragas_metrics.csv")
    if results is None or ragas is None:
        return
    fig, axes = plt.subplots(1, 4, figsize=(18, 5.5), sharey=True)
    for ax, k in zip(axes, K_VALUES):
        rs = results[results["K Value"] == k][[
            "pipeline_key", "F1 Score", "Answerability Accuracy",
        ]].rename(columns={
            "pipeline_key": "pipeline",
            "Answerability Accuracy": "Answerability",
        })
        rg = ragas[ragas["k"] == k][["pipeline", "faithfulness"]].rename(
            columns={"faithfulness": "Faithfulness"}
        )
        df = rs.merge(rg, on="pipeline")
        df["pipeline_label"] = df["pipeline"].map(PIPELINE_LABELS)
        df["pipeline_label"] = pd.Categorical(
            df["pipeline_label"], categories=PIPELINE_ORDER_LABELS, ordered=True
        )
        df = df.sort_values("pipeline_label")
        long = df.melt(
            id_vars=["pipeline_label"],
            value_vars=["F1 Score", "Faithfulness", "Answerability"],
            var_name="Metric",
            value_name="Score",
        )
        sns.barplot(
            data=long,
            x="pipeline_label",
            y="Score",
            hue="Metric",
            ax=ax,
            palette=METRIC_COLORS,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.set_title(f"k = {k}")
        ax.set_xlabel("")
        ax.set_ylabel("Score" if ax is axes[0] else "")
        ax.set_ylim(0, 0.85)
        ax.tick_params(axis="x", rotation=30)
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", ncol=3,
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
    )
    fig.suptitle("Overall Pipeline Performance Across k Values", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    _save(fig, "7_overall_all_k")


def main() -> None:
    """Render the 8 Chapter 5 figures."""
    _apply_global_style()
    figure_1_retrieval_vs_k()
    figure_2_ragas_vs_k()
    figure_3_category_all_k()
    figure_4_qlength_all_k()
    figure_5_answerability_all_k()
    figure_6_hallucination_all_k()
    figure_7_overall_all_k()
    LOGGER.info("figures written to %s", FIGURES_DIR)


if __name__ == "__main__":
    main()
