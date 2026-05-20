"""Plot the cross-pipeline results as publication-quality seaborn figures.

Reads the aggregate CSVs written by steps 16–22 and writes one figure per
research dimension into ``outputs/figures/``. Each figure is saved as PNG
(300 dpi) and, where useful, PDF for thesis embedding.

Run after the per-pipeline runners (11–15) and the eval scripts (16–19b)
have populated ``outputs/``. Missing inputs are skipped with a warning, so
the script is safe to run incrementally as cells fill in.
"""

from __future__ import annotations

import argparse
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

PIPELINE_ORDER: list[str] = ["bm25", "dense", "sentwin", "hybrid", "pc"]
PIPELINE_LABELS: dict[str, str] = {
    "bm25": "BM25",
    "dense": "Dense",
    "sentwin": "Sentence Window",
    "hybrid": "Hybrid (RRF)",
    "pc": "Parent-Child",
}
K_ORDER: list[int] = [1, 3, 5, 10]
PALETTE = sns.color_palette("crest", n_colors=len(PIPELINE_ORDER))
PIPELINE_PALETTE: dict[str, tuple[float, float, float]] = dict(
    zip(PIPELINE_ORDER, PALETTE)
)

sns.set_theme(
    context="paper",
    style="whitegrid",
    font_scale=1.15,
    rc={
        "axes.titleweight": "semibold",
        "axes.labelweight": "medium",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    },
)


def _label_pipelines(df: pd.DataFrame, column: str = "pipeline") -> pd.DataFrame:
    """Replace pipeline keys with their human-readable labels and order them."""
    df = df.copy()
    df[column] = df[column].map(PIPELINE_LABELS).fillna(df[column])
    order = [PIPELINE_LABELS[p] for p in PIPELINE_ORDER if p in df[column].unique().tolist() or PIPELINE_LABELS[p] in df[column].unique().tolist()]
    df[column] = pd.Categorical(df[column], categories=order, ordered=True)
    return df


def _save(fig: plt.Figure, name: str, also_pdf: bool = True) -> None:
    """Persist a figure to PNG (and optionally PDF) under FIGURES_DIR."""
    png = FIGURES_DIR / f"{name}.png"
    fig.savefig(png)
    LOGGER.info("wrote %s", png)
    if also_pdf:
        pdf = FIGURES_DIR / f"{name}.pdf"
        fig.savefig(pdf)
        LOGGER.info("wrote %s", pdf)
    plt.close(fig)


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        LOGGER.warning("missing %s — skipping plots that depend on it", path)
        return None
    return pd.read_csv(path)


def plot_overall_comparison(results: pd.DataFrame) -> None:
    """Grouped bar chart at k=5 across the five headline metrics."""
    df = results.copy()
    df = df[df["K Value"] == 5]
    if df.empty:
        LOGGER.warning("no k=5 rows in results.csv; skipping overall comparison")
        return

    metrics = {
        "Recall@K": "Recall@5",
        "MRR": "MRR",
        "F1 Score": "F1",
        "Groundedness": "Groundedness",
        "Answerability Accuracy": "Answerability",
    }
    df = df.rename(columns={"pipeline_key": "pipeline"})
    df = _label_pipelines(df)
    long = df.melt(
        id_vars=["pipeline"],
        value_vars=list(metrics.keys()),
        var_name="metric_raw",
        value_name="value",
    )
    long["metric"] = long["metric_raw"].map(metrics)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.barplot(
        data=long,
        x="metric",
        y="value",
        hue="pipeline",
        palette=[PIPELINE_PALETTE[p] for p in PIPELINE_ORDER],
        ax=ax,
        edgecolor="white",
    )
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_title("Pipeline comparison at k=5", pad=12)
    ax.set_ylim(0, max(1.0, long["value"].max() * 1.15))
    ax.legend(title="Pipeline", bbox_to_anchor=(1.02, 1.0), loc="upper left", frameon=False)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=2, fontsize=8)
    _save(fig, "01_overall_comparison_k5")


def plot_retrieval_vs_k(retrieval: pd.DataFrame) -> None:
    """Recall@K, MRR, nDCG@K vs k — line plot with shaded 95% CI bands."""
    df = _label_pipelines(retrieval)
    metric_cols = {
        "recall_at_k": ("Recall@K", "recall_at_k_lo", "recall_at_k_hi"),
        "mrr": ("MRR", "mrr_lo", "mrr_hi"),
        "ndcg_at_k": ("nDCG@K", "ndcg_at_k_lo", "ndcg_at_k_hi"),
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True)
    for ax, (col, (label, lo, hi)) in zip(axes, metric_cols.items()):
        for pipeline_key in PIPELINE_ORDER:
            sub = df[df["pipeline"] == PIPELINE_LABELS[pipeline_key]].sort_values("k")
            if sub.empty:
                continue
            color = PIPELINE_PALETTE[pipeline_key]
            ax.plot(
                sub["k"], sub[col],
                marker="o", linewidth=2.0, color=color,
                label=PIPELINE_LABELS[pipeline_key],
            )
            if lo in sub.columns and hi in sub.columns:
                ax.fill_between(sub["k"], sub[lo], sub[hi], color=color, alpha=0.12)
        ax.set_xticks(K_ORDER)
        ax.set_xlabel("k")
        ax.set_ylabel(label)
        ax.set_title(label, pad=8)
    axes[0].legend(title="Pipeline", frameon=False, loc="lower right")
    fig.suptitle("Retrieval quality across depths (shaded = 95% CI)", y=1.02)
    _save(fig, "02_retrieval_vs_k")


def plot_generation_vs_k(generation: pd.DataFrame) -> None:
    """F1 / ROUGE-L / Semantic Similarity vs k with 95% CI bands."""
    df = _label_pipelines(generation)
    metrics = {
        "f1": ("F1", "f1_lo", "f1_hi"),
        "rouge_l": ("ROUGE-L", "rouge_l_lo", "rouge_l_hi"),
        "semantic_similarity": (
            "Semantic Similarity",
            "semantic_similarity_lo",
            "semantic_similarity_hi",
        ),
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True)
    for ax, (col, (label, lo, hi)) in zip(axes, metrics.items()):
        for pipeline_key in PIPELINE_ORDER:
            sub = df[df["pipeline"] == PIPELINE_LABELS[pipeline_key]].sort_values("k")
            if sub.empty:
                continue
            color = PIPELINE_PALETTE[pipeline_key]
            ax.plot(
                sub["k"], sub[col],
                marker="o", linewidth=2.0, color=color,
                label=PIPELINE_LABELS[pipeline_key],
            )
            if lo in sub.columns and hi in sub.columns:
                ax.fill_between(sub["k"], sub[lo], sub[hi], color=color, alpha=0.12)
        ax.set_xticks(K_ORDER)
        ax.set_xlabel("k")
        ax.set_ylabel(label)
        ax.set_title(label, pad=8)
    axes[0].legend(title="Pipeline", frameon=False, loc="upper left")
    fig.suptitle("Generation quality across depths (shaded = 95% CI)", y=1.02)
    _save(fig, "03_generation_vs_k")


def plot_grounded_faithfulness_vs_k(
    generation: pd.DataFrame,
    ragas: pd.DataFrame | None,
) -> None:
    """Groundedness (every cell) + Faithfulness (where RAGAS is available)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)

    g = _label_pipelines(generation)
    ax = axes[0]
    for pipeline_key in PIPELINE_ORDER:
        sub = g[g["pipeline"] == PIPELINE_LABELS[pipeline_key]].sort_values("k")
        if sub.empty:
            continue
        color = PIPELINE_PALETTE[pipeline_key]
        ax.plot(sub["k"], sub["groundedness"], marker="o", color=color, linewidth=2.0,
                label=PIPELINE_LABELS[pipeline_key])
        if "groundedness_lo" in sub.columns:
            ax.fill_between(
                sub["k"], sub["groundedness_lo"], sub["groundedness_hi"],
                color=color, alpha=0.12,
            )
    ax.set_xticks(K_ORDER)
    ax.set_xlabel("k")
    ax.set_ylabel("Groundedness")
    ax.set_title("Groundedness vs k (lexical)", pad=8)
    ax.legend(title="Pipeline", frameon=False, fontsize=9)

    ax = axes[1]
    if ragas is not None and not ragas.empty:
        r = _label_pipelines(ragas)
        sns.barplot(
            data=r,
            x="pipeline", y="faithfulness",
            order=[PIPELINE_LABELS[p] for p in PIPELINE_ORDER if PIPELINE_LABELS[p] in r["pipeline"].astype(str).unique()],
            palette=[PIPELINE_PALETTE[p] for p in PIPELINE_ORDER if PIPELINE_LABELS[p] in r["pipeline"].astype(str).unique()],
            ax=ax, edgecolor="white",
        )
        ax.set_ylabel("Faithfulness (RAGAS)")
        ax.set_xlabel("")
        ax.set_title("Faithfulness (LLM-as-judge, available cells)", pad=8)
        ax.set_ylim(0, 1.0)
        ax.tick_params(axis="x", rotation=15)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", padding=2, fontsize=9)
    else:
        ax.text(0.5, 0.5, "RAGAS metrics not yet available", ha="center", va="center")
        ax.set_axis_off()

    fig.suptitle("Faithfulness: cheap lexical vs RAGAS judge", y=1.02)
    _save(fig, "04_faithfulness_and_groundedness")


def plot_latency(latency: pd.DataFrame, results: pd.DataFrame) -> None:
    """Stacked bar of retrieval vs generation latency at each k."""
    if latency is None or latency.empty:
        LOGGER.warning("no latency_detail.csv — falling back to results.csv totals")
        df = results.rename(columns={"pipeline_key": "pipeline", "K Value": "k"}).copy()
        df = df[["pipeline", "k", "Retrieval Latency (s)", "Generation Latency (s)"]].rename(
            columns={
                "Retrieval Latency (s)": "retrieval_s",
                "Generation Latency (s)": "generation_s",
            }
        )
    else:
        df = latency.copy()
        df["retrieval_s"] = df["retrieval_ms_mean"] / 1000.0
        df["generation_s"] = df["generation_ms_mean"] / 1000.0

    df = _label_pipelines(df)
    df = df.sort_values(["pipeline", "k"])

    fig, ax = plt.subplots(figsize=(11, 5.5))
    pipelines = [PIPELINE_LABELS[p] for p in PIPELINE_ORDER if PIPELINE_LABELS[p] in df["pipeline"].astype(str).unique()]
    width = 0.18
    xs = np.arange(len(pipelines))
    for offset_idx, k in enumerate(K_ORDER):
        gen_vals = []
        ret_vals = []
        for pipeline in pipelines:
            sub = df[(df["pipeline"] == pipeline) & (df["k"] == k)]
            ret_vals.append(sub["retrieval_s"].iloc[0] if not sub.empty else 0.0)
            gen_vals.append(sub["generation_s"].iloc[0] if not sub.empty else 0.0)
        positions = xs + (offset_idx - 1.5) * width
        ax.bar(positions, ret_vals, width=width, color=PIPELINE_PALETTE[PIPELINE_ORDER[1]],
               alpha=0.55, edgecolor="white",
               label="Retrieval" if offset_idx == 0 else None)
        ax.bar(positions, gen_vals, width=width, bottom=ret_vals,
               color=PIPELINE_PALETTE[PIPELINE_ORDER[3]], edgecolor="white",
               label="Generation" if offset_idx == 0 else None)
        for x, total, k_val in zip(positions, np.array(ret_vals) + np.array(gen_vals), [k] * len(pipelines)):
            ax.text(x, total + 0.005, f"k={k_val}", ha="center", va="bottom",
                    fontsize=7, color="#555")

    ax.set_xticks(xs)
    ax.set_xticklabels(pipelines)
    ax.set_ylabel("Latency per question (s)")
    ax.set_title("Latency breakdown: retrieval (light) vs generation (dark) at k ∈ {1, 3, 5, 10}", pad=12)
    ax.legend(loc="upper left", frameon=False)
    _save(fig, "05_latency_breakdown")


def plot_category_heatmap(category: pd.DataFrame, metric: str = "f1") -> None:
    """Heatmap of metric across (pipeline × category) at k=5."""
    df = category[category["k"] == 5].copy()
    if df.empty:
        LOGGER.warning("no k=5 rows in category_metrics.csv; skipping heatmap")
        return
    df = _label_pipelines(df)
    pivot = df.pivot_table(
        index="pipeline", columns="category", values=metric, aggfunc="mean",
        observed=True,
    )
    pivot = pivot.reindex([PIPELINE_LABELS[p] for p in PIPELINE_ORDER if PIPELINE_LABELS[p] in pivot.index])

    fig, ax = plt.subplots(figsize=(9, 4.2))
    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap="crest",
        cbar_kws={"label": metric.upper()}, linewidths=0.5, linecolor="white",
        ax=ax,
    )
    ax.set_title(f"{metric.upper()} by category at k=5", pad=10)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=20)
    _save(fig, f"06_category_heatmap_{metric}_k5")


def plot_qbucket_heatmap(qbucket: pd.DataFrame) -> None:
    """Heatmap of F1 by (pipeline × question-length bucket) at k=5."""
    df = qbucket[qbucket["k"] == 5].copy()
    if df.empty:
        LOGGER.warning("no k=5 rows in qbucket_metrics.csv; skipping heatmap")
        return
    df = _label_pipelines(df)
    order = ["short", "medium", "long"]
    pivot = df.pivot_table(index="pipeline", columns="q_bucket", values="f1",
                           aggfunc="mean", observed=True)
    pivot = pivot[[c for c in order if c in pivot.columns]]
    pivot = pivot.reindex([PIPELINE_LABELS[p] for p in PIPELINE_ORDER if PIPELINE_LABELS[p] in pivot.index])

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap="rocket_r",
        cbar_kws={"label": "F1"}, linewidths=0.5, linecolor="white",
        ax=ax,
    )
    ax.set_title("F1 by question-length bucket at k=5", pad=10)
    ax.set_xlabel("")
    ax.set_ylabel("")
    _save(fig, "07_qbucket_heatmap_f1_k5")


def plot_answerability(results: pd.DataFrame) -> None:
    """Answerability accuracy across k as a line plot with marker labels."""
    df = results.rename(columns={"pipeline_key": "pipeline", "K Value": "k"}).copy()
    df = _label_pipelines(df)

    fig, ax = plt.subplots(figsize=(9, 5))
    for pipeline_key in PIPELINE_ORDER:
        sub = df[df["pipeline"] == PIPELINE_LABELS[pipeline_key]].sort_values("k")
        if sub.empty:
            continue
        color = PIPELINE_PALETTE[pipeline_key]
        ax.plot(sub["k"], sub["Answerability Accuracy"], marker="o",
                color=color, linewidth=2.0, label=PIPELINE_LABELS[pipeline_key])
        for k, v in zip(sub["k"], sub["Answerability Accuracy"]):
            ax.annotate(f"{v:.2f}", (k, v), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8, color=color)

    ax.set_xticks(K_ORDER)
    ax.set_xlabel("k")
    ax.set_ylabel("Answerability accuracy")
    ax.set_ylim(0.5, 0.9)
    ax.set_title("Answerability accuracy across retrieval depth", pad=12)
    ax.legend(title="Pipeline", frameon=False, bbox_to_anchor=(1.02, 1.0), loc="upper left")
    _save(fig, "08_answerability_vs_k")


def plot_radar_at_k5(results: pd.DataFrame) -> None:
    """Radar / spider plot of the five pipelines on six normalised metrics at k=5."""
    df = results[results["K Value"] == 5].rename(columns={"pipeline_key": "pipeline"})
    if df.empty:
        LOGGER.warning("no k=5 rows for radar plot; skipping")
        return

    metrics = ["Recall@K", "MRR", "F1 Score", "Groundedness",
               "Answerability Accuracy", "Semantic Similarity"]
    labels = ["Recall@5", "MRR", "F1", "Grounded", "Answerable", "Sem.Sim"]

    norm = df.set_index("pipeline")[metrics].copy()
    for col in norm.columns:
        lo, hi = norm[col].min(), norm[col].max()
        if hi - lo < 1e-9:
            norm[col] = 0.5
        else:
            norm[col] = (norm[col] - lo) / (hi - lo)

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    for pipeline_key in PIPELINE_ORDER:
        if pipeline_key not in norm.index:
            continue
        values = norm.loc[pipeline_key].tolist() + [norm.loc[pipeline_key].tolist()[0]]
        color = PIPELINE_PALETTE[pipeline_key]
        ax.plot(angles, values, color=color, linewidth=2.0, label=PIPELINE_LABELS[pipeline_key])
        ax.fill(angles, values, color=color, alpha=0.10)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_rlabel_position(180 / len(metrics))
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title("Pipeline strengths at k=5 (each axis min–max normalised across pipelines)",
                 pad=22)
    ax.legend(loc="upper right", bbox_to_anchor=(1.30, 1.10), frameon=False)
    _save(fig, "09_radar_k5")


def plot_metric_correlation(results: pd.DataFrame) -> None:
    """Correlation heatmap of all numeric headline metrics across (pipeline, k) cells."""
    cols = [
        "Recall@K", "MRR", "nDCG@K", "F1 Score", "ROUGE-L", "Semantic Similarity",
        "Groundedness", "Answerability Accuracy", "Long Context Accuracy",
        "Noise Robustness", "Avg Latency / Question (s)",
    ]
    df = results[[c for c in cols if c in results.columns]].copy()
    if df.empty:
        LOGGER.warning("no metric columns for correlation; skipping")
        return
    corr = df.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="vlag", center=0,
        vmin=-1, vmax=1, square=True, linewidths=0.4, linecolor="white",
        cbar_kws={"label": "Pearson r"}, ax=ax,
    )
    ax.set_title("Metric correlations across (pipeline, k) cells", pad=12)
    _save(fig, "10_metric_correlation")


def plot_ranking_dotplot(results: pd.DataFrame) -> None:
    """Dot-plot of F1 with horizontal bars showing pipeline ranking per k."""
    df = results.rename(columns={"pipeline_key": "pipeline", "K Value": "k"}).copy()
    df = _label_pipelines(df)
    df = df.sort_values(["k", "F1 Score"], ascending=[True, False])

    fig, axes = plt.subplots(1, len(K_ORDER), figsize=(15, 4.8), sharey=True)
    for ax, k in zip(axes, K_ORDER):
        sub = df[df["k"] == k].sort_values("F1 Score", ascending=True)
        if sub.empty:
            continue
        colors = [PIPELINE_PALETTE[p] for p in PIPELINE_ORDER if PIPELINE_LABELS[p] in sub["pipeline"].astype(str).tolist()]
        ax.hlines(y=sub["pipeline"], xmin=0, xmax=sub["F1 Score"],
                  color=colors, alpha=0.55, linewidth=3)
        ax.scatter(sub["F1 Score"], sub["pipeline"], color=colors, s=110, zorder=3,
                   edgecolor="white", linewidth=1.2)
        for f1, name in zip(sub["F1 Score"], sub["pipeline"]):
            ax.text(f1 + 0.0015, name, f"{f1:.3f}", va="center", fontsize=9)
        ax.set_title(f"k = {k}")
        ax.set_xlabel("F1")
        ax.set_xlim(0, sub["F1 Score"].max() * 1.25)
    axes[0].set_ylabel("")
    fig.suptitle("F1 by pipeline at each retrieval depth", y=1.02)
    _save(fig, "11_f1_ranking_by_k")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        nargs="*",
        choices=[
            "overall", "retrieval", "generation", "faithfulness",
            "latency", "category", "qbucket", "answerability",
            "radar", "correlation", "ranking",
        ],
        help="Only plot the named figures (default: all).",
    )
    parser.add_argument("--no-pdf", action="store_true", help="Skip the PDF copies.")
    args = parser.parse_args()

    if args.no_pdf:
        global _save
        _orig = _save

        def _save_png_only(fig: plt.Figure, name: str, also_pdf: bool = True) -> None:
            _orig(fig, name, also_pdf=False)

        _save = _save_png_only

    results = _read_csv(OUTPUT_DIR / "results.csv")
    retrieval = _read_csv(OUTPUT_DIR / "retrieval_metrics.csv")
    generation = _read_csv(OUTPUT_DIR / "generation_metrics.csv")
    ragas = _read_csv(OUTPUT_DIR / "ragas_metrics.csv")
    latency = _read_csv(OUTPUT_DIR / "latency_detail.csv")
    category = _read_csv(OUTPUT_DIR / "category_metrics.csv")
    qbucket = _read_csv(OUTPUT_DIR / "qbucket_metrics.csv")

    selected: set[str] = set(args.only) if args.only else {
        "overall", "retrieval", "generation", "faithfulness", "latency",
        "category", "qbucket", "answerability", "radar", "correlation", "ranking",
    }

    if "overall" in selected and results is not None:
        plot_overall_comparison(results)
    if "retrieval" in selected and retrieval is not None:
        plot_retrieval_vs_k(retrieval)
    if "generation" in selected and generation is not None:
        plot_generation_vs_k(generation)
    if "faithfulness" in selected and generation is not None:
        plot_grounded_faithfulness_vs_k(generation, ragas)
    if "latency" in selected and results is not None:
        plot_latency(latency, results)
    if "category" in selected and category is not None:
        plot_category_heatmap(category, metric="f1")
        plot_category_heatmap(category, metric="answerability_acc")
    if "qbucket" in selected and qbucket is not None:
        plot_qbucket_heatmap(qbucket)
    if "answerability" in selected and results is not None:
        plot_answerability(results)
    if "radar" in selected and results is not None:
        plot_radar_at_k5(results)
    if "correlation" in selected and results is not None:
        plot_metric_correlation(results)
    if "ranking" in selected and results is not None:
        plot_ranking_dotplot(results)

    LOGGER.info("figures written to %s", FIGURES_DIR)


if __name__ == "__main__":
    main()
