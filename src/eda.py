"""Per-split EDA summaries and plots."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from config.settings import EDA_PLOTS_DIR, WILSON_VOTE_THRESHOLD

logger = logging.getLogger(__name__)


def _vote_counts(answer: dict) -> tuple[int, int]:
    """Return (helpful_votes, total_votes) from an AmazonQA answer dict.

    The raw `helpful` field is a 2-element array [helpful, total]. Older / partial
    records may store separate `helpful` / `unhelpful` scalars instead.
    """
    helpful_field = answer.get("helpful", 0)
    if hasattr(helpful_field, "__len__") and not isinstance(helpful_field, (str, bytes)):
        seq = list(helpful_field)
        if len(seq) >= 2:
            return int(seq[0]), int(seq[1])
        if len(seq) == 1:
            return int(seq[0]), int(seq[0])
        return 0, 0
    helpful = int(helpful_field) if helpful_field is not None else 0
    unhelpful = int(answer.get("unhelpful", 0) or 0)
    return helpful, helpful + unhelpful


def summarise_split(df: pd.DataFrame, split_name: str) -> dict:
    """Compute a one-row-per-split summary."""
    n = len(df)
    duplicates = int(df.duplicated(subset=["qid"]).sum()) if "qid" in df.columns else 0
    answerable = int((df["is_answerable"] == 1).sum()) if "is_answerable" in df.columns else 0
    unanswerable = int((df["is_answerable"] == 0).sum()) if "is_answerable" in df.columns else 0
    qtype_counts = (
        df["questionType"].value_counts(dropna=False).to_dict()
        if "questionType" in df.columns else {}
    )

    total_votes = pd.Series(dtype=float)
    if "answers" in df.columns:
        def _total(answers: list[dict]) -> int:
            return sum(_vote_counts(a)[1] for a in answers)
        total_votes = df["answers"].apply(_total)

    zero = int((total_votes == 0).sum()) if not total_votes.empty else 0
    low = int(((total_votes > 0) & (total_votes < WILSON_VOTE_THRESHOLD)).sum()) if not total_votes.empty else 0
    wilson = int((total_votes >= WILSON_VOTE_THRESHOLD).sum()) if not total_votes.empty else 0

    return {
        "split": split_name,
        "n_rows": n,
        "n_duplicates_qid": duplicates,
        "n_answerable": answerable,
        "n_unanswerable": unanswerable,
        "qtype_counts": qtype_counts,
        "votes_zero": zero,
        "votes_low": low,
        "votes_wilson_applicable": wilson,
        "n_categories": int(df["category"].nunique()) if "category" in df.columns else 0,
        "median_n_snippets": float(df["n_snippets"].median()) if "n_snippets" in df.columns else 0.0,
        "median_n_answers": float(df["n_answers"].median()) if "n_answers" in df.columns else 0.0,
    }


def make_plots(df: pd.DataFrame, split_name: str, out_dir: Path = EDA_PLOTS_DIR) -> None:
    """Write 6 EDA plots for a split."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    if df.empty:
        logger.warning("Skipping plots for %s: dataframe is empty", split_name)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")

    if "answers" in df.columns:
        votes = df["answers"].apply(
            lambda answers: sum(_vote_counts(a)[1] for a in answers)
        )
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(votes.clip(upper=50), bins=30, ax=ax)
        ax.set_title(f"{split_name}: vote distribution (clipped at 50)")
        fig.tight_layout()
        fig.savefig(out_dir / f"{split_name}_vote_distribution.png", dpi=120)
        plt.close(fig)

    if "is_answerable" in df.columns and "questionType" in df.columns:
        ct = (
            df.groupby(["questionType", "is_answerable"], dropna=False)
            .size()
            .unstack(fill_value=0)
        )
        if not ct.empty and ct.to_numpy().sum() > 0:
            fig, ax = plt.subplots(figsize=(8, 4))
            ct.plot(kind="bar", stacked=True, ax=ax)
            ax.set_title(f"{split_name}: dataset profile")
            ax.set_ylabel("count")
            fig.tight_layout()
            fig.savefig(out_dir / f"{split_name}_dataset_profile.png", dpi=120)
            plt.close(fig)

    if "category" in df.columns:
        top10 = df["category"].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x=top10.values, y=top10.index, ax=ax)
        ax.set_title(f"{split_name}: top-10 categories")
        fig.tight_layout()
        fig.savefig(out_dir / f"{split_name}_top_categories.png", dpi=120)
        plt.close(fig)

    if "n_snippets" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df["n_snippets"].clip(upper=20), bins=20, ax=ax)
        ax.set_title(f"{split_name}: review snippets per question")
        fig.tight_layout()
        fig.savefig(out_dir / f"{split_name}_snippet_quality.png", dpi=120)
        plt.close(fig)

    if "n_answers" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df["n_answers"].clip(upper=20), bins=20, ax=ax)
        ax.set_title(f"{split_name}: answers per question")
        fig.tight_layout()
        fig.savefig(out_dir / f"{split_name}_answer_grounding.png", dpi=120)
        plt.close(fig)

    if "answers" in df.columns:
        close_calls = df["answers"].apply(_close_call_score)
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(close_calls.dropna(), bins=20, ax=ax)
        ax.set_title(f"{split_name}: close-call score (lower = closer)")
        fig.tight_layout()
        fig.savefig(out_dir / f"{split_name}_close_call.png", dpi=120)
        plt.close(fig)


def _close_call_score(answers: list[dict]) -> float | None:
    """Return abs Jeffreys-difference between top two answers; None if <2 candidates."""
    if answers is None or len(answers) < 2:
        return None
    scored: list[float] = []
    for a in answers:
        helpful, total = _vote_counts(a)
        scored.append((helpful + 0.5) / (total + 1.0))
    scored.sort(reverse=True)
    return abs(scored[0] - scored[1])
