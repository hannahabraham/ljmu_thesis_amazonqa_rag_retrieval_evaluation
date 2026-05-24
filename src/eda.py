"""Per-split EDA summaries and plots."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from config.settings import (
    EDA_PLOTS_DIR,
    WILSON_VOTE_THRESHOLD,
)

LOGGER = logging.getLogger(__name__)


def _vote_counts(answer: dict[str, Any]) -> tuple[int, int]:
    """Return (helpful_votes, total_votes) from an answer dict."""
    helpful_field = answer.get("helpful", 0)

    if hasattr(helpful_field, "__len__") and not isinstance(
        helpful_field,
        (str, bytes),
    ):
        sequence = list(helpful_field)

        if len(sequence) >= 2:
            return int(sequence[0]), int(sequence[1])

        if len(sequence) == 1:
            return int(sequence[0]), int(sequence[0])

        return 0, 0

    helpful = int(helpful_field) if helpful_field is not None else 0
    unhelpful = int(answer.get("unhelpful", 0) or 0)

    return helpful, helpful + unhelpful


def _total_votes(answers: list[dict[str, Any]]) -> int:
    """Compute total votes across all answers."""
    return sum(
        _vote_counts(answer)[1]
        for answer in answers
    )


def _close_call_score(
    answers: list[dict[str, Any]],
) -> float | None:
    """Return Jeffreys-score gap between top two answers."""
    if answers is None or len(answers) < 2:
        return None

    scores: list[float] = []

    for answer in answers:
        helpful, total = _vote_counts(answer)

        score = (helpful + 0.5) / (total + 1.0)
        scores.append(score)

    scores.sort(reverse=True)

    return abs(scores[0] - scores[1])


def summarise_split(
    dataframe: pd.DataFrame,
    split_name: str,
) -> dict[str, Any]:
    """Compute summary statistics for one dataset split."""
    total_rows = len(dataframe)

    duplicate_qids = (
        int(dataframe.duplicated(subset=["qid"]).sum())
        if "qid" in dataframe.columns
        else 0
    )

    answerable_count = (
        int((dataframe["is_answerable"] == 1).sum())
        if "is_answerable" in dataframe.columns
        else 0
    )

    unanswerable_count = (
        int((dataframe["is_answerable"] == 0).sum())
        if "is_answerable" in dataframe.columns
        else 0
    )

    question_type_counts = (
        dataframe["questionType"]
        .value_counts(dropna=False)
        .to_dict()
        if "questionType" in dataframe.columns
        else {}
    )

    total_votes = pd.Series(dtype=float)

    if "answers" in dataframe.columns:
        total_votes = dataframe["answers"].apply(_total_votes)

    zero_votes = (
        int((total_votes == 0).sum())
        if not total_votes.empty
        else 0
    )

    low_votes = (
        int(
            (
                (total_votes > 0)
                & (total_votes < WILSON_VOTE_THRESHOLD)
            ).sum()
        )
        if not total_votes.empty
        else 0
    )

    wilson_votes = (
        int((total_votes >= WILSON_VOTE_THRESHOLD).sum())
        if not total_votes.empty
        else 0
    )

    return {
        "split": split_name,
        "n_rows": total_rows,
        "n_duplicates_qid": duplicate_qids,
        "n_answerable": answerable_count,
        "n_unanswerable": unanswerable_count,
        "qtype_counts": question_type_counts,
        "votes_zero": zero_votes,
        "votes_low": low_votes,
        "votes_wilson_applicable": wilson_votes,
        "n_categories": (
            int(dataframe["category"].nunique())
            if "category" in dataframe.columns
            else 0
        ),
        "median_n_snippets": (
            float(dataframe["n_snippets"].median())
            if "n_snippets" in dataframe.columns
            else 0.0
        ),
        "median_n_answers": (
            float(dataframe["n_answers"].median())
            if "n_answers" in dataframe.columns
            else 0.0
        ),
    }


def make_plots(
    dataframe: pd.DataFrame,
    split_name: str,
    out_dir: Path = EDA_PLOTS_DIR,
) -> None:
    """Generate and save EDA plots for one dataset split."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    if dataframe.empty:
        LOGGER.warning(
            "Skipping plots for %s: dataframe is empty",
            split_name,
        )
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    sns.set_style("whitegrid")
    sns.set_palette("colorblind")

    if "answers" in dataframe.columns:
        votes = dataframe["answers"].apply(_total_votes)

        fig, axis = plt.subplots(figsize=(8, 4))

        sns.histplot(
            votes.clip(upper=50),
            bins=30,
            ax=axis,
        )

        axis.set_title(
            f"{split_name}: vote distribution (clipped at 50)"
        )

        fig.tight_layout()

        fig.savefig(
            out_dir / f"{split_name}_vote_distribution.png",
            dpi=120,
        )

        plt.close(fig)

    if {
        "is_answerable",
        "questionType",
    }.issubset(dataframe.columns):
        cross_tab = (
            dataframe.groupby(
                ["questionType", "is_answerable"],
                dropna=False,
            )
            .size()
            .unstack(fill_value=0)
        )

        if not cross_tab.empty and cross_tab.to_numpy().sum() > 0:
            fig, axis = plt.subplots(figsize=(8, 4))

            cross_tab.plot(
                kind="bar",
                stacked=True,
                ax=axis,
            )

            axis.set_title(f"{split_name}: dataset profile")
            axis.set_ylabel("count")

            fig.tight_layout()

            fig.savefig(
                out_dir / f"{split_name}_dataset_profile.png",
                dpi=120,
            )

            plt.close(fig)

    if "category" in dataframe.columns:
        top_categories = (
            dataframe["category"]
            .value_counts()
            .head(10)
        )

        fig, axis = plt.subplots(figsize=(8, 4))

        sns.barplot(
            x=top_categories.values,
            y=top_categories.index,
            ax=axis,
        )

        axis.set_title(f"{split_name}: top-10 categories")

        fig.tight_layout()

        fig.savefig(
            out_dir / f"{split_name}_top_categories.png",
            dpi=120,
        )

        plt.close(fig)

    if "n_snippets" in dataframe.columns:
        fig, axis = plt.subplots(figsize=(8, 4))

        sns.histplot(
            dataframe["n_snippets"].clip(upper=20),
            bins=20,
            ax=axis,
        )

        axis.set_title(
            f"{split_name}: review snippets per question"
        )

        fig.tight_layout()

        fig.savefig(
            out_dir / f"{split_name}_snippet_quality.png",
            dpi=120,
        )

        plt.close(fig)

    if "n_answers" in dataframe.columns:
        fig, axis = plt.subplots(figsize=(8, 4))

        sns.histplot(
            dataframe["n_answers"].clip(upper=20),
            bins=20,
            ax=axis,
        )

        axis.set_title(
            f"{split_name}: answers per question"
        )

        fig.tight_layout()

        fig.savefig(
            out_dir / f"{split_name}_answer_grounding.png",
            dpi=120,
        )

        plt.close(fig)

    if "answers" in dataframe.columns:
        close_calls = dataframe["answers"].apply(
            _close_call_score
        )

        fig, axis = plt.subplots(figsize=(8, 4))

        sns.histplot(
            close_calls.dropna(),
            bins=20,
            ax=axis,
        )

        axis.set_title(
            f"{split_name}: close-call score "
            "(lower = closer)"
        )

        fig.tight_layout()

        fig.savefig(
            out_dir / f"{split_name}_close_call.png",
            dpi=120,
        )

        plt.close(fig)
