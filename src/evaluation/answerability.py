"""Answerability accuracy table (Sheet Table 6).

           | refused=False  | refused=True
gold yes   | Correct        | Wrongly Refused
gold no    | Wrongly Answered| Correct
"""
from __future__ import annotations

import pandas as pd


def classify_answerability(is_answerable: int, refused: bool) -> str:
    if is_answerable == 1 and not refused:
        return "correctly_answered"
    if is_answerable == 1 and refused:
        return "wrongly_refused"
    if is_answerable == 0 and refused:
        return "correctly_refused"
    return "wrongly_answered"


def compute_answerability_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """Return a one-row table of answerability accuracy + cell counts."""
    df = results_df.copy()
    df["bucket"] = [
        classify_answerability(int(row["is_answerable"]), bool(row["refused"]))
        for _, row in df.iterrows()
    ]
    counts = df["bucket"].value_counts().to_dict()
    correct = counts.get("correctly_answered", 0) + counts.get("correctly_refused", 0)
    total = len(df)
    acc = correct / total if total else 0.0
    return pd.DataFrame([{
        "n": total,
        "correctly_answered": counts.get("correctly_answered", 0),
        "wrongly_refused": counts.get("wrongly_refused", 0),
        "correctly_refused": counts.get("correctly_refused", 0),
        "wrongly_answered": counts.get("wrongly_answered", 0),
        "answerability_acc": acc,
    }])
