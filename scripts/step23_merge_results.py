import pandas as pd
from pathlib import Path

OUT = Path("outputs")
res = pd.read_csv(OUT / "results.csv")
ragas = pd.read_csv(OUT / "ragas_metrics.csv")          # faithfulness, context_precision, context_recall
hall  = pd.read_csv(OUT / "hallucination_metrics.csv")  # hallucination_rate, refusal_rate_on_answerable

# normalise the join keys
res["pipeline_key"] = res["pipeline_key"].str.lower()
key = ["pipeline_key", "K Value"]
ragas = ragas.rename(columns={"pipeline": "pipeline_key", "k": "K Value"})
hall  = hall.rename(columns={"pipeline": "pipeline_key", "k": "K Value"})

# drop the stale all-rows hallucination + lexical columns that conflict with the thesis
res = res.drop(columns=[c for c in ["Hallucination Rate", "Groundedness"] if c in res.columns])

merged = (res
          .merge(ragas[key + ["faithfulness", "context_precision", "context_recall"]], on=key, how="left")
          .merge(hall[key + ["hallucination_rate", "refusal_rate_on_answerable"]], on=key, how="left"))

# fill the previously-empty RAGAS columns from the join
merged["Faithfulness Score"]  = merged["faithfulness"]
merged["Context Precision"]   = merged["context_precision"]
merged["Context Recall"]      = merged["context_recall"]
merged["Hallucination Rate"]  = merged["hallucination_rate"]   # faithfulness-based, matches thesis

merged = merged.drop(columns=["faithfulness", "context_precision", "context_recall", "hallucination_rate"])
merged.to_csv(OUT / "results.csv", index=False)