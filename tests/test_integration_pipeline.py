"""End-to-end pipeline test on five fake records.

Exercises load, knowledge-base construction, chunking, BM25 retrieval,
stubbed generation, and answerability evaluation. No external services.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.chunking import build_passage_chunks
from src.evaluation.answerability import compute_answerability_table
from src.generation.rag_generator import generate_rag_answer
from src.knowledge_base_builder import build_knowledge_base
from src.retrievers.bm25 import BM25Retriever


@pytest.mark.integration
def test_pipeline_end_to_end_on_fake_records() -> None:
    """Test the full offline RAG pipeline on fake answerable records."""
    fake_records = pd.DataFrame(
        [
            {
                "record_id": f"REC_{index:03d}",
                "qid": f"Q_{index}",
                "asin": f"B0FAKE{index}",
                "category": "Electronics",
                "source_file": "test",
                "questionType": "yesno",
                "is_answerable": 1,
                "questionText": f"Is product {index} waterproof?",
                "review_snippets": [
                    f"Product {index} is fully waterproof to 10 metres deep.",
                    f"I dropped product {index} in water it survived completely.",
                    f"Battery on product {index} lasts about eight hours.",
                ],
                "top_sentences_IR": [],
                "top_review_helpful": [],
                "top_review_wilson": [],
            }
            for index in range(5)
        ]
    )

    knowledge_base = build_knowledge_base(fake_records)
    assert len(knowledge_base) >= 10

    chunks = build_passage_chunks(knowledge_base)
    assert not chunks.empty

    bm25_retriever = BM25Retriever(chunks, text_col="text")

    fake_groq = MagicMock()
    fake_groq.invoke.return_value = "Yes, this product is waterproof."

    results: list[dict] = []

    for _, record in fake_records.iterrows():
        retrieved_docs = list(
            bm25_retriever.retrieve(
                record["questionText"],
                record["asin"],
                k=3,
            )
        )
        generation = generate_rag_answer(
            question=record["questionText"],
            retrieved_docs=retrieved_docs,
            groq_manager=fake_groq,
            retrieval_ms=1.0,
        )

        results.append(
            {
                "qid": record["qid"],
                "is_answerable": record["is_answerable"],
                "refused": generation["refused"],
                "retrieved_doc_ids": [
                    document["doc_id"] for document in retrieved_docs
                ],
                "generated_answer": generation["generated_answer"],
                "generation_ms": generation["generation_ms"],
            }
        )

    results_df = pd.DataFrame(results)

    assert len(results_df) == 5
    assert results_df["generated_answer"].notna().all()
    assert (~results_df["refused"]).all()
    assert (results_df["retrieved_doc_ids"].str.len() > 0).all()
    assert results_df["generation_ms"].gt(0).all()

    table = compute_answerability_table(results_df)

    assert "answerability_acc" in table.columns
    assert 0.0 <= table["answerability_acc"].iloc[0] <= 1.0