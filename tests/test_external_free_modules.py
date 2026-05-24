"""Coverage for external-facing modules using local fakes only."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import data_loader, indexing
from src.evaluation import ragas_metrics
from src.llm_clients import loader, ragas_judge
from src.llm_clients.gemini_key_manager import GeminiKeyManager
from src.llm_clients.parallel_groq import GroqClient
from src.llm_clients.round_robin_gemini import RotatingGeminiChat
from src.retrievers.dense import DenseRetriever
from src.retrievers.hybrid import HybridRetriever, reciprocal_rank_fusion
from src.retrievers.parent_child import ParentChildRetriever
from src.retrievers.sentence_window import SentenceWindowRetriever


class _FakeEmbedder:
    """Small embedder that returns deterministic vectors."""

    def encode(self, texts: list[str], **_: Any) -> np.ndarray:
        """Return one tiny vector per text."""
        return np.array([[float(index + 1), 0.0] for index, _ in enumerate(texts)])


class _Hit:
    """Fake Qdrant hit."""

    def __init__(self, payload: dict[str, Any], score: float = 0.9) -> None:
        self.payload = payload
        self.score = score


class _Response:
    """Fake Qdrant query response."""

    def __init__(self, points: list[_Hit]) -> None:
        self.points = points


class _FakeClient:
    """Fake vector client with configurable hits."""

    def __init__(self, points: list[_Hit] | None = None) -> None:
        self.points = points or []
        self.created: list[str] = []
        self.upserts: list[tuple[str, list[Any]]] = []
        self.deleted: list[str] = []

    def query_points(self, **_: Any) -> _Response:
        """Return configured points."""
        return _Response(self.points)

    def collection_exists(self, collection_name: str) -> bool:
        """Pretend collections ending with old already exist."""
        return collection_name.endswith("old")

    def delete_collection(self, collection_name: str) -> None:
        """Record deleted collection."""
        self.deleted.append(collection_name)

    def create_collection(self, collection_name: str, **_: Any) -> None:
        """Record collection creation."""
        self.created.append(collection_name)

    def create_payload_index(self, **_: Any) -> None:
        """Accept payload-index creation."""

    def upsert(self, collection_name: str, points: list[Any]) -> None:
        """Record upserted points."""
        self.upserts.append((collection_name, points))


def test_load_jsonl_skips_malformed_lines(tmp_path: Path) -> None:
    """Test JSONL loader keeps valid rows and skips malformed ones."""
    path = tmp_path / "data.jsonl"
    path.write_text('{"id": 1}\nnot-json\n\n{"id": 2}\n', encoding="utf-8")

    df = data_loader.load_jsonl(path)

    assert df["id"].tolist() == [1, 2]


def test_loader_reads_primary_and_numbered_keys(monkeypatch) -> None:
    """Test environment key loading without prompting."""
    monkeypatch.setenv("SERVICE_KEY", "primary")
    assert loader.load_single_key("Svc", "SERVICE_KEY", "SERVICE_KEY_") == "primary"

    monkeypatch.delenv("SERVICE_KEY")
    monkeypatch.setenv("SERVICE_KEY_2", "two")
    monkeypatch.setenv("SERVICE_KEY_1", "one")
    assert loader.load_single_key("Svc", "SERVICE_KEY", "SERVICE_KEY_") == "one"
    assert loader.load_all_keys("Svc", "SERVICE_KEY", "SERVICE_KEY_") == ["one", "two"]


def test_indexing_tokenize_embed_and_upsert(monkeypatch) -> None:
    """Test indexing helpers with fake client and embedder."""
    monkeypatch.setattr(indexing, "get_embedder", lambda: _FakeEmbedder())
    client = _FakeClient()
    chunks = pd.DataFrame(
        [
            {
                "chunk_id": "c1",
                "doc_id": "d1",
                "record_id": "r1",
                "asin": "a1",
                "category": "cat",
                "text": "Hello world",
                "extra": "x",
            }
        ]
    )

    assert indexing.tokenize("Hello, WORLD! 123") == ["hello", "world", "123"]
    vectors = indexing._embed(_FakeEmbedder(), ["a", "b"])
    assert vectors.shape == (2, 2)

    indexing.upsert_collection(
        client,
        "test",
        chunks,
        extra_payload_cols=("extra",),
    )

    assert client.created == ["test"]
    assert len(client.upserts[0][1]) == 1
    assert client.upserts[0][1][0].payload["extra"] == "x"


def test_dense_parent_child_and_sentence_window_retrievers() -> None:
    """Test vector retrievers using fake clients and embedders."""
    dense_client = _FakeClient(
        [
            _Hit(
                {
                    "chunk_id": "c1",
                    "doc_id": "d1",
                    "record_id": "r1",
                    "asin": "a1",
                    "category": "cat",
                    "text": "dense text",
                },
                0.8,
            )
        ]
    )
    dense = DenseRetriever(client=dense_client, embedder=_FakeEmbedder())
    dense_results = dense.retrieve("question", "a1", 1)
    assert dense_results[0]["retriever"] == "dense"
    assert dense.retrieve("question", "a1", 0) == []

    pc_client = _FakeClient(
        [
            _Hit({"chunk_id": "c1", "parent_id": "p1", "parent_text": "parent"}),
            _Hit({"chunk_id": "c2", "parent_id": "p1", "parent_text": "dup"}),
            _Hit({"chunk_id": "c3", "parent_id": "p2", "parent_text": "parent 2"}),
        ]
    )
    pc = ParentChildRetriever(client=pc_client, embedder=_FakeEmbedder())
    pc_results = pc.retrieve("question", "a1", 2)
    assert [result["parent_id"] for result in pc_results] == ["p1", "p2"]

    sentence_chunks = pd.DataFrame(
        [
            {
                "chunk_id": "s0",
                "doc_id": "d1",
                "record_id": "r1",
                "asin": "a1",
                "category": "cat",
                "text": "previous",
                "sentence_index": 0,
                "prev_sent_id": None,
                "next_sent_id": "s1",
            },
            {
                "chunk_id": "s1",
                "doc_id": "d1",
                "record_id": "r1",
                "asin": "a1",
                "category": "cat",
                "text": "matched",
                "sentence_index": 1,
                "prev_sent_id": "s0",
                "next_sent_id": "s2",
            },
            {
                "chunk_id": "s2",
                "doc_id": "d1",
                "record_id": "r1",
                "asin": "a1",
                "category": "cat",
                "text": "next",
                "sentence_index": 2,
                "prev_sent_id": "s1",
                "next_sent_id": None,
            },
        ]
    )
    sent_client = _FakeClient([_Hit({"chunk_id": "s1", "asin": "a1"}, 0.7)])
    sent = SentenceWindowRetriever(
        sentence_chunks,
        client=sent_client,
        embedder=_FakeEmbedder(),
    )
    sent_results = sent.retrieve("question", "a1", 1)
    assert sent.get_window_ids("s1") == ["s0", "s1", "s2"]
    assert sent_results[0]["text"] == "previous matched next"


class _StaticRetriever:
    """Fake retriever returning fixed hits."""

    def __init__(self, hits: list[dict[str, Any]]) -> None:
        self.hits = hits

    def retrieve(self, **_: Any) -> list[dict[str, Any]]:
        """Return fixed hits."""
        return self.hits


def test_hybrid_retriever_fuses_contributors() -> None:
    """Test reciprocal rank fusion and hybrid metadata."""
    assert reciprocal_rank_fusion([["a", "b"], ["b"]], k_rrf=0)[0][0] == "b"

    bm25_hits = [
        {"chunk_id": "a", "text": "a", "score": 2.0},
        {"chunk_id": "b", "text": "b", "score": 1.0},
    ]
    dense_hits = [
        {"chunk_id": "b", "text": "b", "score": 0.8},
        {"chunk_id": "c", "text": "c", "score": 0.7},
    ]
    hybrid = HybridRetriever(
        bm25=_StaticRetriever(bm25_hits),  # type: ignore[arg-type]
        dense=_StaticRetriever(dense_hits),  # type: ignore[arg-type]
        k_rrf=0,
    )

    results = hybrid.retrieve("question", "asin", 2)

    assert results[0]["chunk_id"] == "b"
    assert results[0]["contributors"] == ["bm25", "dense"]
    assert hybrid.retrieve("question", "asin", 0) == []


class _FakeGeminiClient:
    """Fake google-genai client."""

    class _Models:
        def generate_content(self, **_: Any) -> Any:
            return types.SimpleNamespace(text="gemini answer")

    models = _Models()


class _TestGeminiManager(GeminiKeyManager):
    """Gemini manager with no external SDK construction."""

    def _build_client(self, api_key: str) -> Any:
        return _FakeGeminiClient()


def test_gemini_and_groq_clients_use_local_fakes(monkeypatch) -> None:
    """Test LLM client retry paths without network calls."""
    gemini = _TestGeminiManager("key", "model")
    assert gemini.invoke("prompt") == "gemini answer"

    class _TestGroq(GroqClient):
        def _build_client(self, api_key: str) -> Any:
            return object()

        def call_one(self, prompt: str) -> str:
            return f"answer:{prompt}"

    groq = _TestGroq("key", "model")
    answers, latencies = groq.batch_invoke(["a", "b"])
    assert answers == ["answer:a", "answer:b"]
    assert all(latency >= 0 for latency in latencies)


class _Delegate:
    """Fake LangChain chat delegate."""

    def __init__(self, failures: int = 0) -> None:
        self.failures = failures

    def _generate(self, **_: Any) -> str:
        if self.failures:
            self.failures -= 1
            raise RuntimeError("quota exceeded")
        return "chat-result"

    async def _agenerate(self, **_: Any) -> str:
        return "async-chat-result"


def test_rotating_gemini_chat_rotates_with_fake_delegate(monkeypatch) -> None:
    """Test Gemini chat rotation without constructing a real LangChain client."""
    delegates = iter([_Delegate(failures=1), _Delegate()])
    monkeypatch.setattr(
        RotatingGeminiChat,
        "_build_delegate",
        lambda self, api_key: next(delegates),
    )

    chat = RotatingGeminiChat(keys=["one", "two"], model="model")

    assert chat._llm_type == "rotating_gemini"
    assert chat._generate(messages=[]) == "chat-result"
    assert chat.active_key == "two"


def test_ragas_metrics_and_judge_wiring_with_fake_modules(monkeypatch) -> None:
    """Test RAGAS wiring using fake imported modules and wrappers."""
    captured: dict[str, Any] = {}

    def fake_evaluate(dataset: Any, **kwargs: Any) -> dict[str, Any]:
        captured["dataset"] = dataset
        captured.update(kwargs)
        return {"ok": True}

    fake_ragas = types.ModuleType("ragas")
    fake_ragas.evaluate = fake_evaluate
    fake_metrics = types.ModuleType("ragas.metrics")
    fake_metrics.faithfulness = object()
    fake_metrics.context_precision = object()
    fake_metrics.context_recall = object()
    fake_run_config = types.ModuleType("ragas.run_config")
    fake_run_config.RunConfig = lambda max_workers: {"max_workers": max_workers}

    class _Dataset:
        @staticmethod
        def from_pandas(df: pd.DataFrame) -> pd.DataFrame:
            return df

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.Dataset = _Dataset

    monkeypatch.setitem(sys.modules, "ragas", fake_ragas)
    monkeypatch.setitem(sys.modules, "ragas.metrics", fake_metrics)
    monkeypatch.setitem(sys.modules, "ragas.run_config", fake_run_config)
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    monkeypatch.setattr(
        ragas_metrics,
        "build_ragas_judge",
        lambda: ("llm", "emb", 3),
    )

    result = ragas_metrics.run_ragas(pd.DataFrame({"question": ["q"]}), workers=None)

    assert result == {"ok": True}
    assert captured["llm"] == "llm"
    assert captured["embeddings"] == "emb"
    assert captured["run_config"] == {"max_workers": 3}

    fake_hf = types.ModuleType("langchain_huggingface")
    fake_hf.HuggingFaceEmbeddings = lambda model_name: ("hf", model_name)
    fake_embeddings = types.ModuleType("ragas.embeddings")
    fake_embeddings.LangchainEmbeddingsWrapper = lambda emb: ("emb-wrap", emb)
    fake_llms = types.ModuleType("ragas.llms")
    fake_llms.LangchainLLMWrapper = lambda llm: ("llm-wrap", llm)

    monkeypatch.setitem(sys.modules, "langchain_huggingface", fake_hf)
    monkeypatch.setitem(sys.modules, "ragas.embeddings", fake_embeddings)
    monkeypatch.setitem(sys.modules, "ragas.llms", fake_llms)
    monkeypatch.setattr(ragas_judge, "load_gemini_keys", lambda: ["k1", "k2"])
    monkeypatch.setattr(
        ragas_judge,
        "RotatingGeminiChat",
        lambda **kwargs: ("chat", kwargs),
    )

    judge_llm, embeddings, worker_count = ragas_judge.build_ragas_judge()

    assert judge_llm[0] == "llm-wrap"
    assert embeddings[0] == "emb-wrap"
    assert worker_count == 1
