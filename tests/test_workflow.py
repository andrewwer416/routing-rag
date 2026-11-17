import hashlib

import numpy as np
import pytest
from fastapi.testclient import TestClient

from agentic_rag.api import create_app
from agentic_rag.config import Settings
from agentic_rag.engine import Store, chunk_document, tokens


class TestEncoder:
    __test__ = False
    model = "test-hashing-not-semantic"

    def encode(self, texts, query=False):
        result = np.zeros((len(texts), 128), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in tokens(text):
                result[i, int(hashlib.sha256(word.encode()).hexdigest(), 16) % 128] += 1
        return result / np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1e-12)


@pytest.fixture
def setup(tmp_path):
    store = Store(tmp_path, TestEncoder())
    store.ingest(
        "football",
        "rules.txt",
        (
            "FIFA governs association football. Association football has eleven players per team. "
            "Rugby permits carrying the ball. American football developed from Rugby."
        ).encode(),
    )
    client = TestClient(create_app(Settings(data_dir=tmp_path, llm_base_url="", api_key=""), store))
    return store, client


def test_empty_pdf_and_invalid_documents():
    with pytest.raises(ValueError, match="0 bytes"):
        chunk_document("football.pdf", b"")
    with pytest.raises(ValueError, match="Could not read"):
        chunk_document("football.pdf", b"not a pdf")
    with pytest.raises(ValueError, match="Supported"):
        chunk_document("data.exe", b"data")


def test_ingest_deduplicate_append_isolate_and_reload(setup):
    store, _ = setup
    content = b"Different knowledge about astronomy and Jupiter."
    first = store.ingest("space", "space.txt", content)
    assert first["status"] == "indexed"
    assert store.ingest("space", "space.txt", content)["status"] == "already_indexed"
    store.ingest("football", "history.txt", b"Gaelic football is played in Ireland.")
    assert len(store.load("football")["documents"]) == 2
    assert len(store.load("space")["documents"]) == 1
    reloaded = Store(store.root, TestEncoder())
    assert reloaded.search("space", "Jupiter", "bm25")["hits"][0]["source"] == "space.txt"
    with pytest.raises(ValueError):
        store.ingest("../escape", "a.txt", content)


@pytest.mark.parametrize(
    "question,method",
    [
        ("Who governs association football?", "bm25"),
        ("Explain the purpose of international sporting governance", "rag"),
        ("Compare Rugby and American football connections", "graphrag"),
    ],
)
def test_retrieval_methods_and_provenance(setup, question, method):
    store, _ = setup
    result = store.search("football", question, method)
    assert result["routing"]["method"] == method
    assert result["hits"][0]["page"] == 1
    if method == "graphrag":
        assert result["graph_trace"]["traversed_edges"]
        assert result["graph_trace"]["communities"]


def test_api_models_queries_streaming_and_failures(setup):
    _, client = setup
    assert client.get("/v1/models").json()["data"][0]["id"] == "kb/football"
    response = client.post(
        "/knowledge-bases/football/query", json={"question": "Who governs association football?"}
    )
    assert response.status_code == 200
    assert "Evidence-only" in response.json()["answer"]
    payload = {
        "model": "kb/football",
        "messages": [{"role": "user", "content": "Who governs association football?"}],
    }
    reply = client.post("/v1/chat/completions", json=payload).json()
    assert reply["choices"][0]["message"]["role"] == "assistant"
    stream = client.post("/v1/chat/completions", json={**payload, "stream": True})
    assert '"finish_reason": "stop"' in stream.text
    assert stream.text.endswith("data: [DONE]\n\n")
    assert (
        client.post("/knowledge-bases/missing/query", json={"question": "hello"}).status_code == 404
    )
    assert (
        client.post(
            "/knowledge-bases/football/query", json={"question": "hello", "method": "invalid"}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/knowledge-bases/football/documents",
            files={"file": ("football.pdf", b"", "application/pdf")},
        ).status_code
        == 400
    )
    assert (
        client.post(
            "/knowledge-bases/football/documents",
            files={"file": ("new.txt", b"New knowledge about referees.", "text/plain")},
        ).status_code
        == 200
    )


def test_auth_and_embedding_mismatch(setup):
    store, _ = setup
    client = TestClient(create_app(Settings(api_key="secret", llm_base_url=""), store))
    assert client.get("/health").status_code == 200
    assert client.get("/v1/models").status_code == 401
    assert client.get("/v1/models", headers={"Authorization": "Bearer secret"}).status_code == 200
    store.encoder.model = "different-model"
    with pytest.raises(ValueError, match="differs"):
        store.search("football", "explain sporting rules", "rag")
    with pytest.raises(ValueError, match="differs"):
        store.ingest("football", "extra.txt", b"Extra facts")


def test_generated_answer_uses_evidence_and_provider_failure(setup, monkeypatch):
    store, _ = setup
    import httpx

    requests = []

    async def post(self, url, **kwargs):
        requests.append(kwargs["json"])
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "FIFA governs association football [1]."}}]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    client = TestClient(create_app(Settings(llm_base_url="http://model/v1", api_key=""), store))
    response = client.post(
        "/knowledge-bases/football/query", json={"question": "Who governs association football?"}
    )
    assert "FIFA governs association football [1]." in response.json()["answer"]
    assert "rules.txt" in requests[0]["messages"][1]["content"]
    assert requests[-1]["max_tokens"] == 512

    async def fail(self, url, **kwargs):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(httpx.AsyncClient, "post", fail)
    assert (
        client.post(
            "/knowledge-bases/football/query",
            json={"question": "Who governs association football?"},
        ).status_code
        == 502
    )


def test_pdf_text_extraction_preserves_page_numbers():
    import io

    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    for text in ["Association football has eleven players.", "Rugby permits carrying the ball."]:
        page = writer.add_blank_page(width=600, height=800)
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
        )
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 50 700 Td ({text}) Tj ET".encode())
        page[NameObject("/Contents")] = stream
    content = io.BytesIO()
    writer.write(content)
    chunks = chunk_document("football.pdf", content.getvalue())
    assert [c["page"] for c in chunks] == [1, 2]
    assert "eleven players" in chunks[0]["text"]


def test_bm25_absent_terms_returns_no_evidence(setup):
    store, _ = setup
    assert store.search("football", "astronomy jupiter", "bm25")["hits"] == []
