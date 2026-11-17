import json

import httpx
import pytest
from fastapi.testclient import TestClient
from test_workflow import TestEncoder

from agentic_rag.api import create_app
from agentic_rag.config import Settings
from agentic_rag.engine import Store


@pytest.fixture
def routing_setup(tmp_path):
    store = Store(tmp_path, TestEncoder())
    store.ingest(
        "football",
        "football.txt",
        (
            "American football grew from rugby. A touchdown scores six points. "
            "Passing rule changes and safety reforms shaped the sport."
        ).encode(),
    )
    store.ingest("space", "space.txt", b"Jupiter is a planet with many moons.")
    settings = Settings(
        data_dir=tmp_path,
        llm_base_url="http://provider/v1",
        api_key="",
        llm_model="qwen-test",
        routing_model="planner-test",
    )
    return store, settings


def mock_provider(monkeypatch, decision):
    calls = []

    async def post(self, url, **kwargs):
        body = kwargs["json"]
        calls.append(body)
        content = json.dumps(decision) if "response_format" in body else "Answer from evidence [1]."
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": content}}]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    return calls


@pytest.mark.parametrize(
    "question,method,complexity",
    [
        # A trigger word must not overrule the model's decision.
        ("Compare touchdown points with field goal points", "bm25", "lookup"),
        # A question can require graph reasoning without any former trigger words.
        (
            "How did school traditions lead to safety reforms and passing rules?",
            "graphrag",
            "multi_hop",
        ),
        ("What does advancing the ball involve?", "rag", "semantic"),
    ],
)
def test_model_decides_with_scoped_context(
    routing_setup, monkeypatch, question, method, complexity
):
    store, settings = routing_setup
    decision = {
        "method": method,
        "complexity": complexity,
        "reason": "Chosen from the evidence requirements.",
        "answerability": "likely",
    }
    calls = mock_provider(monkeypatch, decision)
    client = TestClient(create_app(settings, store))
    result = client.post("/knowledge-bases/football/query", json={"question": question})
    assert result.status_code == 200
    data = result.json()
    assert data["routing"] == {**decision, "source": "model", "model": "planner-test"}
    assert "model: planner-test" in data["answer"]
    assert len(calls) == 2
    assert calls[0]["model"] == "planner-test"
    assert calls[1]["model"] == "qwen-test"
    assert calls[0]["response_format"]["type"] == "json_schema"
    preview = json.loads(calls[0]["messages"][1]["content"])
    assert preview["question"] == question
    assert preview["knowledge_base_preview"]["id"] == "football"
    assert "Jupiter" not in json.dumps(preview)
    assert preview["knowledge_base_preview"]["graph"]["edges"] > 0
    if method == "graphrag":
        assert data["graph_trace"]["traversed_edges"]


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        '{"method":"unknown","reason":"Guess"}',
        '{"method":"bm25","complexity":"lookup","reason":"","answerability":"likely"}',
        '{"method":"bm25","complexity":"lookup","reason":"x","answerability":"certain"}',
        None,
    ],
)
def test_invalid_model_decision_is_visible_semantic_fallback(routing_setup, monkeypatch, raw):
    store, settings = routing_setup

    async def post(self, url, **kwargs):
        content = raw if "response_format" in kwargs["json"] else "Evidence answer [1]."
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": content}}]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    result = TestClient(create_app(settings, store)).post(
        "/knowledge-bases/football/query", json={"question": "Compare football codes"}
    )
    assert result.status_code == 200
    data = result.json()
    assert data["routing"]["method"] == "rag"
    assert data["routing"]["source"] == "fallback"
    assert "invalid decision" in data["routing"]["reason"]
    assert "(fallback)" in data["answer"]


@pytest.mark.parametrize("failure", ["timeout", "http"])
def test_router_failure_does_not_block_available_answer_provider(
    routing_setup, monkeypatch, failure
):
    store, settings = routing_setup

    async def post(self, url, **kwargs):
        if "response_format" in kwargs["json"]:
            if failure == "timeout":
                raise httpx.ReadTimeout("slow")
            return httpx.Response(503, request=httpx.Request("POST", url))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "Answer [1]."}}]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    data = (
        TestClient(create_app(settings, store))
        .post("/knowledge-bases/football/query", json={"question": "football"})
        .json()
    )
    assert data["routing"]["source"] == "fallback"
    assert data["routing"]["method"] == "rag"


def test_manual_override_skips_planner(routing_setup, monkeypatch):
    store, settings = routing_setup
    calls = mock_provider(monkeypatch, {})
    data = (
        TestClient(create_app(settings, store))
        .post("/knowledge-bases/football/query", json={"question": "touchdown", "method": "bm25"})
        .json()
    )
    assert data["routing"]["source"] == "manual"
    assert len(calls) == 1
    assert "response_format" not in calls[0]


def test_no_model_uses_visible_default_without_keyword_routing(routing_setup):
    store, settings = routing_setup
    settings.llm_base_url = ""
    client = TestClient(create_app(settings, store))
    data = client.post(
        "/knowledge-bases/football/query",
        json={"question": "Compare connections and evolution across football codes"},
    ).json()
    assert data["routing"]["method"] == "rag"
    assert data["routing"]["source"] == "fallback"
    assert client.get("/health").json()["router_mode"] == "semantic-fallback"


def test_cli_auto_uses_same_model_planner(routing_setup, monkeypatch, capsys):
    from agentic_rag import cli

    store, settings = routing_setup
    decision = {
        "method": "graphrag",
        "complexity": "multi_hop",
        "reason": "Requires linked evidence",
        "answerability": "uncertain",
    }
    calls = mock_provider(monkeypatch, decision)
    monkeypatch.setattr(cli, "Settings", lambda: settings)
    monkeypatch.setattr(cli, "Store", lambda *args: store)
    monkeypatch.setattr("sys.argv", ["rag-demo", "query", "football", "How did the sport develop?"])
    cli.main()
    data = json.loads(capsys.readouterr().out)
    assert data["routing"]["source"] == "model"
    assert data["routing"]["method"] == "graphrag"
    assert len(calls) == 1  # CLI retrieves evidence; it does not generate an answer.
