import copy

import httpx
import pytest
from fastapi.testclient import TestClient
from test_workflow import TestEncoder

from agentic_rag.api import create_app
from agentic_rag.config import Settings
from agentic_rag.engine import Store


@pytest.fixture
def kev_setup(tmp_path):
    store = Store(tmp_path, TestEncoder())
    store.ingest(
        "football", "rules.txt", b"A touchdown scores six points. Teams can pass the ball."
    )
    settings = Settings(
        data_dir=tmp_path,
        routing_provider="kev",
        routing_model="",
        kev_base_url="http://kev:8009",
        llm_base_url="http://answers/v1",
        llm_model="answer-only",
        api_key="",
    )
    return store, settings


def reply(choice="rag"):
    return {
        "model": "kev-latest",
        "answers": {
            "retrieval": {
                "type": "choice",
                "choice": choice,
                "confidence": 0.7,
                "probabilities": {
                    m: 0.8 if m == choice else 0.1 for m in ["bm25", "rag", "graphrag"]
                },
            }
        },
        "latency_ms": 10,
    }


def provider(monkeypatch, body):
    calls = []

    async def post(self, url, **kwargs):
        calls.append((url, kwargs))
        payload = (
            body
            if url.endswith("/v1/systemone")
            else {"choices": [{"message": {"content": "A touchdown scores six points [1]."}}]}
        )
        return httpx.Response(200, json=payload, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    return calls


@pytest.mark.parametrize("method", ["bm25", "rag", "graphrag"])
def test_kev_routes_all_methods_and_qwen_only_generates(kev_setup, monkeypatch, method):
    store, settings = kev_setup
    calls = provider(monkeypatch, reply(method))
    client = TestClient(create_app(settings, store))
    response = client.post(
        "/knowledge-bases/football/query", json={"question": "Compare touchdown rules"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["routing"]["method"] == method
    assert data["routing"]["source"] == "model"
    assert data["routing"]["provider"] == "kev"
    assert data["routing"]["model"] == "jaredpalmer/kev-4b@v1.0"
    assert data["routing"]["selected_probability"] == 0.8
    assert data["routing"]["reason_source"] == "policy_criterion"
    assert data["routing"]["answerability_source"] == "not_assessed"
    assert "Route scores:" in data["answer"]
    assert len(calls) == 2
    assert calls[0][0] == "http://kev:8009/v1/systemone"
    state = calls[0][1]["json"]["state"]
    assert state["user_question"] == "Compare touchdown rules"
    assert state["knowledge_base_preview"]["id"] == "football"
    assert set(calls[0][1]["json"]["questions"]["retrieval"]["criteria"]) == {
        "bm25",
        "rag",
        "graphrag",
    }
    assert calls[1][1]["json"]["model"] == "answer-only"
    assert client.get("/health").json()["routing_model"] == "jaredpalmer/kev-4b@v1.0"


@pytest.mark.parametrize(
    "bad",
    [
        "unknown",
        "missing",
        "nan",
        "negative",
        "sum",
        "choice_mismatch",
        "truncated",
        "empty",
    ],
)
def test_invalid_kev_decision_is_visible_fallback(kev_setup, monkeypatch, bad):
    store, settings = kev_setup
    body = copy.deepcopy(reply())
    choice = body["answers"]["retrieval"]
    if bad == "unknown":
        choice["choice"] = "other"
    elif bad == "missing":
        del choice["probabilities"]["bm25"]
    elif bad == "nan":
        # HTTP JSON cannot contain NaN, but a missing probability is also rejected.
        choice["probabilities"]["rag"] = None
    elif bad == "negative":
        choice["probabilities"]["rag"] = -0.1
    elif bad == "sum":
        choice["probabilities"]["rag"] = 0.3
    elif bad == "choice_mismatch":
        choice["choice"] = "bm25"
    elif bad == "truncated":
        body["truncated"] = True
    else:
        body = {}
    provider(monkeypatch, body)
    data = (
        TestClient(create_app(settings, store))
        .post("/knowledge-bases/football/query", json={"question": "touchdown"})
        .json()
    )
    assert data["routing"]["source"] == "fallback"
    assert data["routing"]["method"] == "rag"
    assert "invalid decision" in data["routing"]["reason"]


@pytest.mark.parametrize("failure", ["timeout", "http"])
def test_kev_provider_failure_falls_back_without_calling_chat_planner(
    kev_setup, monkeypatch, failure
):
    store, settings = kev_setup
    calls = []

    async def post(self, url, **kwargs):
        calls.append(url)
        if url.endswith("/v1/systemone"):
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
        .post("/knowledge-bases/football/query", json={"question": "touchdown"})
        .json()
    )
    assert data["routing"]["source"] == "fallback"
    assert calls == ["http://kev:8009/v1/systemone", "http://answers/v1/chat/completions"]


def test_manual_override_skips_kev_and_chat_endpoint_uses_kev(kev_setup, monkeypatch):
    store, settings = kev_setup
    calls = provider(monkeypatch, reply())
    client = TestClient(create_app(settings, store))
    client.post("/knowledge-bases/football/query", json={"question": "touchdown", "method": "bm25"})
    assert len(calls) == 1
    assert calls[0][0] == "http://answers/v1/chat/completions"
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "kb/football",
            "messages": [{"role": "user", "content": "Explain touchdown rules"}],
        },
    )
    assert response.status_code == 200
    assert "jaredpalmer/kev-4b@v1.0" in response.json()["choices"][0]["message"]["content"]


def test_kev_operates_without_an_answer_provider(kev_setup, monkeypatch):
    store, settings = kev_setup
    settings.llm_base_url = ""
    calls = provider(monkeypatch, reply())
    client = TestClient(create_app(settings, store))
    data = client.post("/knowledge-bases/football/query", json={"question": "touchdown"}).json()
    assert data["routing"]["source"] == "model"
    assert "Evidence-only" in data["answer"]
    assert len(calls) == 1
    assert client.get("/health").json()["router_mode"] == "model"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_kev_probabilities_are_rejected(value):
    from agentic_rag.routing import KevChoice

    choice = reply()["answers"]["retrieval"]
    choice["probabilities"]["rag"] = value
    with pytest.raises(ValueError):
        KevChoice.model_validate(choice)
