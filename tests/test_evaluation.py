import asyncio

from agentic_rag import evaluate as evaluation
from agentic_rag.config import Settings


def test_evaluation_counts_fallback_separately_from_model_decisions(monkeypatch):
    class Store:
        def routing_context(self, kb, question):
            return {"id": kb}

    answers = iter(
        [
            {"source": "model", "method": "rag", "model": "planner", "reason": "Local mechanism"},
            {"source": "fallback", "method": "rag", "reason": "Provider unavailable"},
        ]
    )

    async def decide(question, context, settings):
        return next(answers)

    monkeypatch.setattr(evaluation, "decide", decide)
    cases = [
        {"question": q, "expected": "rag", "reason": "Local explanation"}
        for q in ["Explain a rule", "Explain another rule"]
    ]
    report = asyncio.run(evaluation.evaluate(Store(), "football", cases, Settings()))
    assert report["matches_expected"] == 1
    assert report["method_counts"] == {"rag": 1, "fallback": 1}
    assert report["confusion"]["rag"] == {"rag": 1, "fallback": 1}
    assert report["results"][1]["matches_expected"] is False
