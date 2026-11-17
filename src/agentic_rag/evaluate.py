"""Run labeled, live model-routing checks without generating answers.

Example inside the backend container:
python -m agentic_rag.evaluate football /tmp/router_cases.json --output /tmp/report.json
"""

import argparse
import asyncio
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from .config import Settings
from .engine import Embedder, Store
from .routing import decide


class Case(BaseModel):
    question: str = Field(min_length=1, max_length=8000)
    expected: Literal["bm25", "rag", "graphrag"]
    reason: str


async def evaluate(store: Store, kb: str, cases: list[dict], settings: Settings) -> dict:
    results = []
    counts = Counter()
    confusion = defaultdict(Counter)
    for raw in cases:
        case = Case.model_validate(raw)
        start = time.perf_counter()
        context = store.routing_context(kb, case.question)
        decision = await decide(case.question, context, settings)
        actual = decision["method"] if decision["source"] == "model" else "fallback"
        counts[actual] += 1
        confusion[case.expected][actual] += 1
        results.append(
            {
                **case.model_dump(),
                "routing": decision,
                "matches_expected": actual == case.expected,
                "elapsed_seconds": round(time.perf_counter() - start, 3),
            }
        )
    return {
        "knowledge_base": kb,
        "model": settings.routing_identity,
        "provider": settings.routing_provider,
        "total": len(results),
        "matches_expected": sum(r["matches_expected"] for r in results),
        "method_counts": dict(counts),
        "confusion": {expected: dict(actual) for expected, actual in confusion.items()},
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kb")
    parser.add_argument("cases", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--provider", choices=["chat", "kev"], help="Override routing provider")
    parser.add_argument("--model", help="Override the configured planner for this evaluation only")
    parser.add_argument("--check", action="store_true", help="Exit nonzero on any label mismatch")
    args = parser.parse_args()
    settings = Settings()
    if args.provider:
        settings.routing_provider = args.provider
    if args.model:
        if settings.routing_provider == "kev":
            parser.error("Kev checkpoints are selected by its server; --model is for chat routing")
        settings.routing_model = args.model
    store = Store(settings.data_dir, Embedder(settings.embedding_model))
    cases = json.loads(args.cases.read_text())
    report = asyncio.run(evaluate(store, args.kb, cases, settings))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))
    if args.check and report["matches_expected"] != report["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
