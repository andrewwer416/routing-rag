import argparse
import asyncio
import json
from pathlib import Path

from .config import Settings
from .engine import Embedder, Store
from .routing import retrieve


def main():
    parser = argparse.ArgumentParser(description="Manage adaptive retrieval knowledge bases")
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest", help="Create a KB or append a document (deduplicated)")
    ingest.add_argument("kb")
    ingest.add_argument("file", type=Path)
    commands.add_parser("list")
    query = commands.add_parser(
        "query", help="Inspect retrieval; use HTTP API for generated answers"
    )
    query.add_argument("kb")
    query.add_argument("question")
    query.add_argument("--method", choices=["auto", "bm25", "rag", "graphrag"], default="auto")
    commands.add_parser("serve")
    args = parser.parse_args()
    settings = Settings()
    store = Store(settings.data_dir, Embedder(settings.embedding_model))
    try:
        if args.command == "ingest":
            result = store.ingest(args.kb, args.file.name, args.file.read_bytes())
        elif args.command == "list":
            result = store.list()
        elif args.command == "query":
            result = asyncio.run(retrieve(store, args.kb, args.question, settings, args.method))
        else:
            import uvicorn

            uvicorn.run("agentic_rag.api:app", host="0.0.0.0", port=8000)
            return
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(json.dumps(result, indent=2))
