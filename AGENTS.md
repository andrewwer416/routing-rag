# Repository Guidelines

## Layout

`src/agentic_rag/` contains the FastAPI/OpenAI adapter, retrieval engine, model-based planner (chat or Kev), settings and CLI.
`scripts/` contains optional OCR and local-Qwen startup helpers.
`tests/` contains isolated workflow tests; `examples/` holds a labeled original fixture.
`evaluations/` contains labeled live-routing checks and observed results.
`docs/architecture.md` describes retrieval and limits. `football.pdf` is the intended
Wikipedia input. Never silently replace it with a fixture.
Generated knowledge indexes live in ignored `data/`; do not commit them or model downloads.

## Development

Python 3.10+; create `.venv`, then `pip install -e '.[dev]'`. Copy `.env.example` to `.env`.
Run `rag-demo serve`, `rag-demo ingest <kb> <path>`, and `rag-demo list`.
Use `ruff check .`, `ruff format --check .`, and `pytest -q` before submitting changes.
Docker Compose provides retrieval + Open WebUI; exact commands are in README.md.

## Conventions and tests

Use standard Python naming, four-space indentation, type hints and Ruff formatting.
Keep ingestion provenance, model compatibility checks and KB isolation intact. Cover
new behavior and failures with pytest and injected encoders/provider mocks. Do not
claim the test encoder measures semantic quality. Real model tests require downloads.
Keep graph edges explicitly labeled co-occurrence, and document model-routing limits. Auto routing must use the model; do not reintroduce
keyword rules. Planner failure uses explicitly labeled semantic RAG. Test model decisions,
invalid responses, manual overrides and shared HTTP/CLI behavior. Re-run both live
routing sets when changing the planner, prompt or preview; mocked dispatch tests do not
measure whether the actual model selects semantic RAG. Kev returns class probabilities,
not generated reasoning: preserve the distinction between policy criteria and model
explanations. Do not fabricate confidence calibration or answerability estimates.

## Commits and security

Use short imperative commit subjects. PRs should describe behavior, purpose and validation.
Keep secrets in ignored `.env`, document new settings in `.env.example`, and avoid source
text/credentials in logs. Use one writing backend process per data directory. Shared
API-token auth is local-demo protection, not tenant isolation.
