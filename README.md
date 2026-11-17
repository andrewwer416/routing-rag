# Adaptive RAG + Open WebUI

A local demonstration that ingests a document into **BM25**, **semantic embeddings**, and a
**concept/entity graph**, then routes each question to a retrieval strategy. Every answer
shows the selected strategy and source references. Add documents to existing knowledge
bases or create new ones; each knowledge base appears as an Open WebUI model.

The repository's `football.pdf` is an image-only Wikipedia export about American football.
Create a searchable copy using the OCR instructions below before ingestion. The labeled
`examples/football-sample.txt` is an original small fixture for quick smoke tests.

## Run locally

Requires Python 3.10+ (3.11 recommended). First ingestion downloads the FastEmbed
`BAAI/bge-small-en-v1.5` ONNX embedding model; internet access and disk space are required.
Subsequent use can reuse the model cache. No chat model or API key is needed for evidence-only mode.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env
rag-demo ingest football-sample examples/football-sample.txt
# Index the supplied Wikipedia PDF:
rag-demo ingest football football.pdf
rag-demo serve
```

The server listens on port 8000. Open http://localhost:8000/docs for upload and query forms.
The API has no authentication by default and is intended for local use. Set `API_KEY` before
sharing it; local `serve` binds all interfaces. Docker exposes ports on loopback only.

```bash
# Create a new KB, or append a document to an existing KB:
rag-demo ingest my-knowledge /path/to/document.pdf
rag-demo ingest my-knowledge /path/to/another-document.md
rag-demo list
rag-demo query football-sample 'Who governs association football?'
rag-demo query football-sample 'Explain the purpose of international sporting governance'
rag-demo query football-sample 'Compare Rugby and American football connections'
# Override routing for a side-by-side comparison:
rag-demo query football-sample 'Compare football codes' --method bm25
rag-demo query football-sample 'Compare football codes' --method rag
rag-demo query football-sample 'Compare football codes' --method graphrag
```

## Open WebUI + containers

Choose the instructions for your installed container runtime. `docker --version` may
report Podman when Ubuntu's `podman-docker` compatibility package is installed.

### Podman (including Ubuntu 22.04 / Podman 3.4)

The containers extra pins `podman-compose` 1.0.6 for this older Podman runtime.
Newer Compose versions can pass incompatible network and log options to Podman 3.4.
Older Podman versions do not implement `podman compose`. Invoke the separate
`podman-compose` program directly. Install it into the project's virtual environment:

```bash
python3 -m venv .venv  # skip if already created
.venv/bin/pip install -e '.[containers]'
.venv/bin/podman-compose -f compose.yaml up --build -d
.venv/bin/podman-compose -f compose.yaml ps
podman logs --tail=100 agentic-rag-openwebui-demo_retrieval_1
podman logs --tail=100 agentic-rag-openwebui-demo_openwebui_1
```

Keep your existing `.env`; copy `.env.example` only if `.env` does not exist. Stop with
`.venv/bin/podman-compose -f compose.yaml down` (named volumes remain).
Podman 3.4 uses CNI networking and needs the `dnsname` plugin for service-name resolution;
on Ubuntu this is the `podman-dnsname` package. Fully qualified image names in the
Dockerfile avoid interactive registry-selection prompts.

See the [Podman Compose installation and compatibility notes](https://github.com/containers/podman-compose).
Creating `/etc/containers/nodocker` only hides the emulation notice; it does not add Compose.

### Docker Engine

Requires Docker Engine with Compose v2. No Docker installation is bundled.

```bash
cp .env.example .env
docker compose up --build -d
# Ingest into the running backend (not a separate host-side data directory):
curl -f -X POST http://localhost:8000/knowledge-bases/football-sample/documents \
  -F file=@examples/football-sample.txt
# Index the supplied Wikipedia PDF:
curl -f -X POST http://localhost:8000/knowledge-bases/football/documents \
  -F file=@football.pdf
```

Visit http://localhost:3000, create the initial admin account, and select
`kb/football-sample` or `kb/football`. Refresh model discovery after adding a new KB.
Compose configures the OpenAI-compatible backend at `http://retrieval:8000/v1`.
If an existing Open WebUI database has saved connections, set that URL manually under
**Admin Settings → Connections → OpenAI-compatible**. Use the configured `API_KEY`, or
`local-demo` when backend authentication is disabled. With `API_KEY` set, add
`-H 'Authorization: Bearer YOUR_TOKEN'` to curl requests.

For an existing Open WebUI outside Compose, add a connection to a reachable backend URL
ending in `/v1`. A container cannot reach a host service using `localhost`; on Linux add
`host.docker.internal:host-gateway` and use `http://host.docker.internal:8000/v1`.

Do not attach the same documents through Open WebUI's built-in Knowledge feature for this
demo: ingestion and routing live in this backend, while Open WebUI provides the chat UI.
This integration uses Open WebUI's [OpenAI-compatible connections](https://docs.openwebui.com/getting-started/quick-start/connect-a-provider/starting-with-openai-compatible/).

The Compose default uses the moving Open WebUI `main` image for convenience. Set
`OPENWEBUI_TAG` to a tested release or digest for a reproducible deployment.

## Kev-4B routing with Qwen answers

This is the selected local configuration. Kev classifies the query as `bm25`, `rag`, or
`graphrag`; Qwen2.5 1.5B generates the answer from the selected evidence. Open WebUI
continues to use **kb/football**. No documents need to be reindexed when changing routers.

```bash
.venv/bin/pip install -e '.[containers]'
python3 scripts/start_local_kev.py
```

The helper configures the existing Podman project network, selects WSL GPU passthrough,
unloads the previous Qwen routing model, starts Kev, verifies its CUDA device, and warms
the answer model. First setup downloads several GB of CUDA dependencies and base-model
weights. Model files persist in the `kev-models` volume. The existing knowledge and
Open WebUI volumes are retained. Inspect startup using:

```bash
podman logs --tail=100 agentic-rag-openwebui-demo_kev_1
curl -f http://localhost:8000/health
```

Kev uses its own `/v1/systemone` API, not an Ollama model alias. The adapter sends one
multiple-choice question with three retrieval criteria and preserves the resulting
probabilities in query JSON and the chat response. Kev does not generate a rationale:
the displayed explanation is explicitly the selected **policy criterion**. Its scores
have not been calibrated on this demo's labels and are not verified accuracy estimates.
Answerability is reported as unassessed, rather than invented by the adapter.

The sidecar pins upstream Kev code to commit
`6b719c3c3f367295f6ef336f4f751cf5ff970abc` and the model to
`jaredpalmer/kev-4b@v1.0`. It uses BF16, disables CUDA graph buffers and fused kernels,
and bounds its prefix cache to leave memory for the answer model on a 16 GB GPU.
It requires CUDA and fails startup rather than silently selecting CPU. The WSL GPU
mounts apply to both servers. Other Linux GPU hosts need their own runtime/device setup.

Compose files, in order: `compose.yaml`, `compose.podman.yaml`, `compose.qwen.yaml`,
`compose.qwen.wsl-gpu.yaml`, `compose.kev.yaml`, `compose.kev.wsl-gpu.yaml`.
The helper selects the WSL files automatically. Keep the Kev files in future Compose
commands to preserve the router selection. `python3 scripts/start_local_qwen.py`
selects the previous Qwen-only configuration for comparisons.

Provider outages or invalid probability distributions use the existing explicitly
labeled semantic-RAG fallback. Explicit `method` overrides still bypass model routing.
Generic backend configuration uses `ROUTING_PROVIDER=kev`, `KEV_BASE_URL`, and
`KEV_API_KEY`; the Kev overlay sets these automatically.
Initial live routing checks matched **20/26 intended labels**. Kev selected all three
methods, including GraphRAG for the short Super Bowl qualification question. It did not
outperform Qwen on the earlier shared test set; see the [evaluation notes](evaluations/README.md).
See [upstream Kev](https://github.com/jaredpalmer/kev) for its architecture and license.

## Local Qwen test (GPU on WSL, CPU elsewhere)

Use the optional overlay to host Qwen2.5 with Ollama: **7B for routing** and
**1.5B for answer generation**. Model downloads are about 4.7 GB and 1 GB, plus the
Ollama container image. The model
cache persists in a named volume. On WSL2 with an NVIDIA GPU, the startup helper
selects GPU passthrough automatically. Otherwise the base overlay runs on CPU.
Run from the repository:

```bash
# Podman: preserves a prevalidated network on Ubuntu 22.04 / Podman 3.4.
python3 scripts/start_local_qwen.py
# This particular image-only PDF needs OCR first (skip if already generated):
.venv/bin/pip install -e '.[ocr]'
python scripts/ocr_pdf.py football.pdf data/football-searchable.pdf
curl -f -X POST http://localhost:8000/knowledge-bases/football/documents \
  -F file=@data/football-searchable.pdf
curl -f http://localhost:8000/health
curl -f http://localhost:8000/v1/models
```

The health endpoint should report `answer_mode: llm`; the model list should contain
`kb/football`. Refresh Open WebUI and select **kb/football** in the model picker.
The 7B planner selects retrieval; the 1.5B model writes the answer. Both run behind
that knowledge-base model; it does not appear as a separate raw-chat
model through this backend. If the picker is empty, verify the OpenAI-compatible
connection points to `http://retrieval:8000/v1` and refresh discovery. Ingestion is
required even when the PDF exists on the host: the backend uses its own persistent index.

For Docker, use `docker compose -f compose.yaml -f compose.qwen.yaml up -d`,
then pull both `qwen2.5:1.5b` and `qwen2.5:7b` using
`docker compose -f compose.yaml -f compose.qwen.yaml exec ollama ollama pull <model>`.
The Podman helper also loads `compose.podman.yaml`, which marks the network as external
so old Podman cannot recreate it with an unsupported CNI version. It backs up the
project network configuration before applying the version workaround.
Always include both Compose files when updating this local-Qwen stack. Using only the
base file can reset retrieval to the provider settings from `.env`.
`QWEN_MODEL` selects the answer model; `ROUTING_MODEL` selects the planner
(default `qwen2.5:7b` in the local-Qwen overlay). The helper downloads both effective
models, including values from `.env`, and warms each model before the first chat.
Ollama keeps up to two models loaded for 30 minutes
to avoid reloading them on every request. Adjust `OLLAMA_MAX_LOADED_MODELS` and
`OLLAMA_KEEP_ALIVE` if GPU memory is limited. Ollama is accessible only on
the internal project network; no extra host port is published.

On WSL2, `compose.qwen.wsl-gpu.yaml` passes `/dev/dxg` and mounts `/usr/lib/wsl`
read-only so Ollama can load the Windows-provided CUDA driver libraries. This was
validated with rootless Podman 3.4 and an RTX 5080. This overlay is WSL-specific;
ordinary Linux NVIDIA hosts should use NVIDIA Container Toolkit with a supported
runtime instead. No Linux NVIDIA kernel driver is installed inside WSL or the container.
Use `python3 scripts/start_local_qwen.py --cpu` only when CPU inference is intended.

After sending a chat request, verify the model actually runs on the GPU:

```bash
podman exec agentic-rag-openwebui-demo_ollama_1 ollama ps
```

The processor column should say `100% GPU`. When managing this setup directly with
Compose, include `-f compose.qwen.wsl-gpu.yaml` along with the base, Podman and Qwen
files so container recreation preserves GPU access. The startup helper adds it for you.
GPU passthrough accelerates Ollama routing and generation; graph traversal and the
FastEmbed vector search still run on CPU. Answer generation is bounded by
`ANSWER_MAX_TOKENS` (default 512), preventing unbounded small-model repetition. Increase
the limit in `.env` if longer answers are needed.

Try: “How many points is a touchdown worth?”, “Explain the purpose of the line of scrimmage”,
and “Compare American football and rugby”. Small model answers and citations
still need checking against the displayed source passages.

## Generated answers (optional)

Point the backend at an OpenAI-compatible chat provider. For a local Ollama server (Docker):

```bash
ollama pull llama3.2:3b
# .env for Docker retrieval connecting to host Ollama:
# LLM_BASE_URL=http://host.docker.internal:11434/v1
# LLM_MODEL=llama3.2:3b
# LLM_API_KEY=local
docker compose -f compose.yaml -f compose.docker-host.yaml up -d retrieval
```

The Docker host override adds the host-gateway mapping only when needed. The base
Compose file omits it for compatibility with older Podman. With Podman, use its
`host.containers.internal` hostname in `LLM_BASE_URL` and verify host reachability for
your rootless network setup; do not load `compose.docker-host.yaml`.

For a host-side backend use `http://localhost:11434/v1`. Ollama must listen on an address
reachable from Docker if using the container setup. Hosted providers are also supported;
set their base URL, model and key. Evidence is sent to the configured chat provider.
If the provider fails, the backend returns a clear 502 error. Without `LLM_BASE_URL`, it
returns retrieved excerpts explicitly labeled **evidence-only**, without synthesizing an answer.

## How routing works

For `method: auto` (the default), the model first plans retrieval. It receives the question
and a bounded KB preview of metadata and lexical candidate passages, estimates the evidence and reasoning needed, and returns a
validated choice of BM25, semantic RAG, or GraphRAG with a short explanation. The model
then answers using the passages retrieved by that choice. **Keywords do not force routes.**
A simple comparison can be an exact lookup; connecting several historical events can
require GraphRAG even without a word such as “compare”.

With `ROUTING_PROVIDER=chat`, the generic backend uses `LLM_MODEL` for planning unless
`ROUTING_MODEL` is set. With `ROUTING_PROVIDER=kev`, the decision server scores the
three criteria; it returns probabilities instead of a generated rationale.
The local-Qwen overlay defaults to a separate 7B planner: the original 1.5B planner
failed the semantic-RAG routing checks even with clearer instructions. Set `ROUTING_MODEL` to
another model available on the same provider if desired. `ROUTING_TIMEOUT` defaults
to 60 seconds. Model output is schema-constrained for chat routing; those providers must support OpenAI-style
`response_format: json_schema`. Invalid output, provider failures or no configured model
produce an explicitly labeled **semantic RAG fallback**, never a keyword decision.
If the answer provider is also unavailable, answer generation still returns a 502.

For chat routers, query JSON includes `routing.source` (`model`, `manual`, or `fallback`), `method`, and
`reason`; successful model decisions also include the model name, estimated complexity,
and preliminary answerability. These estimates are not calibrated confidence scores.
The answer header exposes the choice and decision source in Open WebUI. API `method`
and CLI `--method` remain available for controlled comparisons and bypass the planner.
`/health` reports whether model routing is configured. CLI auto queries also use the model
planner, but return retrieved evidence without the second answer-generation call.

## Inspect the workflow

```bash
curl -f http://localhost:8000/v1/models
curl -f http://localhost:8000/knowledge-bases
curl -f http://localhost:8000/knowledge-bases/football-sample/graph
curl -f http://localhost:8000/knowledge-bases/football-sample/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"Compare Rugby and American football connections","method":"auto","top_k":5}'
```

Query results include routing reasoning, ranked passages with page/chunk IDs, selected
communities and traversed graph edges. HTTP queries generate answers when configured;
CLI queries inspect retrieval only. Chat supports text requests and OpenAI SSE framing;
it sends the completed answer after retrieval/generation, rather than live token streaming.
Questions are standalone: follow-up pronouns are not resolved from chat history.

## Check model routing quality

The labeled checks in `evaluations/` verify the intended distinction between exact
lookup, semantic explanation and graph synthesis. They call the actual routing model,
without generating answers. Unit tests with mocked models verify dispatch and failures;
they do not measure model routing quality.

```bash
podman cp evaluations/router_cases.json agentic-rag-openwebui-demo_retrieval_1:/tmp/router_cases.json
podman exec agentic-rag-openwebui-demo_retrieval_1 python -m agentic_rag.evaluate \
  football /tmp/router_cases.json --output /tmp/routing-report.json --check
```

Repeat with `router_holdout.json` for the separate check set. Use `--model` to compare
another downloaded chat planner with `--provider chat` (stop Kev first if GPU memory
is limited). Kev checkpoints are selected by its server, not `--model`. Reports include
method counts, expected-versus-actual
counts, explanations and failures; fallbacks never count as successful model decisions.
See [evaluation notes](evaluations/README.md) for observed results and limitations.

## Checks

```bash
ruff check .
ruff format --check .
pytest -q
```

Tests use an explicitly non-semantic deterministic test encoder, never as the production
embedding model, and mock the chat provider. See [architecture](docs/architecture.md)
for model routing, graph semantics, persistence and limitations.

## Podman 3.4 troubleshooting

Use `podman logs` directly on this version: newer `podman-compose logs` can send an
unsupported `--color` flag. This is a log-viewer error, not an application crash.

If startup reports `address already in use`, inspect `ss -ltnp` before restarting.
A successful later start means the earlier conflict has cleared. Keep unrelated services
running; change the host port mapping if the conflict persists (the internal API port stays 8000).

On Ubuntu 22.04, a generated project CNI configuration can declare version `1.0.0`
when the installed firewall plugin only supports `0.4.0`. Back up the project's file
under `~/.config/cni/net.d/`, change only its `cniVersion` to `0.4.0`, then restart the
project containers. Do not reset Podman's storage. See the
[upstream discussion](https://github.com/podman-container-tools/podman/discussions/17362).
The Compose file explicitly attaches both services to the project network.

Check both host access and service-name resolution:

```bash
curl -f http://localhost:8000/health
curl -I http://localhost:3000
podman exec agentic-rag-openwebui-demo_openwebui_1 python -c \
  "import urllib.request; print(urllib.request.urlopen('http://retrieval:8000/health', timeout=10).read().decode())"
```

## OCR for image-only PDFs

The optional `ocr` dependencies support `scripts/ocr_pdf.py`. It renders pages locally,
recognizes text with RapidOCR, and adds an invisible searchable layer to a new PDF.
Page order and original page images are retained; the input is never overwritten.
Generated copies belong in ignored `data/`. OCR can misread names, columns and tables;
compare quoted passages against the original image. Runtime is CPU-dependent and
progress is printed per page. OCR dependencies are only needed on the machine creating
the searchable copy; the backend accepts the resulting PDF through normal ingestion.
