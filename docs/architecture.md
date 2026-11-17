# Retrieval workflow and extension points

1. Upload PDF, UTF-8 text or Markdown via CLI or API. Reject empty, unreadable or textless
   documents; scanned PDFs need external OCR. Uploads are limited to 25 MiB at the HTTP API.
2. Extract pages, split into 180-word chunks with 30-word overlap, retain filename/page/ID.
3. Compute dense embeddings using FastEmbed's BGE small model. Store model identity with
   the vectors. Query and document embeddings use the respective FastEmbed methods,
   normalized for cosine similarity. Changing models requires a new KB/reingestion.
4. Extract capitalized phrases and frequent lexical concepts; build weighted undirected
   co-occurrence edges with originating chunk IDs. Partition with NetworkX greedy modularity.
   Community summaries are deterministic extractive excerpts, not LLM-generated facts.
5. Atomically persist a complete JSON snapshot per KB. Content hashes deduplicate uploads.
   Appending a document recomputes the graph and communities over all chunks. BM25 is
   reconstructed at query time (BM25Plus with delta=0 and positive IDF for small corpora);
   stored vectors and graph survive restarts.
6. Ask the configured LLM to choose BM25, semantic RAG, or GraphRAG. The planner receives
   the question, KB document names, up to three lexical candidate excerpts and graph
   counts. Unrelated representative passages and community previews are omitted because
   they biased the small planner toward graph synthesis. Lexical candidate
   search provides evidence for planning; its scores never directly select a route.
   The model judges exact lookup, semantic explanation, multiple reasoning hops or global
   synthesis, then returns a schema-constrained method, complexity, short reason and
   preliminary answerability estimate. Trigger words have no special behavior. A simple
   comparison may use BM25; connecting events can use GraphRAG without saying 'compare'.
   The preview is bounded, not a complete view of the KB, so answerability is an estimate.
   `ROUTING_MODEL` optionally selects a separate planner; otherwise it uses `LLM_MODEL`.
   Both use the same OpenAI-compatible provider. The local-Qwen deployment uses a 7B
   planner and a 1.5B answer model, kept resident together on the GPU. HTTP and CLI
   share this planner. Model explanations of local mechanisms use semantic RAG even
   when they involve multiple concepts or steps; graph traversal is reserved for
   indirect chains or corpus-wide synthesis.
   JSON validation rejects unknown methods and malformed decisions. With no model, or on
   planner timeout/provider/schema failure, use semantic RAG and explicitly label the
   decision `source: fallback`. There is no keyword-based fallback. Explicit API/CLI
   overrides skip the planner and are labeled `source: manual`. Model decisions carry
   `source: model` and the model name. The answer header shows that source and reason.
7. BM25 returns lexical matches. Semantic RAG uses cosine ranking. GraphRAG identifies
   query entities (or semantic seed entities), expands two graph hops, selects related
   communities, and fuses graph evidence ranking with semantic ranking using reciprocal
   rank fusion (constant 60). Return original passages and graph trace for inspection.
8. Optionally synthesize an answer from passages using a configurable chat provider,
   requesting citations and abstention when evidence is insufficient. Open WebUI sees
   one model per KB through `/v1/models` and `/v1/chat/completions`.

This is a lightweight graph-augmented RAG demonstration, **not Microsoft's GraphRAG
package or its full global map/reduce pipeline**. Both graph and semantic retrieval share
the indexed document embeddings. The graph has co-occurrence edges, not typed factual
relations; deterministic extraction can merge homonyms, miss lowercase concepts, and
produce noisy edges. Source passages, rather than edges alone, ground generated claims.

Sources for implementation choices: [FastEmbed](https://qdrant.github.io/fastembed/),
[NetworkX communities](https://networkx.org/documentation/stable/reference/algorithms/community.html),
and [Open WebUI connection protocol](https://docs.openwebui.com/getting-started/quick-start/connect-a-provider/starting-with-openai-compatible/).

## Scope and next steps

The demo prioritizes inspectability over scale: dense vectors and graph snapshots load
into memory, graph construction can grow quadratically per chunk, and greedy modularity
can be expensive. Use a single backend worker/process against a data directory; the lock
serializes ingestion within that process, not across independent CLI/server processes.
Avoid concurrent CLI writes while the server is ingesting; use the upload API instead.
Atomic replacement lets readers see complete snapshots. No pickle or executable indexes.

Retrieval scores do not certify answerability. The routing prompt is written in English;
small models can overestimate graph complexity or misclassify questions. Out-of-domain
questions can still retrieve
irrelevant evidence. Generated
abstention and citation correctness depend on the model and are not guaranteed. A rigorous
next step is a manually labeled set of factual, paraphrase, comparison, multi-hop and
unanswerable football questions, measuring routing accuracy, recall@k, answer faithfulness,
latency and cost across all three forced routes. The sample fixture tests plumbing, not retrieval quality on Wikipedia. The labeled
checks in `evaluations/` test routing policy on 12 development and six separate football
questions; they do not measure answer faithfulness or broad benchmark accuracy.

For larger or multilingual deployments, replace entity extraction with NER/typed relation
extraction, evaluate/tune the model router and add query reformulation, use a vector
store and graph database, add background ingestion jobs and tenant-specific authorization.
The shared API token grants access to every KB; it is not a multi-user authorization system.
Filenames are display metadata and are never used as storage paths; KB IDs are validated.
Store directories contain source text and should be protected and excluded from Git.

## Kev provider

`ROUTING_PROVIDER=kev` replaces only retrieval selection. The state contains the scoped
KB preview and user question, and a typed `choice` question supplies the three algorithm
criteria. No code branches on question keywords. The adapter validates all three scores,
finite/range/sum constraints, selected option and agreement with the largest probability.
Truncated, malformed or failed responses use the labeled semantic fallback. Kev cannot
generate a natural-language rationale or answerability estimate; the response labels its
reason as `policy_criterion` and answerability as `not_assessed`. Class probabilities and
the vendor confidence are carried through without claiming task-specific calibration.

The separate server uses pinned upstream code and a cached pretrained checkpoint. BF16
and disabled CUDA-graph caches reduce memory use; the server explicitly requires CUDA.
Only the answer model remains in Ollama. Model scores are useful for evaluation, but a
confidence threshold should be selected on representative labels before enabling
automatic abstention or escalation. Live routing checks accept `--provider kev` or
`--provider chat` to select the implementation being measured. Stop unused large models
when comparing on a shared GPU.
