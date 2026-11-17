"""Persisted, provenance-preserving indexes and inspectable adaptive retrieval."""

import hashlib
import io
import json
import re
import threading
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Protocol

import networkx as nx
import numpy as np
from pypdf import PdfReader
from rank_bm25 import BM25Plus

STOP = set(
    "a an the is are was were of to in on for and or with by from as at it that this "
    "what who when where how why which does do can about explain describe".split()
)


def tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[\w-]+", text.lower()) if t not in STOP and len(t) > 1]


class Encoder(Protocol):
    model: str

    def encode(self, texts: list[str], query: bool = False) -> np.ndarray: ...


class Embedder:
    def __init__(self, model: str):
        self.model = model
        self._encoder = None
        self._lock = threading.Lock()

    def encode(self, texts: list[str], query: bool = False) -> np.ndarray:
        with self._lock:
            if self._encoder is None:
                from fastembed import TextEmbedding

                self._encoder = TextEmbedding(model_name=self.model)
            # FastEmbed applies the selected model's query/document preprocessing.
            method = self._encoder.query_embed if query else self._encoder.passage_embed
            values = np.asarray(list(method(texts)), dtype=np.float32)
        return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def extract(filename: str, content: bytes) -> list[tuple[int, str]]:
    if not content:
        raise ValueError(f"{filename} is empty (0 bytes). Supply a non-empty PDF or text file.")
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        try:
            reader = PdfReader(io.BytesIO(content))
            pages = [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
        except Exception as exc:
            raise ValueError("Could not read PDF; check that it is valid and unencrypted.") from exc
    elif suffix in {".txt", ".md"}:
        try:
            pages = [(1, content.decode("utf-8"))]
        except UnicodeDecodeError as exc:
            raise ValueError("Text files must use UTF-8 encoding.") from exc
    else:
        raise ValueError("Supported files: .pdf, .txt, .md")
    if not any(text.strip() for _, text in pages):
        raise ValueError("No extractable text found. Run OCR on scanned PDFs before ingestion.")
    return pages


def chunk_document(filename: str, content: bytes) -> list[dict]:
    chunks = []
    for page, text in extract(filename, content):
        words = text.split()
        for start in range(0, len(words), 150):
            section = " ".join(words[start : start + 180])
            if not tokens(section):
                continue
            chunks.append(
                {
                    "id": str(len(chunks)),
                    "source": Path(filename).name,
                    "page": page,
                    "text": section,
                }
            )
            if start + 180 >= len(words):
                break
    if not chunks:
        raise ValueError("Document has no indexable words.")
    return chunks


def entities(text: str) -> set[str]:
    # Generic, deterministic extraction: named phrases plus salient lexical concepts.
    # Edges mean co-occurrence, never a claimed factual/causal relationship.
    names = re.findall(r"\b[A-Z][\w-]*(?:\s+[A-Z][\w-]*){0,3}\b", text)
    concepts = [t for t, _ in Counter(tokens(text)).most_common(10) if len(t) > 3]
    return {n.lower() for n in names if n.lower() not in STOP} | set(concepts)


def build_graph(chunks: list[dict]) -> dict:
    graph = nx.Graph()
    for chunk in chunks:
        nodes = sorted(entities(chunk["text"]))[:30]
        for node in nodes:
            if node not in graph:
                graph.add_node(node, chunks=[])
            graph.nodes[node]["chunks"].append(chunk["id"])
        for a, b in combinations(nodes, 2):
            if not graph.has_edge(a, b):
                graph.add_edge(a, b, weight=0, chunks=[])
            graph[a][b]["weight"] += 1
            graph[a][b]["chunks"].append(chunk["id"])
    groups = (
        list(nx.community.greedy_modularity_communities(graph))
        if graph.number_of_edges()
        else [{n} for n in graph]
    )
    communities = []
    for group in groups:
        ids = sorted({c for n in group for c in graph.nodes[n]["chunks"]}, key=int)
        communities.append(
            {
                "entities": sorted(group),
                "chunks": ids,
                "summary": " ".join(chunks[int(c)]["text"][:400] for c in ids[:4]),
            }
        )
    return {
        "nodes": [{"id": n, **d} for n, d in graph.nodes(data=True)],
        "edges": [{"source": a, "target": b, **d} for a, b, d in graph.edges(data=True)],
        "communities": communities,
    }


class Store:
    def __init__(self, root: Path, encoder: Encoder):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.encoder = encoder
        self._lock = threading.RLock()

    def path(self, kb: str) -> Path:
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", kb):
            raise ValueError("Knowledge-base ID must be 1–64 lowercase letters, digits, _ or -.")
        return self.root / f"{kb}.json"

    def list(self) -> list[dict]:
        return [
            {
                "id": p.stem,
                "documents": len(d["documents"]),
                "chunks": len(d["chunks"]),
                "embedding_model": d["embedding_model"],
            }
            for p in sorted(self.root.glob("*.json"))
            for d in [json.loads(p.read_text())]
        ]

    def load(self, kb: str) -> dict:
        path = self.path(kb)
        if not path.exists():
            raise KeyError(f"Unknown knowledge base: {kb}")
        return json.loads(path.read_text())

    def ingest(self, kb: str, filename: str, content: bytes) -> dict:
        path = self.path(kb)
        new = chunk_document(filename, content)
        digest = hashlib.sha256(content).hexdigest()
        with self._lock:
            data = (
                self.load(kb)
                if path.exists()
                else {"documents": [], "chunks": [], "embedding_model": self.encoder.model}
            )
            if data["embedding_model"] != self.encoder.model:
                raise ValueError("Embedding model differs; create a new knowledge base to reindex.")
            if any(d["sha256"] == digest for d in data["documents"]):
                return {"id": kb, "status": "already_indexed", "chunks": len(data["chunks"])}
            vectors = self.encoder.encode([c["text"] for c in new])
            old_count = len(data["chunks"])
            for i, chunk in enumerate(new):
                chunk["id"] = str(old_count + i)
            data["chunks"].extend(new)
            data.setdefault("vectors", []).extend(vectors.tolist())
            data["documents"].append({"filename": Path(filename).name, "sha256": digest})
            data["graph"] = build_graph(data["chunks"])
            temp = path.with_suffix(".tmp")
            temp.write_text(json.dumps(data))
            temp.replace(path)
        return {
            "id": kb,
            "status": "indexed",
            "chunks": len(data["chunks"]),
            "entities": len(data["graph"]["nodes"]),
        }

    def routing_context(self, kb: str, question: str) -> dict:
        """Bounded preview for the planner; lexical matches inform, never select, the route."""
        data = self.load(kb)
        chunks = data["chunks"]
        scores = BM25Plus([tokens(c["text"]) for c in chunks], delta=0).get_scores(tokens(question))
        best = [int(i) for i in np.argsort(-scores)[:3] if scores[i] > 0]
        return {
            "id": kb,
            "documents": [d["filename"][:120] for d in data["documents"][:10]],
            "chunk_count": len(chunks),
            "graph": {
                "entities": len(data["graph"]["nodes"]),
                "edges": len(data["graph"]["edges"]),
                "community_count": len(data["graph"]["communities"]),
            },
            "lexical_candidates": [
                {"page": chunks[i]["page"], "text": chunks[i]["text"][:650]} for i in best
            ],
        }

    def search(self, kb: str, question: str, method: str = "rag", top_k: int = 5) -> dict:
        # The orchestration layer supplies a validated model decision or explicit override.
        if method not in {"bm25", "rag", "graphrag"}:
            raise ValueError("Search requires a resolved bm25, rag or graphrag method")
        data = self.load(kb)
        chunks = data["chunks"]
        # Positive IDF avoids negative/zero matches in tiny or single-chunk KBs.
        lexical = BM25Plus([tokens(c["text"]) for c in chunks], delta=0)
        scores = lexical.get_scores(tokens(question))
        decision = {"method": method, "source": "manual", "reason": "Explicit retrieval method."}
        selected = decision["method"]
        trace = {"seed_entities": [], "traversed_edges": [], "communities": []}
        if selected in {"rag", "graphrag"}:
            if data["embedding_model"] != self.encoder.model:
                raise ValueError("Embedding model differs from index; restore configuration.")
            query = self.encoder.encode([question], query=True)[0]
            scores = np.asarray(data["vectors"]) @ query
        # RRF blends semantic seeds and graph evidence without comparing incompatible scores.
        if selected == "graphrag":
            graph = data["graph"]
            terms = set(tokens(question))
            seed_ids = set(np.argsort(-scores)[:2].astype(str))
            seeds = [n["id"] for n in graph["nodes"] if terms & set(tokens(n["id"]))]
            if not seeds:
                seeds = [n["id"] for n in graph["nodes"] if seed_ids & set(n["chunks"])][:8]
            trace["seed_entities"] = seeds
            frontier = set(seeds)
            evidence = Counter()
            for _ in range(2):
                next_nodes = set()
                for edge in graph["edges"]:
                    if edge["source"] in frontier or edge["target"] in frontier:
                        evidence.update(edge["chunks"])
                        next_nodes.update([edge["source"], edge["target"]])
                        if edge not in trace["traversed_edges"]:
                            trace["traversed_edges"].append(edge)
                frontier = next_nodes - frontier
            for community in graph["communities"]:
                if set(seeds) & set(community["entities"]):
                    trace["communities"].append(community)
                    evidence.update(community["chunks"])
            fused = np.zeros(len(chunks))
            for rank, i in enumerate(np.argsort(-scores)):
                fused[i] += 1 / (60 + rank + 1)
            for rank, (cid, _) in enumerate(evidence.most_common()):
                fused[int(cid)] += 1 / (60 + rank + 1)
            scores = fused
        if selected not in {"bm25", "rag", "graphrag"}:
            raise ValueError("Unknown retrieval method")
        indices = np.argsort(-scores)[:top_k]
        hits = [
            {**chunks[int(i)], "score": float(scores[i])}
            for i in indices
            if selected != "bm25" or scores[i] > 0
        ]
        return {
            "knowledge_base": kb,
            "question": question,
            "routing": decision,
            "hits": hits,
            "graph_trace": trace,
            "warning": "Retrieval ranks evidence; it does not prove answerability.",
        }
