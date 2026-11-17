# Routing quality checks

The fixtures define this demo's intended retrieval policy: named/exact facts use BM25;
local explanations and paraphrases use semantic RAG; indirect chains and collection-wide
synthesis use GraphRAG. More than one algorithm can sometimes retrieve useful evidence;
these are policy labels, not mathematically unique answers.

Observed on the indexed American football Wikipedia PDF, using GPU Ollama and
schema-constrained routing:

| Planner / prompt | Matched labels | Model-selected RAG |
| --- | --- | --- |
| Original 1.5B planner and original preview | 6/12 | 0/12 |
| 1.5B with revised instructions and focused preview | 7/12 | 0/12 |
| 7B with revised instructions and focused preview | 12/12 | 4/12 |
| 7B on the separate six-question set | 6/6 | 2/6 |

The revised prompt examples cover other domains, not these football questions. The six
separate questions were checked after choosing the candidate instructions and model.
No code recognizes these questions or forces their routes. Both sets are small and
hand-labeled; these results do not establish general routing accuracy or answer quality.
The scores concern selected methods, not exact agreement on complexity labels or reasons.

Examples now selecting regular RAG:

- Explain in plain English how a zone defense works.
- Why does earning a fresh set of downs help an offense keep possession?
- Explain what a quarterback is trying to accomplish with a play-action pass.

Run `python -m agentic_rag.evaluate <kb> <cases.json> --output <report.json> --check`
in the backend environment. `--model` evaluates another installed planner without
changing server settings. `--check` exits nonzero on label mismatches. Keep generated
reports under ignored `data/` or `/tmp`. Provider failures are reported as fallbacks;
they never count as correct model-selected RAG. Recheck both sets after changing the
planner, prompt or knowledge-base contents. Expand the labels to other knowledge bases
before relying on this as a general-purpose router.

## Kev-4B zero-shot results

With the typed-choice adapter, BF16 CUDA, the same scoped KB preview, and no fine-tuning:

| Set | Matched labels | Selected methods: BM25 / RAG / GraphRAG |
| --- | --- | --- |
| `router_cases.json` | 10/12 | 4 / 6 / 2 |
| `router_holdout.json` | 4/6 | 1 / 4 / 1 |
| `router_paths.json` | 6/8 | 1 / 3 / 4 |
| Combined | 20/26 | 6 / 13 / 7 |

The first two sets are unchanged from the Qwen evaluation: Kev matches 14/18 there,
versus the previous Qwen-7B configuration's 18/18. This is not evidence that Kev is a
more accurate zero-shot router on this workload. The typed interface provides scores
and avoids generated rationales; it does not replace task-specific evaluation.

`router_paths.json` adds short qualification-path questions plus exact-fact and local
mechanism counterexamples. Its labels reflect the intended full-path interpretation of
questions such as "how does a team advance to the super bowl". A narrower interpretation
can sometimes be answered by one passage; no algorithm uniquely owns these questions.
The current football PDF states the conference-champion end condition but lacks a full
playoff-seeding description, so selecting a graph route cannot supply missing evidence.

Kev selected GraphRAG for that exact Super Bowl question with scores of approximately
0.068 BM25, 0.401 RAG and 0.531 GraphRAG in the initial check. That closeness is useful
information, not a claim of verified accuracy. No confidence threshold or fine-tuning
was chosen from this small set. Expand independently labeled data before calibrating
or training the classifier. Generated detailed reports belong in ignored `data/evaluations/`.
