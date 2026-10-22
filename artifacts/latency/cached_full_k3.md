# Latency benchmark - `full_pipeline_with_llm_cache`

- Created: 2026-10-20T19:13:02+00:00; command: `python -m claim_checker latency --suite smoke --limit 10 --repeat`
- 10 `smoke` claims (82, 182, 214, 246, 278, 292, 523, 550, 696, 734); full pipeline, hybrid + re-rank, evidence agent, k = 3; LLM disk cache on; model `qwen3:4b` (think=False, temperature 0.0); embeddings on `cpu`.
- Cache check: 51 LLM cache hits, 0 retrieval cache hits, 0 model calls; 0 errors.
- Warm-up (excluded below): resources 30.14 s + first claim 0.90 s = **31.03 s** per process.

| stage | n | mean s | median s | p95 s |
|---|---|---|---|---|
| validate_input | 10 | 0.00 | 0.00 | 0.00 |
| retrieve | 10 | 1.24 | 1.12 | 1.66 |
| retrieve.bm25 | 10 | 0.03 | 0.03 | 0.03 |
| retrieve.dense | 10 | 0.03 | 0.03 | 0.03 |
| retrieve.fuse | 10 | 0.00 | 0.00 | 0.00 |
| retrieve.rerank | 10 | 1.19 | 1.06 | 1.60 |
| select_evidence | 10 | 0.00 | 0.00 | 0.00 |
| judge | 10 | 0.00 | 0.00 | 0.01 |
| aggregate | 10 | 0.00 | 0.00 | 0.00 |
| supervise | 10 | 0.00 | 0.00 | 0.00 |
| explain | 10 | 0.00 | 0.00 | 0.00 |
| **total** | 10 | 1.25 | 1.13 | 1.67 |

Repeat pass (same claims, same process): total mean 0.01 s, median 0.01 s, p95 0.01 s; 51 LLM cache hits, 0 model calls, 10 retrieval cache hits.

GPU before the run: NVIDIA GeForce RTX 4050 Laptop GPU 0 % busy, 3944 / 6141 MiB

Per-claim statistics exclude warm-up. Stage timings are wall-clock seconds of each LangGraph node; retrieve.* are sub-steps of retrieve. The judge is skipped without a model call for an abstract with no selected evidence, so judge time varies with the evidence agent's output. Measured on one laptop with a local Ollama server; other GPU work at the same time slows the model stages.

Raw data: `artifacts/latency/cached_full_k3.json`.
