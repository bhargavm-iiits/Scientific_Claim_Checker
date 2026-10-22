# Latency benchmark - `cold_full_pipeline`

- Created: 2026-10-20T19:12:07+00:00; command: `python -m claim_checker latency --suite smoke --limit 10 --no-cache`
- 10 `smoke` claims (82, 182, 214, 246, 278, 292, 523, 550, 696, 734); full pipeline, hybrid + re-rank, evidence agent, k = 3; LLM disk cache **off**; model `qwen3:4b` (think=False, temperature 0.0); embeddings on `cpu`.
- Cache check: 0 LLM cache hits, 0 retrieval cache hits, 52 model calls; 0 errors.
- Warm-up (excluded below): resources 19.95 s + first claim 4.44 s = **24.40 s** per process.

| stage | n | mean s | median s | p95 s |
|---|---|---|---|---|
| validate_input | 10 | 0.00 | 0.00 | 0.00 |
| retrieve | 10 | 1.20 | 1.07 | 1.60 |
| retrieve.bm25 | 10 | 0.02 | 0.02 | 0.03 |
| retrieve.dense | 10 | 0.03 | 0.03 | 0.03 |
| retrieve.fuse | 10 | 0.00 | 0.00 | 0.00 |
| retrieve.rerank | 10 | 1.15 | 1.02 | 1.56 |
| select_evidence | 10 | 3.49 | 3.51 | 3.95 |
| judge | 10 | 3.22 | 3.49 | 5.43 |
| aggregate | 10 | 0.00 | 0.00 | 0.00 |
| supervise | 10 | 0.00 | 0.00 | 0.00 |
| explain | 10 | 0.00 | 0.00 | 0.00 |
| **total** | 10 | 7.92 | 8.42 | 10.07 |

GPU before the run: NVIDIA GeForce RTX 4050 Laptop GPU 0 % busy, 3944 / 6141 MiB

Per-claim statistics exclude warm-up. Stage timings are wall-clock seconds of each LangGraph node; retrieve.* are sub-steps of retrieve. The judge is skipped without a model call for an abstract with no selected evidence, so judge time varies with the evidence agent's output. Measured on one laptop with a local Ollama server; other GPU work at the same time slows the model stages.

Raw data: `artifacts/latency/cold_full_k3.json`.
