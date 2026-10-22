# Runbook

How to install, prepare, index, launch, evaluate and troubleshoot the Scientific Claim Checker without
reading the implementation. Commands are for Windows PowerShell from the project folder
`D:\Additional Project\Gen_AI`.

```powershell
cd "D:\Additional Project\Gen_AI"
$py = "C:\Users\Bhargav\anaconda3\python.exe"     # conda base, Python 3.14.6
```

Always call the conda interpreter by its full path. On this machine a different Python
(`C:\Python314\python.exe`, 3.14.7) is first on `PATH`, so a bare `python` does **not** run the
environment in which the project is installed.

## 1. Install (once, needs internet)

| Step | Command | Expected result |
|---|---|---|
| Install package + dev tools | `& $py -m pip install -e ".[dev]"` | Editable install of `claim-checker` 0.1.0 with pytest, ruff, Jupyter, nbconvert, nbclient, ipykernel |
| Install Ollama | download from https://ollama.com/download and start it (tray app or `ollama serve`) | `http://127.0.0.1:11434/api/version` answers (tested with 0.40.6) |
| Download the model | `ollama pull qwen3:4b` | About 2.5 GB, Q4_K_M, digest `359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7` |
| Check everything | `& $py -m claim_checker doctor` | `[OK]` lines for Python, imports, writable folders, Ollama server, model and a structured-output test; `[WARN]` for data/index until steps 2–3 are done |

Optional: `& $py -m pip install -r requirements.lock.txt` pins the exact versions that were verified
(datasets 5.1.0, torch 2.14.1, sentence-transformers 6.1.0, langgraph 1.2.16, ollama 0.6.3, …).
`doctor --json` gives machine-readable output; `doctor --skip-model-test` skips the generation test.

## 2. Prepare data (once, needs internet)

```powershell
& $py -m claim_checker prepare-data
```

What happens (about 1–2 minutes):

1. `load_dataset("mteb/scifact", "corpus")` → 5,183 abstracts.
2. `load_dataset("allenai/scifact_entailment")` is attempted; with datasets 5.1.0 it fails with
   `RuntimeError: Dataset scripts are no longer supported, but found scifact_entailment.py`. This is
   expected and recorded.
3. Fallback: the builder script is downloaded as text (never executed) to read the official archive
   URL; `data.tar.gz` is downloaded to `data/raw/scifact/` and read with `load_dataset("json", ...)`.
4. IDs, titles, texts and sentence positions are validated; processed records go to
   `data/processed/`; the data-quality report is `data/processed/data_quality.json`; the route and all
   hashes are in `data/raw/source_manifest.json`.
5. Splits are frozen in `data/splits/` (`final.json` 100 claims, `development.json` 808,
   `smoke.json` 20).

Re-running is safe: the cached archive is reused and unchanged splits report `[unchanged]`.
`--force-download` re-downloads the archive. `--force-splits` overwrites a frozen final split — only
for a new, separately reported experiment.

## 3. Build the index (once per corpus/config change)

```powershell
& $py -m claim_checker build-index
```

About 8 minutes on CPU (window construction ~1.5 min, encoding ~6.5 min). Produces
`artifacts/index/` with 5,670 windows for 5,183 documents (455 documents need more than one window)
and fingerprint `3fa9e53a7857605c`. The first run downloads `BAAI/bge-small-en-v1.5` into the Hugging
Face cache.

## 4. Check claims

```powershell
& $py -m claim_checker check --claim "A deficiency of folate increases blood levels of homocysteine."
& $py -m claim_checker check --claim "..." --json              # full ClaimResult JSON
& $py -m claim_checker check --claim "..." --k 5 --no-rerank   # ablations
& $py -m claim_checker check --claim "..." --evidence-mode full_abstract
& $py -m claim_checker check --claim "..." --variant judge_only   # any variant in configs/experiments.yaml
& $py -m claim_checker check --claim "..." --variant oracle --oracle-doc-ids 13734012   # oracle needs ids
```

The `oracle` variant is an evaluation upper bound: during `evaluate` the runner supplies each claim's
gold reference abstracts. From the CLI it therefore requires `--oracle-doc-ids` (one or more corpus
document ids); without ids it fails with `invalid_input`, and an id that is not in the corpus fails
with `data_integrity` — neither is ever reported as NEI. `--oracle-doc-ids` is refused for
non-oracle variants.

Exit code 0 = result produced, 2 = operational error (message and hint printed). The first check in a
process takes about 20–30 s longer because models are loaded (measured in `artifacts/latency/`); the
re-ranker is downloaded on its first use.

## 5. Launch the Streamlit app

```powershell
& $py -m streamlit run ui/app.py --server.address 127.0.0.1
```

Open `http://127.0.0.1:8501`. The UI is **ClaimLens**, a dark "research console" theme
(`.streamlit/config.toml` + `ui/assets/theme.css`; fonts fall back to Segoe UI / Consolas offline).
The header shows status pills (Ollama, model, index, data: green ready, amber/red not ready). The
sidebar holds the settings (k, retrieval method, cross-encoder re-rank, evidence agent), the system
checklist with **Re-check** and **Load models** (warms everything up before a demo, about 20–30 s;
see `artifacts/latency/`), and the session history (click a card to re-display a stored result; it
is not re-run). Tabs:

- **Check a claim** — claim box with a live character counter (Ctrl+Enter checks), example chips
  (one SUPPORT, one CONTRADICT, one NEI claim from SciFact validation; **More** opens training-split
  examples with their reference labels, for comparison only). While a check runs, a live pipeline
  board shows each stage as waiting / running / done / failed (with a warm-up note on the first
  check of a server session). The result shows a verdict card (Supported / Contradicted / Not enough
  information, icon + text; the confidence bar appears only when the model reported one and is
  labelled model-reported, uncalibrated), a conflict banner when abstracts disagree, evidence quote
  cards (`doc <id> · S<n>`, one-based sentence numbers), the explanation, a pipeline timeline with
  per-stage and retrieval sub-step durations, run details, source cards with scores and the full
  abstract (cited sentences highlighted), JSON download and the raw JSON record. Operational errors
  are a separate "Check failed – no verdict" card with error type, stage, message and fix. Blank or
  over-long claims (2,000 characters / 256 BGE tokens) are rejected inline before the index,
  embedding model or Ollama are touched; the token check loads only the tokenizer (~0.2 s).
- **How it works** — plain-language pipeline, verdict meanings, aggregation rule, NEI vs. error.
- **Evaluation results** — saved runs from `artifacts/runs/` and the retrieval comparison from
  `artifacts/retrieval/`; nothing on this tab runs the model.

Reference screenshots: `artifacts/ui_screenshots/`. Stop with Ctrl+C. The server binds to localhost
only.

## 6. Offline operation

After steps 1–3 and one successful check (so that the cross-encoder is cached too), the checker runs
without internet:

- Embedding and re-ranking models load from the local Hugging Face cache first
  (`local_files_only=True`) and only fall back to downloading if they are missing.
- Data and index are local files; Ollama is local.
- For extra safety set `$env:HF_HUB_OFFLINE = "1"` before starting the CLI, UI or notebook.
- `prepare-data` itself needs internet (it records dataset revisions and downloads); do not run it
  offline.
- Ollama must be running even when every answer is in the LLM cache, because cache keys include the
  model digest.

## 7. Evaluate and resume

```powershell
& $py -m claim_checker retrieval-eval --split final                       # a few minutes, CPU
& $py -m claim_checker evaluate --suite smoke --run-id smoke_v1           # full + oracle, 20 claims
& $py -m claim_checker evaluate --suite final --run-id final_v1           # 8 variants x 100 claims
& $py -m claim_checker evaluate --suite final --run-id final_v1 --variants full oracle   # subset
& $py -m claim_checker evaluate --suite final --run-id try1 --limit 5     # quick partial check
& $py -m claim_checker report --run-id final_v1                           # recompute metrics/report
& $py -m claim_checker latency --suite smoke --limit 10 --no-cache        # cold latency, ~2.5 min
& $py -m claim_checker latency --suite smoke --limit 10 --repeat          # with LLM cache + repeat pass
```

`latency` runs the full k = 3 pipeline on the first 10 smoke (train) claims in a fresh process,
measures the warm-up (resource loading plus one unmeasured first claim) separately, and writes
per-stage mean / median / p95 to `artifacts/latency/cold_full_k3.json` and `.md` (with `--no-cache`
the LLM disk cache is disabled; the artifact verifies 0 cache hits). It also records `nvidia-smi`
and Ollama `/api/ps` snapshots, so close other GPU work first.

- Progress is saved after every claim. **To resume, run exactly the same command again**; completed
  `(variant, claim)` pairs are skipped. Before continuing, the run removes diagnostic rows
  (`retrieval.jsonl`, `pair_predictions.jsonl`, `events.jsonl`) left by a claim that was interrupted
  before its prediction was saved, and logs how many it removed; nothing is rewritten if the files are
  already consistent.
- A run resumes only if data, splits, index, model digest, prompt hashes, configuration and variant
  definitions are unchanged. Otherwise you get `ERROR (run_mismatch)`; start a new `--run-id`.
- If Ollama stops during a run, the command stops with `model_unavailable` and keeps completed claims;
  restart Ollama and rerun.
- The full final suite takes hours on a laptop (8 variants × 100 claims, about 8 s per claim for a
  cold k = 3 pipeline call, less when calls are cached; see `artifacts/latency/cold_full_k3.json`).
  Run it in a terminal you can leave open, for example with output redirected to
  `artifacts/logs/final_v1.log`.
- Run ids name a folder directly inside `artifacts/runs/`: letters, digits, `.`, `_` and `-` only
  (`..`, path separators and absolute paths are refused with `invalid_input`).
- After a run, write `artifacts/runs/<run_id>/failure_analysis.md` by hand from `failure_cases.csv`.

## 8. Notebook

```powershell
& $py -m jupyter lab notebooks/Scientific_Claim_Checker.ipynb
# or headless, from a clean kernel:
& $py -m jupyter nbconvert --to notebook --execute notebooks/Scientific_Claim_Checker.ipynb `
    --output-dir artifacts/notebooks --output Scientific_Claim_Checker.executed.ipynb `
    --ExecutePreprocessor.timeout=1800
```

The notebook uses the `python3` kernel. Its kernelspec launches `python`; Jupyter substitutes the
interpreter that runs Jupyter itself, so start Jupyter with the conda interpreter as shown. The first
code cell prints `sys.executable` and stops with a clear message if `claim_checker` is not importable.
Heavy steps read saved artifacts; only one live retrieval demo and one live claim check run, so a clean
run takes a few minutes (Ollama must be running for the live check; otherwise that cell prints a
setup message and continues).

## 9. Tests and lint

```powershell
& $py -m pytest -q      # offline: synthetic fixtures and a mocked Ollama; no downloads
& $py -m ruff check .
```

CI (`.github/workflows/ci.yml`) runs the same on Python 3.11 with CPU-only torch and
`HF_HUB_OFFLINE=1`. Its dependencies are installed unpinned (not from `requirements.lock.txt`), and
the workflow has not been run or verified locally: the suite was verified on Python 3.14.6 (conda
base) only.

## 10. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ERROR (model_unavailable): Cannot reach Ollama at http://127.0.0.1:11434` | Ollama not running | Start the Ollama app or `ollama serve`; check with `curl http://127.0.0.1:11434/api/version`; rerun `doctor` |
| `Model 'qwen3:4b' is not downloaded in Ollama` | Model missing, or a different tag installed | `ollama pull qwen3:4b`; `ollama list` should show it. Names are matched the same way by `doctor`/readiness and the client: an untagged name means `:latest` (`qwen3` = `qwen3:latest`), other tags must match exactly (`qwen3:8b` does not satisfy `qwen3:4b`) |
| `model.host must be a local loopback endpoint …` when loading the configuration | `model.host` in `configs/default.yaml` is not an `http`/`https` URL with hostname exactly `localhost`, `127.0.0.1` or `[::1]`, or contains credentials | Use `http://127.0.0.1:11434` (or another loopback port). Remote or cloud endpoints are deliberately refused |
| `ERROR (stale_index): Index at … is stale: corpus_sha256 changed …` | Corpus or retrieval settings (embedding model, window size, query instruction, tokenizer) changed after the index was built | `& $py -m claim_checker build-index` |
| `ERROR (index_not_ready): Index files missing` | Index never built or deleted | Run `build-index` (after `prepare-data`) |
| `ERROR (data_not_ready): Prepared data not found` | `prepare-data` not run | Run `prepare-data` (needs internet once) |
| `claim-checker` is not recognized | `C:\Users\Bhargav\anaconda3\Scripts` is not on `PATH` | Use `& $py -m claim_checker …`, or add that folder to `PATH` |
| `ModuleNotFoundError: claim_checker` | Wrong interpreter (`python` on PATH is `C:\Python314`) or package not installed | Use `$py`; reinstall with `& $py -m pip install -e ".[dev]"` |
| `ERROR (context_overflow): … needs about N tokens but only 7424 fit` | Prompt larger than `num_ctx − num_predict` (8192 − 768); possible for the full-abstract judge or the single-prompt baseline with long abstracts | Use the evidence agent (default), a smaller `--k`, or raise `model.context_tokens` in `configs/default.yaml` (more VRAM). The system refuses rather than letting Ollama silently truncate the prompt. |
| `ERROR (model_timeout)` | GPU/CPU overloaded (e.g. an evaluation running) | Wait for the other job; timeout is 180 s per request (`model.request_timeout_seconds`) |
| `ERROR (model_invalid_output)` | Neither the first answer nor the single repair was schema-valid JSON (if the first answer was valid and only the repair was not, the first answer is kept and a `repair_fallback` warning event is recorded instead) | Retry; inspect the prompt; it is reported as an operational error, never as NEI |
| `ERROR (run_mismatch)` when resuming | Something in the run identity changed (prompt, config, model digest, index, split) | Use a new `--run-id`; do not mix results |
| `ERROR (frozen_split)` from `prepare-data` | The final split would change | Investigate why (different source data?). `--force-splits` only for a new, separately reported experiment |
| First check is slow (15–20 s extra) | Lazy loading of corpus, BM25, BGE, cross-encoder | Expected; use the UI's **Load models** or `ClaimChecker().warmup()` before a demo |
| Model answers are slow on CPU | Ollama could not use the GPU | Check `ollama ps`; close other GPU programs; the CPU fallback keeps the same model but is much slower |
| Hugging Face symlink warnings on Windows | No developer mode | Harmless; the package sets `HF_HUB_DISABLE_SYMLINKS_WARNING=1` |
| `pip` reports conflicts after installing into conda base | Installing into base upgraded shared packages: `spyder 6.1.5` and `python-lsp-server 1.14.0` require `jedi<0.20.0` (now 0.20.0); `spyder-kernels 3.1.5` requires `ipykernel<7` (now 7.3.0); `numba 0.65.1` requires `numpy<2.5` (now 2.5.2); `s3fs` pins `fsspec` | Does not affect this project. If you need Spyder/numba, reinstall them with `conda install spyder numba` or, better, install this project into a separate environment (`conda create -n claimcheck python=3.11` then `pip install -e ".[dev]"`) |
| Streamlit shows "not ready" after fixing something | Readiness is cached for 60 s per session | Press **Re-check** in the sidebar |
| Results differ from a saved run | Prompts, model build or config changed | Compare `prompt_hashes`, `model_digest` and `index_fingerprint` in the result metadata or run manifest |

## 11. Resetting

| To reset | Delete | Then |
|---|---|---|
| LLM answers (force fresh generations) | `artifacts/cache/llm/` | — |
| Index | `artifacts/index/` | `build-index` |
| Prepared data (not the frozen splits) | `data/processed/` | `prepare-data` (splits stay frozen if identical) |
| A run | `artifacts/runs/<run_id>/` | `evaluate … --run-id <run_id>` |
