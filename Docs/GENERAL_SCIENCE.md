# General science extension

This extension adds a separate evidence collection and reader profile. The
requested **accuracy >95% and three-class Macro-F1 >0.95 remain unmet**. No
absence-of-overfitting guarantee is made. SciFact results do not measure this mode.

## Evidence coverage

The local corpus contains **14,056 passages**: 14,055 extracted from 1,049
OpenStax textbook modules, plus one attributed USGS water-science excerpt.
Subjects include physics, chemistry, biology and astronomy. It is a fixed local
collection, not a search of every scientific subject or the live web.

Official OpenStax repositories are cached at pinned commits. Their raw source,
licenses and snapshot metadata are in `data/raw/general_science`. Each passage
retains its publisher, subject, book/module, source permalink, commit and checksum.
The prepared manifest and attribution file are in `data/processed/general_science`.
Physics has a CC BY 4.0 license; the other acquired repositories specify
CC BY-NC-SA 4.0. Preserve source licenses and attribution when redistributing.

Text extraction excludes exercises, answers, teacher material, navigation, figures
and tables. This also limits coverage of equations and facts expressed in diagrams.
Source whitespace is normalized; quotations are copied from the resulting stored
sentences. Exact quotation matching verifies traceability, not correct inference.

The complete provenance audit checked **91,207 stored sentences** from all 14,056
passages against 1,025 used cached source files/excerpts, with zero mismatches and
a matching corpus manifest. Some downloaded modules produce no retained passages.
Re-run it with `python -m scripts.audit_general_sources`; the report is
`artifacts/improvement/general_science_source_audit.json`.

The OpenStax extraction did not include the phrase "universal solvent". A verified
excerpt from the [USGS explanation](https://www.usgs.gov/water-science-school/science/water-universal-solvent)
provides evidence about that conventional name and its limitations. Direct HTTP
downloads returned empty responses and were rejected; this is **one verified
excerpt**, not a successful crawl of USGS articles. The acquisition route and text
are recorded in `data/sources/usgs_water_verified_excerpt.json` and the manifest.
"Water is a universal solvent" and "water dissolves everything" have different
meanings. Verdicts must follow the actual wording and the available evidence.

## Runtime and overfitting controls

`configs/general_science.yaml` uses a separate corpus, index, cache and run folder.
It retrieves and reranks passages, reads them with the existing generic local NLI
checkpoint, and returns SUPPORT, CONTRADICT or NEI with quotations and source links.
It uses full passages rather than the biomedical evidence-selection model.
This is an architectural alternative to the assignment's Ollama agent workflow.

The initial recipe was recorded before inference: top three sources, relevance
gate 0, support/contradiction thresholds 0.9, titles included. No new fitting or
threshold search on general-science demo labels was performed. Duplicate passages
across textbook editions are collapsed before reranking.

The earlier biomedical audit found late-epoch overfitting and excessive development
search. Its grouped TRAIN fitting/stopping/auditing protocol remains separate.
Only one of five grouped folds has completed; this does not establish that
overfitting is cured. See `OVERFITTING_AUDIT_2026-10-22.md`.

## Run on this computer (Anaconda Prompt / Command Prompt)

```bat
cd /d "D:\Additional Project\Gen_AI"
set "TEST_PYTHON=C:\Users\Bhargav\anaconda3\python.exe"
set "PYTHONPATH=D:\Additional Project\Gen_AI\tmp\audit_deps"
set "HF_HUB_OFFLINE=1"
"%TEST_PYTHON%" -m claim_checker --config configs/general_science.yaml check --claim "Water is a Universal Solvent"
```

Launch directly into the new UI mode:

```bat
set "CLAIM_CHECKER_CONFIG=configs/general_science.yaml"
"%TEST_PYTHON%" -m streamlit run ui/app.py --server.address 127.0.0.1
```

The sidebar also lets you switch collections. The general-science profile uses
local classifiers and does not require Ollama for claim checking. CPU inference
is available; the first check loads the models and takes longer.

## Evaluation

The first fixed GPU smoke run (`general_smoke_v1`) completed with **14/15 correct,
93.33% accuracy and 0.932660 Macro-F1**, zero operational errors and **46/46
source-exact quotations**. The Wilson accuracy interval is approximately
70.18%–98.81%. Both numerical targets remain unmet even on this small demo set.
The incorrect case, "The Moon is a star", returned NEI because the reader produced
conflicting document decisions. One quotation associated with SUPPORT describes
the Moon's apparent motion; its presence in the source does not establish that it
supports the claim. This illustrates the difference between exact quotation
traceability and inference correctness. Settings were not tuned after this result.

Separate actual CPU checks (`artifacts/improvement/general_water_cpu_checks.json`)
returned SUPPORT for the exact user wording "Water is a Universal Solvent",
CONTRADICT for "Water dissolves every substance", and SUPPORT for "Water does not
dissolve every substance". All checks completed normally with source quotations.
These are wording diagnostics, not a second accuracy benchmark.

The 15 cases in `data/examples/general_science_smoke_v1.jsonl` are developer-written
smoke examples, with five examples per class. They are not an independently
reviewed or representative benchmark. Even 15/15 would give a Wilson 95% accuracy
interval with a lower bound of only about 79.6%.

Run their fixed evaluation on CPU:

```bat
"%TEST_PYTHON%" scripts/evaluate_general_science.py --claims data/examples/general_science_smoke_v1.jsonl --benchmark-metadata data/examples/general_science_smoke_v1.metadata.json --run-id general_smoke_v1_cpu --purpose smoke
```

On this computer the optional GPU dependencies can be selected with:

```bat
set "PYTHONPATH=D:\Additional Project\Gen_AI\tmp\gpu_deps;D:\Additional Project\Gen_AI\tmp\audit_deps"
"%TEST_PYTHON%" scripts/evaluate_general_science.py --claims data/examples/general_science_smoke_v1.jsonl --benchmark-metadata data/examples/general_science_smoke_v1.metadata.json --run-id general_smoke_v1 --purpose smoke --device cuda
```

Results go to `artifacts/runs/general_science/<run-id>`. The evaluator freezes
benchmark, configuration, source/index, weight and inference-code identities,
rejects changed resumes, and records benchmark reuse. Labels do not enter
`ClaimChecker.check`. All three classes are included; operational errors count as
incorrect. It also reports source-exact quotation validity and an accuracy interval.
Holdout evaluation requires documented source, label review and limitations, but
the script cannot certify independence or absence of pretraining contamination.

Verification: all **350 regression tests** pass, including collection switching,
source parsing/citations, benchmark isolation and the existing biomedical pipeline;
Ruff passes. These checks establish software behavior, not scientific accuracy.

Reaching the requested targets requires a sufficiently large, independently
labeled, untouched general-science benchmark, verified training separation and
successful end-to-end evaluation. Adding more passages or passing curated demos
does not establish either target. Out-of-scope and poorly retrieved claims can
still return NEI or receive incorrect verdicts.

## Rebuild from the cached sources

```bat
"%TEST_PYTHON%" scripts/prepare_general_science.py
"%TEST_PYTHON%" -m scripts.prepare_usgs_science --verified-excerpts
"%TEST_PYTHON%" -m claim_checker --config configs/general_science.yaml build-index
```

These steps do not retrain the reader. The first step rebuilds textbooks only;
run the second step to append the recorded USGS excerpt before building the index.
