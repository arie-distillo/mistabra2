# Counterpoint — PoC step 1

Ingest → semantic chunking → atomization → cached lift grid → validation matrix.

Terminology (fixed): **data points** are atomic corpus units; **hypotheses** are what
we validate against the corpus. Scoring is lift-based; this is a graded validation
engine, **not** an ACH tool (the bundled scenarios happen to carry ACH ground-truth,
which is used only as a test oracle and never fed to the scorer).

## Install & run

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[test]"            # add ".[llm]" for the real oracle, ".[embed]" for bge

# offline demo (deterministic mock LLM, hashing embeddings — no network, no key)
python demo.py --scenario 0
python demo.py --scenario 3 --noise 4 --duplicates 1

# real oracle + real semantic chunking
export OPENROUTER_API_KEY=sk-or-...
export COUNTERPOINT_EMBEDDING_BACKEND=sentence_transformers   # bge-base semantic embeddings
python demo.py --scenario 0

# web UI + API
uvicorn counterpoint.app:app --reload   # http://127.0.0.1:8000  ·  /docs
```

Scenario indices are alphabetical over `data/*.yaml` (0 = burnt_bread, …).

## What runs where

| concern | choice |
|---|---|
| chunking | semantic (sentence embeddings, split at similarity troughs) — `ingest.py` |
| embeddings | pluggable backend: `hashing` (offline/tests), `tfidf`, `sentence_transformers` (bge-base) — `embeddings.py` |
| atomization | LLM, one call per chunk, deduped, content-addressed — `atomize.py` |
| scoring | lift = P(D\|H)/P(D), multiplier scale, separate relatedness, neutral framing, cached — `scoring.py` |
| cache | SQLite, keyed `(hyp, dp, model)` and `(dp, model)` — `store.py` |
| matrix | pure arithmetic: score, verdicts, diagnosticity — `validation.py` |
| API | FastAPI — `app.py` |
| UI | FastHTML matrix + hypothesis detail — `app.py` |

## API (step-1 user stories)

```
POST /api/documents        {name, text, meta}           -> chunk + atomize
POST /api/episodes         {content, meta}              -> single pre-atomized unit
POST /api/hypotheses       {text, context_summary}
POST /api/load_scenario    {index, noise, duplicates}   -> bundled YAML + optional noise
POST /api/reset            {in_memory}                  -> clear corpus (dev/test)
POST /api/matrix                                        -> full scored matrix
GET  /api/hypotheses/{id}                               -> cited, ranked evidence
GET  /                                                  -> matrix UI
GET  /hyp/{id}                                          -> hypothesis detail UI
```

## Configuration & secrets

Copy `.env.example` to `.env` and fill it in. `.env` is gitignored; `config.py`
loads it via `python-dotenv` before reading any variable, so `OPENROUTER_API_KEY`
and friends are picked up automatically.

## Tests

Two clearly separated tiers, matching how and where they run:

### Unit tests — local only, offline, one command

Pure functions with no HTTP surface and no external services. Deterministic, fast,
never touch the network or an LLM. This is the everyday command:

```bash
python -m pytest
```

Covers: the lift-scale arithmetic (incl. the guard that "much_less" never inverts
above 1), verdict bands, robust JSON extraction (incl. the `None`-content regression
that once caused a retry storm), fault-injection contract tests for malformed model
output, and semantic chunking. ~38 tests, a few seconds, no key required.

### Acceptance tests — drive the system through its API, local or cloud

Black-box: they talk to a running system **only over HTTP**, so the *same* tests run
against a local server or a cloud deployment (Railway, etc.). Not pytest-based — a
standalone runner with rich step-by-step logging for troubleshooting.

```bash
# launch a throwaway local mock server, test it, tear it down (costs nothing)
python -m tests.acceptance.run --local

# against a running local server
python -m tests.acceptance.run --base-url http://127.0.0.1:8000

# against a cloud deployment
python -m tests.acceptance.run --base-url https://your-app.up.railway.app

# machine-readable summary for CI
python -m tests.acceptance.run --local --json
```

Every step logs the request (method, URL, body), the response (status, latency,
preview), and each assertion (`[PASS]`/`[FAIL]` with the actual value), so a failure
against a remote deployment is diagnosable from the log alone. Exit code 0 = all
passed, 1 = failure. Against a cloud target the deployment must expose `POST /api/reset`
(which uses a throwaway in-memory backend) — so point it only at test/staging, never
production.

### Evaluation — quality measurement (separate from tests)

Directional quality vs the YAML ground truth, across models. Not pass/fail; a report
tracked over time. Uses the real LLM.

```bash
python -m counterpoint.eval
python -m counterpoint.eval --models anthropic/claude-sonnet-4.5,openai/gpt-5
```

### Why the split

Unit tests must never cost money, hit the network, or be slow — so they never touch
the API or an LLM. Acceptance tests must verify the *deployed* system as a black box —
so they only ever use HTTP and work identically against local or cloud. Keeping these
apart is what prevents the failure mode where a "unit" run silently calls a real model
(which once turned a test run into a 59-minute, real-cost job).

## Honest limits of the offline demo

The bundled `MockLLM` is a lexical-overlap heuristic, not a reasoner. It exercises all
plumbing deterministically but produces flat/neutral verdicts and can leave sentence
fragments from atomization. The graded verdicts and clean atomization the design targets
require the real oracle (`OPENROUTER_API_KEY`) and, ideally,
`COUNTERPOINT_EMBEDDING_BACKEND=sentence_transformers`.
Everything the mock can't do well is an LLM-quality question, not a pipeline defect —
the tests assert the pipeline, the oracle supplies the intelligence.

## Configuration (environment variables)

| variable | default | meaning |
|---|---|---|
| `OPENROUTER_API_KEY` | *(empty → mock LLM)* | OpenRouter key; without it the offline `MockLLM` is used |
| `ORACLE_MODEL` | `anthropic/claude-sonnet-4.5` | model id used as the scoring oracle |
| `COUNTERPOINT_CACHE_DB_PATH` | `counterpoint.db` | SQLite file holding the corpus and the lift-score cache |
| `COUNTERPOINT_EMBEDDING_BACKEND` | `hashing` | embedding strategy: `hashing` / `tfidf` / `sentence_transformers` |
| `COUNTERPOINT_EMBEDDING_MODEL` | `BAAI/bge-base-en-v1.5` | model name, used only when the backend is `sentence_transformers` |

## Not in step 1 (later)

Conjunction / minimal-refuting-set extraction; residual detection + abduction;
incremental re-checking on new data points; the distilled scorer for corpus scale.
