# Contributing

## Dev setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"          # installs pytest + ruff
cp .env.example .env             # set GITHUB_TOKEN, and an LLM key (see below)
```

## Running tests & lint

```bash
pytest -q          # unit tests (no network / no API keys needed)
ruff check .       # lint
ruff format .      # auto-format
```

CI (`.github/workflows/ci.yml`) runs the test suite on every push/PR. Lint is
currently advisory; once the codebase has had a `ruff check --fix && ruff format`
pass, flip `continue-on-error` off in the `lint` job to make it gate merges.

Tests live in `tests/`, use synthetic SQLite fixtures (`tests/helpers.py`,
`tests/conftest.py`), and never call the network or an LLM — LLM calls are
monkeypatched. Please add a test alongside any bug fix.

## Architecture

```
src/vllm_insights/
  db.py              # SQLite schema + idempotent migrations
  github.py          # GraphQL/REST client with rate-limit handling
  fetcher/           # incremental sync per entity (releases/PRs/commits/...)
  analysis/          # derived signals (perf claims, topics, benchmarks, ...)
  analyzer/          # pandas aggregations + PR↔release linking
  summarize.py       # LLM backends (bailian/anthropic/github) + digest prompts
  report.py          # assembles the weekly digest markdown (+ fallback cache)
  build_site.py      # static HTML dashboard
  cli.py             # Typer CLI (sync / analyze / digest / site / ...)
```

Data flows: **fetch → analyze → summarize → build/publish**. The weekly digest
is the product's core value; its reliability path (fingerprint cache + last-good
fallback in `report.py`) is intentionally defensive — keep it that way.

## LLM backends

The digest backend is auto-selected by env var (see `summarize._detect_backend`):
`DASHSCOPE_API_KEY` → bailian, else `ANTHROPIC_API_KEY` → anthropic, else
`GITHUB_TOKEN` → github. Override with `LLM_BACKEND` / `LLM_MODEL` / `LLM_TIMEOUT`.
