# vllm-insights

Automated insights for [vllm-project/vllm](https://github.com/vllm-project/vllm):
release dynamics, PR flow, technical-area trends — distilled into a **weekly,
LLM-written technical digest** that explains what shipped and why it matters.

The pipeline runs on GitHub Actions every 3 hours (sync + publish) and emails a
teaching-oriented weekly digest once per day.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .            # add ".[dev]" for tests + lint
copy .env.example .env
# edit .env:
#   GITHUB_TOKEN       (public_repo scope is enough)
#   DASHSCOPE_API_KEY  (Alibaba Bailian; powers the LLM digest)
```

## Usage

```powershell
# Incremental sync from GitHub into local SQLite (first run takes a while)
vllm-insights sync --all

# Derived analyses (perf claims, PR↔issue links, embeddings, clustering)
vllm-insights analyze --all

# Weekly themed digest (writes docs/weekly/latest.md)
vllm-insights digest --days 7 --llm

# Build the static dashboard, or launch the Streamlit app
vllm-insights site --docs docs
vllm-insights dash
```

## The weekly digest

`vllm-insights digest` produces a theme-sliced, **teaching-oriented** digest:

- **TL;DR** — the shape of the week in a few sentences.
- **Deep dives** — the 2–4 most significant updates, each explained (concept →
  what changed → why it matters → who's affected).
- **Theme sections** (Kernels & attention, Quantization, Parallelism, Model
  support, Hardware, API & serving, Watch list) — scannable bullets.
- A collapsed raw list of every merged PR.

Reliability: each successful generation is cached with a fingerprint of the
windowed data. If the data hasn't changed, the LLM call is skipped; if a live
call fails, the digest falls back to the last successful one instead of
publishing an error. Configure via `LLM_BACKEND` / `LLM_MODEL` / `LLM_TIMEOUT`.

## LLM backends

Auto-selected by env var (`summarize._detect_backend`):

| Env var | Backend | Default model |
|---|---|---|
| `DASHSCOPE_API_KEY` | bailian (Alibaba Bailian / DashScope, OpenAI-compatible) | `qwen3.7-max` |
| `ANTHROPIC_API_KEY` | anthropic | `claude-haiku-4-5` |
| `GITHUB_TOKEN` | github (GitHub Models) | `openai/gpt-4o-mini` |

## Automation (GitHub Actions)

- **`daily-sync.yml`** — every 3h: sync → analyze → digest → build → publish to
  Pages. Once per UTC day it emails the digest by creating/commenting on a
  per-ISO-week GitHub issue assigned to the repo owner (no SMTP needed — set your
  GitHub notification email to receive it). The SQLite history is backed up to a
  `data` branch and restored on cache miss. A failing digest opens an alert issue.
- **`ci.yml`** — runs `pytest` on every push/PR; lint is advisory.

Required repo secret: `DASHSCOPE_API_KEY`. Optional: `OPENAI_API_KEY`
(embeddings), `ANTHROPIC_API_KEY` (cluster labels).

## Layout

```
src/vllm_insights/
  db.py              # SQLite schema + helpers
  github.py          # GraphQL/REST client with rate-limit handling
  fetcher/           # incremental sync per entity
  analysis/          # derived signals (perf claims, topics, benchmarks, ...)
  analyzer/          # pandas aggregations + PR↔release linking
  summarize.py       # LLM backends + digest prompts
  report.py          # weekly digest assembly (fingerprint cache + fallback)
  build_site.py      # static HTML dashboard
  cli.py             # Typer CLI
tests/               # pytest suite (synthetic SQLite, no network)
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for dev workflow. Licensed under
[MIT](LICENSE).
