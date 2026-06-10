# vllm-insights

Automated insights for [vllm-project/vllm](https://github.com/vllm-project/vllm):
release dynamics, PR flow, technical-area trends — distilled into a **weekly,
LLM-written technical digest** that explains what shipped and why it matters.

The pipeline runs on GitHub Actions once a week (sync → analyze → digest → build →
publish) and emails the teaching-oriented weekly digest to your mailbox via
private SMTP.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .            # add ".[dev]" for tests + lint
copy .env.example .env
# edit .env:
#   GITHUB_TOKEN       (public_repo scope is enough)
#   DASHSCOPE_API_KEY  (Alibaba Bailian; powers the LLM digest)
#   SMTP_HOST / SMTP_USERNAME / SMTP_PASSWORD / MAIL_TO  (private digest email)
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

- **`daily-sync.yml`** — runs weekly (Monday 08:00 UTC): sync → analyze → digest →
  build → publish to Pages, then emails the digest **privately via SMTP**
  (`vllm-insights email-digest`) straight to `MAIL_TO` — nothing is posted
  publicly and no repo watchers are notified. Change the day/time via the `cron`
  line. The history is backed up to a `data` branch and restored on cache miss.
  A failing digest alerts you privately (email if SMTP is set, otherwise just a
  workflow log warning — never a public issue).
- **`ci.yml`** — runs `pytest` on every push/PR; lint is advisory.

Outbound data and the guarantee that nothing notifies third parties are documented
in [PRIVACY.md](PRIVACY.md).

### Private digest email (SMTP)

Set these repo secrets (any provider). Example for QQ mail:

| Secret | Value |
|--------|-------|
| `SMTP_HOST` | `smtp.qq.com` |
| `SMTP_PORT` | `465` |
| `SMTP_USERNAME` | your full address |
| `SMTP_PASSWORD` | the SMTP **authorization code** (not the login password) |
| `MAIL_TO` | `subowen5@huawei.com` |

`SMTP_SECURITY` (`ssl`/`starttls`/`plain`) and `MAIL_FROM` are optional. Gmail/
Outlook use port `587` + `starttls` and an app password. If the secrets are
absent, the email step simply skips.

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
