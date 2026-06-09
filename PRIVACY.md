# Privacy & outbound-notification policy

This document records exactly what `vllm-insights` sends outside the machine it
runs on, and to whom. It is the result of an audit of every code path that could
notify a person or post to a third party.

## TL;DR

> Apart from a **private SMTP email of the weekly digest to your own `MAIL_TO`
> address**, the project makes **no outbound write that notifies any third
> party**. It never @-mentions anyone, never opens issues/PRs/comments on any
> GitHub repo, and never contacts upstream vLLM maintainers or PR authors.

## What goes out, and to whom

| Destination | Direction | What | Notifies a human? |
|-------------|-----------|------|-------------------|
| `api.github.com` (`github.py`) | **read-only** (`GET`) | releases, PRs, commits, issues, forks, trees, compares, workflow artifacts of `vllm-project/vllm` | No — never writes |
| `raw.githubusercontent.com` (`registry_sync.py`, `analysis/history.py`) | read | upstream `registry.py` / source files | No |
| Hacker News Algolia API (`analysis/social.py`) | read | public "vllm" mentions | No |
| LLM backend — Bailian/DashScope, GitHub Models, or Anthropic (`summarize.py`) | send | PR titles/bodies + release notes, to generate the digest | No (API call, not a notification) |
| Embedding backend — OpenAI or GitHub Models (`analysis/embeddings.py`) | send | PR/issue titles+bodies for clustering | No |
| **Your SMTP server → `MAIL_TO`** (`mailer.py`) | send | the weekly digest email | **Yes — to you only** |
| Your GitHub repo (`git push`) | write | `docs/weekly/*.md` to `main`; a gzipped DB snapshot to the `data` branch | No (pushes don't notify watchers; commit messages contain no `@`/`#`) |
| GitHub Pages | publish | the static dashboard | No |

> **Data-egress note:** the digest and embeddings send upstream PR/issue text to
> whichever LLM/embedding provider you configure (Alibaba DashScope, Anthropic,
> OpenAI, or GitHub Models). That is public GitHub data, but be aware it leaves
> your environment for those providers. Choose the backend via `LLM_BACKEND` /
> the relevant API-key secret.

## What the project deliberately does **not** do

- **No GitHub mutations.** `github.py` is a read-only client — every request is a
  `GET`. There is no code path that creates issues, comments, PRs, reviews,
  reactions, or workflow dispatches on `vllm-project/vllm` or any other repo. So
  vLLM maintainers and PR authors are never contacted.
- **No `@`-mentions / `#`-cross-references that ping.** PR authors in the digest
  are rendered as profile **links** (`<a href="https://github.com/handle">`), not
  bare `@handle`. The LLM is instructed never to emit `@`-mentions, and
  `report.neutralize_mentions()` defangs any stray `@handle` / `#1234` before any
  GitHub-bound content is posted. (Mentions inside committed *file content* do not
  notify anyone regardless; only commit messages / issue / PR / comment bodies do,
  and ours are clean.)
- **No newsletter / third-party signup.** The Buttondown subscribe form has been
  removed. The About page only offers an RSS Atom feed and a "watch the repo"
  suggestion — neither sends email.
- **No GitHub issue for alerts.** If the digest fails to generate, the workflow
  emails you privately when SMTP is configured, otherwise just logs a warning. It
  never opens an issue (which would surface to repo watchers).

## The private digest email (SMTP)

Delivery is a direct SMTP send to the address(es) in the `MAIL_TO` secret only —
no CC/BCC, no list, no public surface. See [README](README.md) for the required
secrets. TLS is verified (`ssl.create_default_context()`); the mailer refuses to
send credentials over an unencrypted connection unless `SMTP_ALLOW_INSECURE_AUTH=1`
is explicitly set.

## How this was verified

A multi-agent audit reviewed six vectors — the email path, GitHub issue/comment
writes, committed/published content, the upstream API client, git operations, and
LLM-echoed mentions — and each finding was independently re-checked by a skeptic
agent. The only paths that reach a human were confirmed to be (a) the private
SMTP email to you and (b) — before it was removed — the failure-alert issue and
the disabled Buttondown form. Both of the latter were then eliminated.

If you change the delivery mechanism or add any GitHub-writing step, re-run that
reasoning: the invariant to preserve is *no outbound write notifies anyone but
the operator.*
