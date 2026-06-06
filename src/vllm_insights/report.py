"""Weekly themed digest.

We used to emit one markdown file per day. That's noise — most days the only
thing in a vLLM repo is 20 incremental PRs none of which deserve a reader's
attention. We now emit a single themed weekly digest, written under
`docs/weekly/`:

  - The LLM section is theme-sliced (Kernels & attention / Quantization /
    Parallelism & scheduling / Model support / Hardware / API & serving /
    Watch list) so each section answers a specific operator question.
  - The "stats" tables (PR-tech bar chart, top committers, monthly merge time)
    used to live here but they duplicate the homepage charts; they're gone.

The legacy `generate_daily_report` name is kept as an alias because the GH
Actions workflow still calls it during the rollout.
"""
from datetime import datetime, timedelta, timezone
from html import escape
from pathlib import Path

import pandas as pd

from .analyzer.queries import releases_df, prs_df
from .db import connect


def _load_cached_digest(db_path: Path, key: str = "weekly") -> tuple[str, str] | None:
    """Return (content, generated_at) of the last good digest, or None."""
    try:
        with connect(db_path) as conn:
            row = conn.execute(
                "SELECT content, generated_at FROM digest_cache WHERE key = ?",
                (key,),
            ).fetchone()
    except Exception:
        return None
    if row and (row["content"] or "").strip():
        return row["content"], row["generated_at"]
    return None


def _store_cached_digest(db_path: Path, content: str, key: str = "weekly") -> None:
    """Persist a freshly generated good digest for future fallback."""
    try:
        with connect(db_path) as conn:
            conn.execute(
                """INSERT INTO digest_cache(key, content, generated_at)
                   VALUES(?, ?, ?)
                   ON CONFLICT(key) DO UPDATE SET
                     content = excluded.content,
                     generated_at = excluded.generated_at""",
                (key, content, datetime.now(timezone.utc).isoformat()),
            )
    except Exception:
        # Caching is best-effort; never let it break digest generation.
        pass


def generate_weekly_digest(
    db_path: Path,
    days: int = 7,
    repo: str = "vllm-project/vllm",
    include_llm: bool = True,
    llm_backend: str | None = None,
    llm_model: str | None = None,
) -> str:
    """Produce the weekly themed digest as markdown."""
    now = datetime.now(timezone.utc)
    since = pd.Timestamp(now - timedelta(days=days))   # tz-aware

    iso_year, iso_week, _ = now.isocalendar()
    lines: list[str] = [
        f"# vLLM weekly digest — {now:%Y-%m-%d} (W{iso_week:02d})",
        "",
        f"_Window: last {days} days · upstream: [{repo}](https://github.com/{repo})_",
        "",
    ]

    # ----- LLM themed digest at the top (the actual content) -----
    if include_llm:
        from .summarize import summarize_window
        try:
            digest = summarize_window(
                db_path, days=days, model=llm_model,
                backend=llm_backend, repo=repo, include_header=False,
            ).strip()
            lines += [digest, ""]
            # Remember this good generation so a future failure can fall back.
            _store_cached_digest(db_path, digest)
        except Exception as e:
            # The LLM section IS the core value of this page. Rather than publish
            # a digest whose insight section is just an error line, fall back to
            # the most recent successful digest if we have one cached.
            cached = _load_cached_digest(db_path)
            if cached:
                content, gen_at = cached
                stamp = gen_at[:10]
                lines += [
                    f"> ℹ️ _Live summary unavailable ({type(e).__name__}); "
                    f"showing the last successful digest from {stamp}._",
                    "",
                    content.strip(),
                    "",
                ]
            else:
                lines += [
                    f"_LLM digest skipped: {type(e).__name__}: {e}_",
                    "",
                ]

    # ----- Releases that landed in the window -----
    rel = releases_df(db_path)
    if not rel.empty:
        recent_rel = rel[rel["published_at"] >= since]
        if not recent_rel.empty:
            lines += [f"## Releases this window", ""]
            for _, r in recent_rel.iterrows():
                lines += [
                    f"- [`{r['tag']}`]({r['url']}) — {r['published_at']:%Y-%m-%d %H:%M UTC}"
                ]
            lines += [""]

    # ----- Top merged PRs in the window (terse — the LLM digest carries the narrative) -----
    prs = prs_df(db_path)
    if not prs.empty:
        merged = prs.dropna(subset=["merged_at"])
        recent = merged[merged["merged_at"] >= since].sort_values("merged_at", ascending=False)
        if not recent.empty:
            # Render the raw list as explicit HTML inside the <details> block.
            # Markdown bullets placed inside a raw-HTML <details> are NOT reparsed
            # by many renderers (GitHub issue emails, python-markdown without the
            # md_in_html extension), which collapses every line into one unreadable
            # blob. Emitting <ul><li> renders correctly everywhere.
            lines += [
                f"## PRs merged this window ({len(recent)})",
                "",
                "<details>",
                "<summary>Click to expand the raw list</summary>",
                "",
                "<ul>",
            ]
            for _, p in recent.head(60).iterrows():
                rt = p.get("release_tag")
                rel_tag = (
                    f" → <code>{escape(str(rt))}</code>"
                    if pd.notna(rt) and rt else ""
                )
                title = escape(str(p["title"]))
                author = escape(str(p["author"]))
                lines.append(
                    f'<li><a href="{escape(str(p["url"]))}">#{p["number"]}</a> '
                    f"{title} — @{author}{rel_tag}</li>"
                )
            if len(recent) > 60:
                lines.append(f"<li><em>…and {len(recent) - 60} more</em></li>")
            lines += ["</ul>", "</details>", ""]

    return "\n".join(lines)


# Backwards-compat alias — the workflow currently imports this name.
def generate_daily_report(*args, **kwargs):  # pragma: no cover - thin shim
    """Deprecated alias for `generate_weekly_digest`. Kept so the workflow keeps
    working through the rollout. Drop after the workflow is updated."""
    # Old kwargs: include_llm, llm_days, llm_backend, llm_model
    kwargs.pop("llm_days", None)
    return generate_weekly_digest(*args, **kwargs)
