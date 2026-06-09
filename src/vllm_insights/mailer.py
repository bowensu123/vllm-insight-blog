"""Private SMTP delivery of the weekly digest.

This replaces the GitHub-issue notification mechanism with a direct SMTP send to
your own mailbox — nothing is posted publicly, no repo watchers are notified, and
@mentions / #refs in the body are just inert text in an email (they can't ping
anyone). Provider-agnostic: configure entirely via env vars / GitHub secrets.

Required:
  SMTP_HOST        e.g. smtp.qq.com / smtp.163.com / smtp.gmail.com
  SMTP_USERNAME    the login (usually the full sending address)
  SMTP_PASSWORD    the SMTP password / app-password / authorization code
  MAIL_TO          recipient(s), comma-separated

Optional:
  SMTP_PORT        default 465
  SMTP_SECURITY    'ssl' (default for 465) | 'starttls' (default for 587) | 'plain'
  MAIL_FROM        default = SMTP_USERNAME
  SMTP_TIMEOUT     seconds, default 60
"""
from __future__ import annotations

import os
import smtplib
import ssl
from email.message import EmailMessage

import markdown as _md


class MailNotConfigured(RuntimeError):
    """Raised when required SMTP settings are absent — caller should skip, not fail."""


def _resolve_config() -> dict:
    host = os.getenv("SMTP_HOST", "").strip()
    user = os.getenv("SMTP_USERNAME", "").strip()
    password = os.getenv("SMTP_PASSWORD", "").strip()
    to_raw = os.getenv("MAIL_TO", "").strip()

    missing = [
        name for name, val in (
            ("SMTP_HOST", host),
            ("SMTP_USERNAME", user),
            ("SMTP_PASSWORD", password),
            ("MAIL_TO", to_raw),
        ) if not val
    ]
    if missing:
        raise MailNotConfigured("missing SMTP settings: " + ", ".join(missing))

    port = int(os.getenv("SMTP_PORT", "465") or "465")
    security = (os.getenv("SMTP_SECURITY", "").strip().lower()
                or ("starttls" if port == 587 else "ssl"))
    if security not in ("ssl", "starttls", "plain"):
        raise ValueError(f"SMTP_SECURITY must be ssl|starttls|plain, got {security!r}")

    recipients = [a.strip() for a in to_raw.split(",") if a.strip()]
    if not recipients:
        # e.g. MAIL_TO=", ," — treat as not-configured (skip), not a send error.
        raise MailNotConfigured("MAIL_TO has no valid addresses")
    return {
        "host": host,
        "port": port,
        "user": user,
        "password": password,
        "sender": os.getenv("MAIL_FROM", "").strip() or user,
        "recipients": recipients,
        "security": security,
        "timeout": float(os.getenv("SMTP_TIMEOUT", "60") or "60"),
        "allow_insecure_auth": os.getenv("SMTP_ALLOW_INSECURE_AUTH", "").strip()
        not in ("", "0", "false", "no"),
    }


_EMAIL_CSS = (
    "body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;"
    "line-height:1.55;color:#1f2328;max-width:760px;margin:0 auto;padding:16px}"
    "h1{font-size:1.5em}h2{font-size:1.25em;border-bottom:1px solid #d0d7de;padding-bottom:.3em}"
    "h3{font-size:1.05em}code{background:#eff1f3;padding:.1em .3em;border-radius:4px}"
    "a{color:#0969da}ul{padding-left:1.3em}blockquote{color:#57606a;border-left:3px solid #d0d7de;"
    "margin:0;padding-left:1em}"
)


def build_message(subject: str, markdown_text: str, sender: str,
                  recipients: list[str]) -> EmailMessage:
    """Build a multipart text+HTML email from the digest markdown."""
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    # Plain-text fallback = the raw markdown (still readable).
    msg.set_content(markdown_text)
    html_body = _md.markdown(
        markdown_text, extensions=["extra", "sane_lists", "nl2br"]
    )
    msg.add_alternative(
        f"<!doctype html><html><head><meta charset='utf-8'>"
        f"<style>{_EMAIL_CSS}</style></head><body>{html_body}</body></html>",
        subtype="html",
    )
    return msg


def send_digest(markdown_text: str, subject: str) -> list[str]:
    """Send the digest via SMTP. Returns the recipient list on success.

    Raises MailNotConfigured if settings are absent (caller should skip), or the
    underlying smtplib/ssl error if the configured send fails.
    """
    cfg = _resolve_config()
    msg = build_message(subject, markdown_text, cfg["sender"], cfg["recipients"])
    context = ssl.create_default_context()

    if cfg["security"] == "ssl":
        with smtplib.SMTP_SSL(cfg["host"], cfg["port"], context=context,
                              timeout=cfg["timeout"]) as server:
            server.login(cfg["user"], cfg["password"])
            server.send_message(msg)
    elif cfg["security"] == "starttls":
        with smtplib.SMTP(cfg["host"], cfg["port"], timeout=cfg["timeout"]) as server:
            server.ehlo()
            server.starttls(context=context)
            server.ehlo()
            server.login(cfg["user"], cfg["password"])
            server.send_message(msg)
    else:  # plain
        with smtplib.SMTP(cfg["host"], cfg["port"], timeout=cfg["timeout"]) as server:
            if cfg["user"]:
                # AUTH over an unencrypted connection would put the username and
                # password on the wire in (base64, trivially reversible) cleartext.
                # Refuse unless the operator explicitly opted in.
                if not cfg["allow_insecure_auth"]:
                    raise RuntimeError(
                        "Refusing to send SMTP credentials over an unencrypted "
                        "'plain' connection. Use SMTP_SECURITY=ssl|starttls, or set "
                        "SMTP_ALLOW_INSECURE_AUTH=1 to override (not recommended)."
                    )
                server.login(cfg["user"], cfg["password"])
            server.send_message(msg)

    return cfg["recipients"]
