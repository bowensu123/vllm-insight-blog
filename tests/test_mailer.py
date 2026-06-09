"""Tests for the private SMTP mailer. No network: smtplib is faked."""
from __future__ import annotations

import smtplib

import pytest

from vllm_insights import mailer


@pytest.fixture
def smtp_env(monkeypatch):
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_USERNAME", "me@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "secret-code")
    monkeypatch.setenv("MAIL_TO", "dest@huawei.com")
    for k in ("SMTP_PORT", "SMTP_SECURITY", "MAIL_FROM"):
        monkeypatch.delenv(k, raising=False)


class FakeSMTP:
    """Records the protocol calls; used for both SMTP and SMTP_SSL."""
    instances: list = []

    def __init__(self, host, port, *a, **k):
        self.host, self.port = host, port
        self.calls = []
        self.kind = type(self).__name__
        FakeSMTP.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def ehlo(self, *a):
        self.calls.append("ehlo")

    def starttls(self, *a, **k):
        self.calls.append("starttls")

    def login(self, u, p):
        self.calls.append(("login", u, p))

    def send_message(self, msg):
        self.calls.append(("send", msg))


class FakeSMTP_SSL(FakeSMTP):
    pass


@pytest.fixture(autouse=True)
def _reset():
    FakeSMTP.instances = []
    yield


def test_missing_config_raises(monkeypatch):
    for k in ("SMTP_HOST", "SMTP_USERNAME", "SMTP_PASSWORD", "MAIL_TO"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(mailer.MailNotConfigured):
        mailer._resolve_config()


def test_build_message_is_multipart_with_html(smtp_env):
    msg = mailer.build_message("Subj", "# Title\n\nhello world", "a@x.com", ["b@y.com"])
    assert msg["Subject"] == "Subj"
    assert msg["To"] == "b@y.com"
    payloads = msg.get_payload()
    types = {p.get_content_type() for p in payloads}
    assert types == {"text/plain", "text/html"}
    html = msg.get_body(preferencelist=("html",)).get_content()
    assert "<h1" in html and "hello world" in html


def test_recipients_split_and_default_from(smtp_env, monkeypatch):
    monkeypatch.setenv("MAIL_TO", "a@x.com, b@y.com ,c@z.com")
    cfg = mailer._resolve_config()
    assert cfg["recipients"] == ["a@x.com", "b@y.com", "c@z.com"]
    assert cfg["sender"] == "me@example.com"  # defaults to SMTP_USERNAME


def test_ssl_path_used_for_port_465(smtp_env, monkeypatch):
    monkeypatch.setattr(smtplib, "SMTP_SSL", FakeSMTP_SSL)
    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)
    recipients = mailer.send_digest("# hi\n\nbody", "Subject")
    assert recipients == ["dest@huawei.com"]
    srv = FakeSMTP.instances[-1]
    assert srv.kind == "FakeSMTP_SSL"
    assert ("login", "me@example.com", "secret-code") in srv.calls
    assert any(isinstance(c, tuple) and c[0] == "send" for c in srv.calls)
    assert "starttls" not in srv.calls


def test_starttls_path_used_for_port_587(smtp_env, monkeypatch):
    monkeypatch.setenv("SMTP_PORT", "587")
    monkeypatch.setattr(smtplib, "SMTP_SSL", FakeSMTP_SSL)
    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)
    mailer.send_digest("# hi\n\nbody", "Subject")
    srv = FakeSMTP.instances[-1]
    assert srv.kind == "FakeSMTP"           # plain SMTP, upgraded via STARTTLS
    assert "starttls" in srv.calls
    assert ("login", "me@example.com", "secret-code") in srv.calls


def test_explicit_security_overrides_port(smtp_env, monkeypatch):
    monkeypatch.setenv("SMTP_PORT", "465")
    monkeypatch.setenv("SMTP_SECURITY", "starttls")
    monkeypatch.setattr(smtplib, "SMTP_SSL", FakeSMTP_SSL)
    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)
    mailer.send_digest("x", "s")
    assert FakeSMTP.instances[-1].kind == "FakeSMTP"  # starttls forced despite 465


def test_blank_recipients_treated_as_not_configured(smtp_env, monkeypatch):
    monkeypatch.setenv("MAIL_TO", ", ,")  # only separators/whitespace
    with pytest.raises(mailer.MailNotConfigured):
        mailer._resolve_config()


def test_plain_refuses_cleartext_auth_by_default(smtp_env, monkeypatch):
    monkeypatch.setenv("SMTP_PORT", "25")
    monkeypatch.setenv("SMTP_SECURITY", "plain")
    monkeypatch.delenv("SMTP_ALLOW_INSECURE_AUTH", raising=False)
    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)
    monkeypatch.setattr(smtplib, "SMTP_SSL", FakeSMTP_SSL)
    with pytest.raises(RuntimeError, match="unencrypted"):
        mailer.send_digest("x", "s")


def test_plain_allows_auth_with_explicit_override(smtp_env, monkeypatch):
    monkeypatch.setenv("SMTP_PORT", "25")
    monkeypatch.setenv("SMTP_SECURITY", "plain")
    monkeypatch.setenv("SMTP_ALLOW_INSECURE_AUTH", "1")
    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)
    monkeypatch.setattr(smtplib, "SMTP_SSL", FakeSMTP_SSL)
    mailer.send_digest("x", "s")
    srv = FakeSMTP.instances[-1]
    assert srv.kind == "FakeSMTP"
    assert any(isinstance(c, tuple) and c[0] == "login" for c in srv.calls)
