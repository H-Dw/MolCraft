"""Bounded null/connection retries, usage receipts, and crash-safe replay."""
import json
from http.client import IncompleteRead
from urllib.error import URLError

import pytest

from molcraft.agent.deepseek import DeepSeekClient, ProviderError, RequestJournal
from molcraft.errors import ConfigurationError, StaleProposalError


@pytest.mark.parametrize("failure", [None, URLError("connection reset"), IncompleteRead(b"{")])
def test_null_and_transport_failures_retry_three_times_then_succeed(tmp_path, monkeypatch, failure):
    client = DeepSeekClient()
    calls, sleeps = [], []

    def request(*args):
        calls.append(args)
        if len(calls) <= 3:
            if isinstance(failure, Exception):
                raise ProviderError("transport failure", retryable=True) from failure
            return failure
        return {"choices": [{"finish_reason": "stop", "message": {"content": '{"drafts":[]}'}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5}}

    monkeypatch.setattr(client, "_request", request)
    monkeypatch.setattr("molcraft.agent.deepseek.time.sleep", sleeps.append)
    journal = RequestJournal(tmp_path, bind_messages=True)
    messages = [{"role": "user", "content": "test"}]
    assert journal.request("draft", client, messages, lambda c: c) == {"drafts": []}
    assert len(calls) == 4 and sleeps == [1, 2, 4]
    receipts = [json.loads(p.read_text()) for p in sorted(tmp_path.glob("draft-*.json"))]
    assert len({r["message_hash"] for r in receipts}) == 1
    assert receipts[-1]["usage"]["prompt_tokens"] == 10
    assert journal.request("draft", client, messages, lambda c: c) == {"drafts": []}
    assert len(calls) == 4 and sleeps == [1, 2, 4]


def test_real_transport_wrapper_classifies_incomplete_reads_for_retry(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "synthetic-test-key")

    def fail(*args, **kwargs):
        raise IncompleteRead(b"{")

    monkeypatch.setattr("molcraft.agent.deepseek.urlopen", fail)
    with pytest.raises(ProviderError) as error:
        DeepSeekClient()._request("/chat/completions", {})
    assert error.value.retryable


@pytest.mark.parametrize("retryable", [False, True])
def test_exhaustion_and_fatal_errors_do_not_reset_on_replay(tmp_path, retryable):
    class Client:
        calls = 0

        def complete(self, messages):
            self.calls += 1
            raise ProviderError("failure", retryable=retryable, usage={"completion_tokens": 12})

    client = Client()
    for _ in range(2):
        with pytest.raises(ProviderError) as error:
            RequestJournal(tmp_path).request("draft", client, [], lambda c: c)
        assert error.value.exhausted is retryable
    assert client.calls == (4 if retryable else 1)
    assert sum(json.loads(p.read_text())["usage"]["completion_tokens"] for p in tmp_path.glob("*.json")) == client.calls*12


def test_retry_budget_is_validated_and_bound_to_receipts(tmp_path):
    for invalid in (-1, 4, True):
        with pytest.raises(ConfigurationError):
            RequestJournal(tmp_path, max_retries=invalid)

    class Client:
        def complete(self, messages):
            return {}, {}

    RequestJournal(tmp_path, max_retries=1).request("draft", Client(), [], lambda c: c)
    with pytest.raises(StaleProposalError, match="retry budget"):
        RequestJournal(tmp_path, max_retries=3).request("draft", Client(), [], lambda c: c)
