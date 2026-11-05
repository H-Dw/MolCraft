"""DeepSeek JSON transport and a bounded request journal."""

import json
import hashlib
from http.client import HTTPException
import os
from pathlib import Path
import socket
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from molcraft.errors import ConfigurationError, MolCraftError, StaleProposalError
from molcraft.io.serialization import save_json


class ProviderError(MolCraftError):
    def __init__(self, message, *, retryable=False, exhausted=False, usage=None, finish_reason=None):
        super().__init__(message)
        self.retryable = retryable
        self.exhausted = exhausted
        self.usage = usage or {}
        self.finish_reason = finish_reason


class DeepSeekClient:
    model = "deepseek-flash"
    base_url = "https://api.deepseek.com"
    retry_backoff_seconds = 1.0

    def __init__(self, timeout=600):
        self.timeout = timeout

    def _request(self, path, payload=None):
        key = os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            raise ProviderError("DEEPSEEK_API_KEY is missing from the task environment")
        data = json.dumps(payload).encode() if payload is not None else None
        request = Request(self.base_url + path, data=data, headers={
            "Authorization": f"Bearer {key}", "Content-Type": "application/json",
        })
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except HTTPError as exc:
            retryable = exc.code in (408, 429) or 500 <= exc.code < 600
            raise ProviderError(f"DeepSeek HTTP {exc.code}", retryable=retryable) from None
        except (URLError, socket.timeout, ConnectionError, OSError, HTTPException) as exc:
            raise ProviderError(f"DeepSeek transport: {type(exc).__name__}",
                                retryable=True) from None
        except (UnicodeError, ValueError):
            raise ProviderError("DeepSeek returned an unreadable response",
                                retryable=True) from None

    def available(self):
        result = self._request("/models")
        if not isinstance(result, dict) or not isinstance(result.get("data"), list):
            raise ProviderError("DeepSeek returned an invalid model list", retryable=True)
        return any(isinstance(row, dict) and row.get("id") == self.model for row in result["data"])

    def complete(self, messages):
        response = self._request("/chat/completions", {
            "model": self.model, "messages": messages,
            "thinking": {"type": "enabled"}, "reasoning_effort": "high",
            "response_format": {"type": "json_object"}, "max_tokens": 16384,
            "stream": False,
        })
        if not isinstance(response, dict):
            raise ProviderError("DeepSeek returned a null or non-object response", retryable=True)
        raw_usage = response.get("usage") or {}
        if not isinstance(raw_usage, dict):
            raw_usage = {}
        usage = {k: v for k, v in raw_usage.items() if type(v) is int and v >= 0}
        details = raw_usage.get("completion_tokens_details", {})
        if isinstance(details, dict) and type(details.get("reasoning_tokens")) is int:
            usage["reasoning_tokens"] = details["reasoning_tokens"]
        finish_reason = None
        try:
            choice = response["choices"][0]
            finish_reason = choice["finish_reason"]
            if finish_reason != "stop":
                raise ValueError
            content = json.loads(choice["message"]["content"])
            if not isinstance(content, dict):
                raise ValueError
        except (ValueError, KeyError, IndexError, TypeError):
            raise ProviderError("DeepSeek returned incomplete or invalid JSON",
                                retryable=True, usage=usage, finish_reason=finish_reason) from None
        return content, usage


class RequestJournal:
    """Initial request plus up to three retries; reservations survive process loss."""

    def __init__(self, root, *, bind_messages=False, max_retries=3):
        if type(max_retries) is not int or not 0 <= max_retries <= 3:
            raise ConfigurationError("API max_retries must be an integer between 0 and 3")
        self.root = Path(root) if root is not None else None
        self.bind_messages = bind_messages
        self.max_retries = max_retries

    def request(self, slot, client, messages, validate):
        feedback = None
        feedback_content = None
        status = "interrupted_request"
        for attempt in range(1, self.max_retries + 2):
            context = messages
            if feedback:
                if self.bind_messages and isinstance(feedback_content, dict):
                    context = [*context, {"role": "assistant", "content": json.dumps(feedback_content)}]
                context = [*context, {"role": "user",
                    "content": f"Previous attempt was rejected: {feedback}. Return corrected JSON."}]
            binding = ({"message_hash": hashlib.sha256(json.dumps(context, sort_keys=True,
                        separators=(",", ":")).encode()).hexdigest()} if self.bind_messages else {})
            path = self.root / f"{slot}-{attempt}.json" if self.root else None
            if path and path.exists():
                receipt = json.loads(path.read_text(encoding="utf-8"))
                if receipt.get("max_retries", self.max_retries) != self.max_retries:
                    raise StaleProposalError("Saved LLM request has a different retry budget")
                if self.bind_messages and receipt.get("message_hash") != binding["message_hash"]:
                    raise StaleProposalError("Saved LLM request has a different prompt or feedback")
                status = receipt["status"]
                if status == "ok":
                    return validate(receipt["content"])
                if status == "provider_error" and receipt.get("retryable") is False:
                    raise ProviderError(receipt["error"], usage=receipt.get("usage"),
                                        finish_reason=receipt.get("finish_reason"))
                feedback = receipt.get("error") if status == "invalid" else None
                feedback_content = receipt.get("content") if status == "invalid" else None
                continue
            if path:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("x", encoding="utf-8") as handle:
                    json.dump({"status": "reserved", **binding, "attempt": attempt,
                               "max_retries": self.max_retries}, handle)
                    handle.flush()
                    os.fsync(handle.fileno())
            started = time.monotonic()
            delay = float(getattr(client, "retry_backoff_seconds", 0)) * (2 ** (attempt - 2)) if attempt > 1 else 0.0
            if delay:
                time.sleep(delay)
            attempt_metadata = {"attempt": attempt, "max_retries": self.max_retries,
                                "retry_delay_seconds": delay}
            usage = {}
            content = None
            try:
                content, usage = client.complete(context)
                result = validate(content)
                receipt = {"status": "ok", "content": content}
            except ProviderError as exc:
                usage = exc.usage
                status = "provider_error"
                receipt = {"status": status, "error": str(exc), "finish_reason": exc.finish_reason,
                           "retryable": exc.retryable}
                if not exc.retryable:
                    if path:
                        save_json({**receipt, **binding, **attempt_metadata, "usage": usage,
                                   "seconds": time.monotonic() - started}, path)
                    raise
            except (MolCraftError, ValueError, TypeError, KeyError) as exc:
                status = "invalid"
                receipt = {"status": status, "error": str(exc)[:300], "content": content}
            if path:
                save_json({**receipt, **binding, **attempt_metadata, "usage": usage, "seconds": time.monotonic() - started}, path)
            if receipt["status"] == "ok":
                return result
            feedback = receipt.get("error") if receipt["status"] == "invalid" else None
            feedback_content = receipt.get("content") if receipt["status"] == "invalid" else None
        if status == "invalid":
            return None
        raise ProviderError("DeepSeek request budget exhausted after transport failure",
                            exhausted=True)
