"""DeepSeek JSON transport and a bounded request journal."""

import json
import os
from pathlib import Path
import socket
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from molcraft.errors import MolCraftError
from molcraft.io.serialization import save_json


class ProviderError(MolCraftError):
    def __init__(self, message, *, retryable=False, exhausted=False):
        super().__init__(message)
        self.retryable = retryable
        self.exhausted = exhausted


class DeepSeekClient:
    model = "deepseek-flash"
    base_url = "https://api.deepseek.com"

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
        except (URLError, socket.timeout, ConnectionError, OSError) as exc:
            raise ProviderError(f"DeepSeek transport: {type(exc).__name__}",
                                retryable=True) from None
        except (UnicodeError, ValueError):
            raise ProviderError("DeepSeek returned an unreadable response",
                                retryable=True) from None

    def available(self):
        result = self._request("/models")
        return any(row.get("id") == self.model for row in result.get("data", []))

    def complete(self, messages):
        response = self._request("/chat/completions", {
            "model": self.model, "messages": messages,
            "thinking": {"type": "enabled"}, "reasoning_effort": "high",
            "response_format": {"type": "json_object"}, "max_tokens": 16384,
            "stream": False,
        })
        try:
            choice = response["choices"][0]
            if choice["finish_reason"] != "stop":
                raise ValueError
            content = json.loads(choice["message"]["content"])
            if not isinstance(content, dict):
                raise ValueError
        except (ValueError, KeyError, IndexError, TypeError):
            raise ProviderError("DeepSeek returned incomplete or invalid JSON",
                                retryable=True) from None
        usage = {k: v for k, v in response.get("usage", {}).items()
                 if k in {"prompt_tokens", "completion_tokens", "total_tokens"}
                 and type(v) is int}
        return content, usage


class RequestJournal:
    """Each logical request has two slots. A reservation survives process loss."""

    def __init__(self, root):
        self.root = Path(root) if root is not None else None

    def request(self, slot, client, messages, validate):
        feedback = None
        status = "interrupted_request"
        for attempt in (1, 2):
            path = self.root / f"{slot}-{attempt}.json" if self.root else None
            if path and path.exists():
                receipt = json.loads(path.read_text(encoding="utf-8"))
                status = receipt["status"]
                if status == "ok":
                    return validate(receipt["content"])
                feedback = receipt.get("error", "Prior call interrupted")
                continue
            if path:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("x", encoding="utf-8") as handle:
                    json.dump({"status": "reserved"}, handle)
                    handle.flush()
                    os.fsync(handle.fileno())
            context = messages
            if feedback:
                context = [*messages, {"role": "user",
                    "content": f"Previous attempt was rejected: {feedback}. Return corrected JSON."}]
            started = time.monotonic()
            usage = {}
            try:
                content, usage = client.complete(context)
                result = validate(content)
                receipt = {"status": "ok", "content": content}
            except ProviderError as exc:
                status = "provider_error"
                receipt = {"status": status, "error": str(exc)}
                if not exc.retryable:
                    if path:
                        save_json({**receipt, "seconds": time.monotonic() - started}, path)
                    raise
            except (MolCraftError, ValueError, TypeError, KeyError) as exc:
                status = "invalid"
                receipt = {"status": status, "error": str(exc)[:300]}
            if path:
                save_json({**receipt, "usage": usage, "seconds": time.monotonic() - started}, path)
            if receipt["status"] == "ok":
                return result
            feedback = receipt["error"]
        if status == "invalid":
            return None
        raise ProviderError("DeepSeek request budget exhausted after transport failure",
                            exhausted=True)
