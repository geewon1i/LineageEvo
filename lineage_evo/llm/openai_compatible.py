"""OpenAI-compatible chat-completions client.

This uses only the Python standard library and is never used by default smoke
runs. Tests mock the HTTP layer, so the suite does not call a real API.
"""

from __future__ import annotations

import json
import time
from urllib import error, request

from lineage_evo.config import LLMConfig
from lineage_evo.llm.client import LLMResponse


class OpenAICompatibleLLMClient:
    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig.from_env()
        self.config.validate_for_request()

    def complete(self, *, system_prompt: str, user_prompt: str) -> LLMResponse:
        body = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        req = request.Request(
            self._chat_url(),
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        payload = self._send_with_retry(req)

        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"LLM API response missing message content: {payload}") from exc
        return LLMResponse(content=str(content))

    def _send_with_retry(self, req: request.Request) -> dict:
        last_error: Exception | None = None
        attempts = max(1, self.config.max_retry)
        for attempt in range(1, attempts + 1):
            try:
                with request.urlopen(req, timeout=self.config.timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
            except error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(f"LLM API request failed with HTTP {exc.code}: {detail}")
            except error.URLError as exc:
                last_error = RuntimeError(f"LLM API request failed: {exc.reason}")
            except TimeoutError as exc:
                last_error = RuntimeError(f"LLM API request timed out: {exc}")
            except OSError as exc:
                last_error = RuntimeError(f"LLM API network error: {exc}")
            if attempt < attempts:
                time.sleep(self.config.retry_wait_seconds)
        if last_error is not None:
            raise last_error
        raise RuntimeError("LLM API request failed")

    def _chat_url(self) -> str:
        base = self.config.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"
