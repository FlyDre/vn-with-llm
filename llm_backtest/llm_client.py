from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass
class LlmDecision:
    signal: str
    target_position: float
    confidence: float
    reason: str


class RetryableResponseError(RuntimeError):
    """Transient upstream response error that should be retried."""


class OpenAICompatibleClient:
    """HTTP client for OpenAI-compatible gateways."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout: int = 120,
        max_retries: int = 4,
        temperature: float = 0.0,
        user_agent: str | None = None,
    ) -> None:
        self.api_key: str = api_key or os.getenv("LLM_API_KEY", "")
        self.model: str = model or os.getenv("LLM_MODEL", "")
        self.base_url: str = (base_url or os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.timeout: int = int(timeout)
        self.max_retries: int = max(1, int(max_retries))
        self.temperature: float = float(temperature)
        self.user_agent: str = (
            user_agent
            or os.getenv(
                "LLM_USER_AGENT",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            )
        )

        if not self.api_key:
            raise ValueError("LLM api_key is empty")
        if not self.model:
            raise ValueError("LLM model is empty")

    def complete(self, system_prompt: str, user_prompt: str) -> LlmDecision:
        request_variants = self._build_request_variants(system_prompt, user_prompt)
        auth_variants = self._build_auth_variants()
        last_error = "unknown error"
        call_retries = 6

        for attempt in range(1, self.max_retries + 1):
            for variant in request_variants:
                for auth in auth_variants:
                    parsed: dict[str, Any] | None = None
                    for call_attempt in range(1, call_retries + 1):
                        try:
                            content = self._http_post_json(variant["url"], variant["payload"], auth)
                            data: dict[str, Any] = self._parse_response_content(content, variant["url"])
                            message: str = self._extract_message_text(data)
                            parsed = self._extract_json(message)
                            break
                        except RetryableResponseError as exc:
                            last_error = f"{variant['label']} ({auth['label']}): {exc}"
                            if call_attempt < call_retries:
                                print("重试中...")
                                wait_seconds = min(4 + call_attempt * 2, 20)
                                time.sleep(wait_seconds)
                                continue
                            parsed = None
                            break
                        except Exception as exc:
                            last_error = f"{variant['label']} ({auth['label']}): {exc}"
                            parsed = None
                            break

                    if parsed is None:
                        continue

                    signal: str = str(parsed.get("signal", "hold")).lower()
                    if signal not in {"buy", "sell", "hold"}:
                        signal = "hold"

                    target_position: float = float(parsed.get("target_position", 0.0))
                    confidence: float = float(parsed.get("confidence", 0.0))
                    reason: str = str(parsed.get("reason", ""))

                    target_position = min(max(target_position, 0.0), 1.0)
                    confidence = min(max(confidence, 0.0), 1.0)

                    return LlmDecision(
                        signal=signal,
                        target_position=target_position,
                        confidence=confidence,
                        reason=reason,
                    )

            if attempt < self.max_retries:
                time.sleep(min(2 ** attempt, 15))

        raise RuntimeError(
            "LLM request failed after retries on all endpoints. "
            f"Last error: {last_error}"
        )

    def _build_request_variants(self, system_prompt: str, user_prompt: str) -> list[dict[str, Any]]:
        responses_payload: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "stream": False,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        chat_payload: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        if self.base_url.endswith("/responses"):
            root = self.base_url[: -len("/responses")].rstrip("/")
            return [
                {"label": "responses", "url": self.base_url, "payload": responses_payload},
                {"label": "chat/completions", "url": f"{root}/chat/completions", "payload": chat_payload},
            ]

        if self.base_url.endswith("/chat/completions"):
            root = self.base_url[: -len("/chat/completions")].rstrip("/")
            return [
                {"label": "chat/completions", "url": self.base_url, "payload": chat_payload},
                {"label": "responses", "url": f"{root}/responses", "payload": responses_payload},
            ]

        root = self.base_url.rstrip("/")
        variants: list[dict[str, Any]] = [
            {"label": "chat/completions", "url": f"{root}/chat/completions", "payload": chat_payload},
        ]
        # Some providers only expose chat/completions; keep responses opt-in.
        if os.getenv("LLM_ENABLE_RESPONSES", "").strip() == "1":
            variants.append(
                {"label": "responses", "url": f"{root}/responses", "payload": responses_payload}
            )
        return variants

    def _build_auth_variants(self) -> list[dict[str, str]]:
        return [
            {"label": "bearer", "authorization": f"Bearer {self.api_key}"},
            {"label": "raw", "authorization": self.api_key},
        ]

    def _http_post_json(self, url: str, payload: dict[str, Any], auth: dict[str, str]) -> str:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Authorization", auth["authorization"])
        req.add_header("Content-Type", "application/json")
        req.add_header("Accept", "application/json")
        req.add_header("User-Agent", self.user_agent)

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                content = resp.read().decode("utf-8", errors="replace")
                if not content.strip():
                    raise RuntimeError(f"{url} status={resp.status} with empty body")
                content_type = str(resp.headers.get("Content-Type", "")).lower()
                if "json" not in content_type and "<!doctype html>" in content.lower():
                    raise RuntimeError(f"{url} status={resp.status} returned HTML page, not JSON")
                return content
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            preview = error_body.replace("\n", " ").strip()
            if len(preview) > 240:
                preview = preview[:237] + "..."
            if exc.code == 429:
                body_lower = error_body.lower()
                if "model_cooldown" in body_lower or "cooling down" in body_lower:
                    raise RetryableResponseError(
                        f"{url} HTTP 429 model cooldown"
                    ) from exc
                raise RetryableResponseError(f"{url} HTTP 429 rate limited") from exc
            if "error code: 1010" in error_body.lower():
                raise RuntimeError(
                    f"{url} HTTP {exc.code} blocked by Cloudflare (1010), gateway denied access"
                ) from exc
            if "<!DOCTYPE html>" in error_body:
                raise RuntimeError(f"{url} HTTP {exc.code}: gateway returned HTML error page")
            raise RuntimeError(f"{url} HTTP {exc.code}: {preview or exc.reason}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"{url} network error: {exc.reason}") from exc

    def _parse_response_content(self, content: str, url: str) -> dict[str, Any]:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        if "data:" in content:
            text_parts: list[str] = []
            saw_chunk = False
            saw_empty_choices_chunk = False
            for raw_line in content.splitlines():
                line = raw_line.strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if not payload or payload == "[DONE]":
                    continue
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict) and event.get("object") == "chat.completion.chunk":
                    saw_chunk = True
                    choices = event.get("choices")
                    if isinstance(choices, list) and not choices:
                        saw_empty_choices_chunk = True
                piece = self._extract_text_from_stream_event(event)
                if piece:
                    text_parts.append(piece)

            merged = "".join(text_parts).strip()
            if merged:
                return {"output_text": merged}
            if saw_chunk and saw_empty_choices_chunk:
                raise RetryableResponseError(
                    f"{url} returned SSE chunks with empty choices and no text payload"
                )

        preview = content.replace("\n", " ").strip()
        if len(preview) > 220:
            preview = preview[:217] + "..."
        raise RuntimeError(f"{url} returned non-JSON body: {preview or '<empty>'}")

    @staticmethod
    def _extract_text_from_stream_event(event: dict[str, Any]) -> str:
        choices = event.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        first = choices[0]
        if not isinstance(first, dict):
            return ""

        delta = first.get("delta", {})
        if isinstance(delta, dict):
            content = delta.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts: list[str] = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        parts.append(str(item.get("text", "")))
                return "".join(parts)

        message = first.get("message", {})
        if isinstance(message, dict):
            content = message.get("content", "")
            if isinstance(content, str):
                return content
        return ""

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise ValueError(f"LLM response is not valid JSON: {text}")
        return json.loads(match.group(0))

    @staticmethod
    def _extract_message_text(data: dict[str, Any]) -> str:
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message", {})
            content = message.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                texts: list[str] = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        texts.append(str(item.get("text", "")))
                if texts:
                    return "\n".join(texts)

        output = data.get("output")
        if isinstance(output, list):
            texts = []
            for item in output:
                if not isinstance(item, dict):
                    continue
                for content in item.get("content", []):
                    if isinstance(content, dict) and content.get("type") == "output_text":
                        texts.append(str(content.get("text", "")))
            if texts:
                return "\n".join(texts)

        if isinstance(data.get("output_text"), str):
            return str(data["output_text"])

        raise ValueError(f"Cannot extract text from LLM response: {json.dumps(data, ensure_ascii=False)}")
