from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from typing import Any


@dataclass
class LlmDecision:
    signal: str
    target_position: float
    confidence: float
    reason: str


class OpenAICompatibleClient:
    """Use PowerShell Invoke-RestMethod to match the user's verified request path."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout: int = 60,
        temperature: float = 0.0,
    ) -> None:
        self.api_key: str = api_key or os.getenv("LLM_API_KEY", "")
        self.model: str = model or os.getenv("LLM_MODEL", "")
        self.base_url: str = (base_url or os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.timeout: int = timeout
        self.temperature: float = temperature

        if not self.api_key:
            raise ValueError("LLM api_key is empty")
        if not self.model:
            raise ValueError("LLM model is empty")

    def complete(self, system_prompt: str, user_prompt: str) -> LlmDecision:
        if self.base_url.endswith("/chat/completions"):
            url = self.base_url
            use_responses_style = False
        elif self.base_url.endswith("/responses"):
            url = self.base_url
            use_responses_style = True
        else:
            url = f"{self.base_url}/responses"
            use_responses_style = True
        if use_responses_style:
            payload: dict[str, Any] = {
                "model": self.model,
                "temperature": self.temperature,
                "input": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }
        else:
            payload = {
                "model": self.model,
                "temperature": self.temperature,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                # Some gateways alias codex-style models to responses semantics.
                "input": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }

        command = (
            "$ProgressPreference='SilentlyContinue'; "
            f"$body = @'\n{json.dumps(payload, ensure_ascii=False)}\n'@; "
            f"$headers = @{{ Authorization = '{self.api_key}' }}; "
            f"$resp = Invoke-WebRequest -Uri '{url}' -Method Post -Headers $headers "
            "-ContentType 'application/json' -Body $body; "
            "$resp.Content"
        )

        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                command,
            ],
            capture_output=True,
            text=True,
            timeout=self.timeout,
            encoding="utf-8",
            errors="replace",
        )

        if result.returncode != 0:
            stderr = result.stderr.strip()
            stdout = result.stdout.strip()
            detail = stderr or stdout or "unknown error"
            raise RuntimeError(f"LLM request failed via PowerShell: {detail}")

        content: str = result.stdout.strip()
        if not content:
            detail = result.stderr.strip() or "stdout and stderr are both empty"
            raise RuntimeError(f"LLM request returned empty response: {detail}")

        data: dict[str, Any] = json.loads(content)
        message: str = self._extract_message_text(data)
        parsed: dict[str, Any] = self._extract_json(message)

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
                texts = []
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
