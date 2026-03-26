from __future__ import annotations

from typing import Any


SYSTEM_PROMPT: str = """You are a quantitative stock timing assistant.
You only work for daily-bar backtests.
You must respond with JSON only.

Allowed JSON schema:
{
  "signal": "buy" | "sell" | "hold",
  "target_position": float,   // between 0.0 and 1.0
  "confidence": float,        // between 0.0 and 1.0
  "reason": string
}

Rules:
- Use only the information provided.
- Do not assume access to future data.
- target_position must be a number between 0 and 1.
- If the trend is unclear, return hold with a conservative target_position.
"""


def build_daily_bar_prompt(payload: dict[str, Any]) -> tuple[str, str]:
    vt_symbol: str = payload["vt_symbol"]
    signal_date: str = payload["signal_date"]
    indicators: dict[str, Any] = payload["indicators"]
    bars: list[dict[str, Any]] = payload["bars"]

    bar_lines: list[str] = []
    for bar in bars:
        line = (
            f"{bar['date']} "
            f"O={bar['open']:.4f} H={bar['high']:.4f} "
            f"L={bar['low']:.4f} C={bar['close']:.4f} "
            f"V={bar['volume']:.0f}"
        )
        bar_lines.append(line)

    user_prompt: str = f"""VT Symbol: {vt_symbol}
Signal date: {signal_date}

Indicators:
- close: {indicators['close']:.4f}
- ma5: {indicators['ma5']:.4f}
- ma20: {indicators['ma20']:.4f}
- volume_ma5: {indicators['volume_ma5']:.2f}
- volume_ratio: {indicators['volume_ratio']:.4f}
- return_5d: {indicators['return_5d']:.4f}
- return_20d: {indicators['return_20d']:.4f}

Recent daily bars:
{chr(10).join(bar_lines)}

Task:
Based only on the information above, decide the next trading day's target action
for a long-only A-share daily-bar backtest strategy.
Return JSON only.
"""

    return SYSTEM_PROMPT, user_prompt

