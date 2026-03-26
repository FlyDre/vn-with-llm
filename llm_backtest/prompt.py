from __future__ import annotations

from typing import Any


ALLOWED_POSITIONS: list[float] = [round(i / 10, 1) for i in range(11)]


SYSTEM_PROMPT: str = """You are a quantitative stock position manager for A-share daily-bar backtests.
You manage a single long-only position and must respond with JSON only.

Allowed JSON schema:
{
  "signal": "buy" | "sell" | "hold",
  "target_position": float,   // must be one of 0.0, 0.1, ..., 1.0
  "confidence": float,        // between 0.0 and 1.0
  "reason": string
}

Rules:
- Use only the information provided.
- Do not assume access to future data.
- target_position must be exactly one of: 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0.
- You are managing position size, not only predicting direction.
- If current close falls below MA5, consider reducing position.
- If current close falls below MA20, prefer strong reduction or full exit.
- If there is a high-volume bearish reversal after a recent rally, consider taking profit or reducing risk.
- If unrealized loss is widening or holding days are long without trend continuation, consider reducing or exiting.
- If trend is strong and healthy, you may increase or keep position.
- Keep signal consistent with target_position change: increasing position => buy, reducing position => sell, unchanged => hold.
"""


def build_daily_bar_prompt(payload: dict[str, Any]) -> tuple[str, str]:
    vt_symbol: str = payload["vt_symbol"]
    signal_date: str = payload["signal_date"]
    indicators: dict[str, Any] = payload["indicators"]
    bars: list[dict[str, Any]] = payload["bars"]
    position_state: dict[str, Any] = payload["position_state"]

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

Current position state:
- current_position: {position_state['current_position']:.4f}
- shares: {position_state['shares']}
- cash: {position_state['cash']:.2f}
- equity: {position_state['equity']:.2f}
- avg_cost: {position_state['avg_cost']}
- unrealized_return: {position_state['unrealized_return']}
- holding_days: {position_state['holding_days']}
- last_signal: {position_state['last_signal']}
- last_target_position: {position_state['last_target_position']:.1f}

Recent daily bars:
{chr(10).join(bar_lines)}

Task:
Based only on the information above, decide the next trading day's target position
for a long-only A-share daily-bar backtest strategy.
Choose target_position from exactly these 11 values:
{", ".join(f"{value:.1f}" for value in ALLOWED_POSITIONS)}
If you want to reduce or clear the existing position, return signal="sell".
If you want to increase exposure, return signal="buy".
If you want to keep current exposure, return signal="hold".
Return JSON only.
"""

    return SYSTEM_PROMPT, user_prompt
