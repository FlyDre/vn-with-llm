from __future__ import annotations

from typing import Any


ALLOWED_POSITIONS: list[float] = [round(i / 10, 1) for i in range(11)]


SYSTEM_PROMPT: str = """你是A股日线回测中的量化投研与仓位管理助手。
你只管理单标的、只做多仓位，并且只能输出JSON。

允许的JSON结构：
{
  "signal": "buy" | "sell" | "hold",
  "target_position": float,   // 必须是 0.0, 0.1, ..., 1.0 之一
  "confidence": float,        // 取值范围 0.0 到 1.0
  "reason": string
}

规则：
- 只能使用提供给你的信息。
- 不得假设你能看到未来数据。
- target_position 必须严格取以下11个值之一：0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0。
- 你的任务是管理仓位大小，而不仅是判断涨跌方向。
- 先判断“近1个月市场风格”，再从策略集合中选择更匹配的策略并综合决策。
- 不要机械套用单一规则，要在不同市场风格下动态切换策略权重。
- signal 必须与 target_position 的变化一致：仓位增加=>buy，仓位减少=>sell，仓位不变=>hold。
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

    user_prompt: str = f"""标的代码: {vt_symbol}
信号日期: {signal_date}

指标信息：
- close: {indicators['close']:.4f}
- ma5: {indicators['ma5']:.4f}
- ma20: {indicators['ma20']:.4f}
- volume_ma5: {indicators['volume_ma5']:.2f}
- volume_ratio: {indicators['volume_ratio']:.4f}
- return_5d: {indicators['return_5d']:.4f}
- return_20d: {indicators['return_20d']:.4f}

当前持仓状态：
- current_position: {position_state['current_position']:.4f}
- shares: {position_state['shares']}
- cash: {position_state['cash']:.2f}
- equity: {position_state['equity']:.2f}
- avg_cost: {position_state['avg_cost']}
- unrealized_return: {position_state['unrealized_return']}
- holding_days: {position_state['holding_days']}
- last_signal: {position_state['last_signal']}
- last_target_position: {position_state['last_target_position']:.1f}

最近日线数据：
{chr(10).join(bar_lines)}

任务：
仅基于以上信息，决定下一交易日的目标仓位（A股、仅做多、日线回测）。
target_position 必须且只能从以下11个值中选择：
{", ".join(f"{value:.1f}" for value in ALLOWED_POSITIONS)}

请使用“多策略集合 + 风格匹配”的方法：
1) 先识别近1个月更接近哪种市场风格（趋势、震荡、弱势下跌、反弹修复等）。
2) 评估并综合以下常见量化思路（可按风格动态加权，不必平均）：
- 趋势跟随（均线斜率/突破）
- 动量延续（短中期收益惯性）
- 均值回归（偏离后的回归）
- 波动率与风险控制（高波动降杠杆，低波动稳步持仓）
- 成交量确认（放量突破/缩量回落）
- 回撤控制与仓位管理（不利走势时分步降仓）
3) 最终给出一个离散的目标仓位，不要输出中间过程。

若你要降低或清空仓位，返回 signal="sell"。
若你要提高仓位，返回 signal="buy"。
若你要维持仓位，返回 signal="hold"。
只返回JSON。
"""

    return SYSTEM_PROMPT, user_prompt
