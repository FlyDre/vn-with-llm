from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.database import get_database

from llm_backtest.llm_client import LlmDecision, OpenAICompatibleClient
from llm_backtest.prompt import ALLOWED_POSITIONS, build_daily_bar_prompt


# Fill these values once, then you can run the script directly without
# re-entering them in PowerShell every time.
LLM_SETTINGS: dict[str, object] = {
    "api_key": "sk-ZGQvW4wh6dkj5L3NOKhbfotgitUqu5Hftp1qtNR8LjH4DnEj",
    "model": "gpt-5.3-codex",
    "base_url": "https://api.squarefaceicon.org/v1",
    "temperature": 0.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate offline LLM signals from vn.py daily bars.")
    parser.add_argument("--vt-symbol", required=True, help="Example: 000559.SZSE")
    parser.add_argument("--start", required=True, help="Example: 2023-01-01")
    parser.add_argument("--end", required=True, help="Example: 2026-03-01")
    parser.add_argument("--window", type=int, default=20, help="Visible lookback bars for each decision")
    parser.add_argument("--output", required=True, help="CSV output path")
    parser.add_argument("--capital", type=float, default=400000.0, help="Initial capital for sequential signal generation")
    parser.add_argument("--lot-size", type=int, default=100, help="Minimum tradable share lot")
    parser.add_argument("--model", default=None, help="Optional override for the model in LLM_SETTINGS")
    parser.add_argument("--base-url", default=None, help="Optional override for the base_url in LLM_SETTINGS")
    parser.add_argument("--api-key", default=None, help="Optional override for the api_key in LLM_SETTINGS")
    parser.add_argument("--temperature", type=float, default=None, help="Optional override for the temperature in LLM_SETTINGS")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output CSV if present")
    return parser.parse_args()


def load_daily_bars(vt_symbol: str, start: datetime, end: datetime) -> list[dict]:
    symbol, exchange_str = vt_symbol.split(".")
    exchange = Exchange(exchange_str)

    database = get_database()
    bars = database.load_bar_data(symbol, exchange, Interval.DAILY, start, end)

    records: list[dict] = []
    for bar in bars:
        records.append(
            {
                "date": bar.datetime.date().isoformat(),
                "open": float(bar.open_price),
                "high": float(bar.high_price),
                "low": float(bar.low_price),
                "close": float(bar.close_price),
                "volume": float(bar.volume),
                "turnover": float(bar.turnover),
            }
        )
    return records


def calc_indicators(window_bars: list[dict]) -> dict[str, float]:
    closes = [bar["close"] for bar in window_bars]
    volumes = [bar["volume"] for bar in window_bars]

    ma5 = sum(closes[-5:]) / min(5, len(closes))
    ma20 = sum(closes[-20:]) / min(20, len(closes))
    volume_ma5 = sum(volumes[-5:]) / min(5, len(volumes))
    close = closes[-1]
    volume_ratio = volumes[-1] / volume_ma5 if volume_ma5 else 0.0
    return_5d = close / closes[-5] - 1 if len(closes) >= 5 and closes[-5] else 0.0
    return_20d = close / closes[0] - 1 if closes[0] else 0.0

    return {
        "close": close,
        "ma5": ma5,
        "ma20": ma20,
        "volume_ma5": volume_ma5,
        "volume_ratio": volume_ratio,
        "return_5d": return_5d,
        "return_20d": return_20d,
    }


def snap_position(value: float) -> float:
    return min(ALLOWED_POSITIONS, key=lambda x: abs(x - value))


def infer_signal(target_position: float, current_position: float) -> str:
    if target_position > current_position + 0.049:
        return "buy"
    if target_position < current_position - 0.049:
        return "sell"
    return "hold"


def apply_target_on_open(
    state: dict[str, float | int | str | None],
    target_position: float,
    execution_price: float,
    lot_size: int,
) -> None:
    shares = int(state["shares"])
    cash = float(state["cash"])
    equity = cash + shares * execution_price
    target_shares_raw = equity * target_position / execution_price if execution_price > 0 else 0.0
    target_shares = int(target_shares_raw // lot_size) * lot_size
    delta = target_shares - shares

    if delta > 0:
        cash -= delta * execution_price
        new_shares = shares + delta
        previous_cost = float(state["avg_cost"]) if state["avg_cost"] is not None else 0.0
        state["avg_cost"] = (
            (previous_cost * shares + execution_price * delta) / new_shares
            if new_shares > 0
            else None
        )
        state["shares"] = new_shares
        if shares == 0:
            state["holding_days"] = 0
    elif delta < 0:
        reduce_shares = abs(delta)
        cash += reduce_shares * execution_price
        new_shares = shares - reduce_shares
        state["shares"] = new_shares
        if new_shares == 0:
            state["avg_cost"] = None
            state["holding_days"] = 0

    state["cash"] = cash


def build_position_state(
    state: dict[str, float | int | str | None],
    close_price: float,
) -> dict[str, float | int | str]:
    shares = int(state["shares"])
    cash = float(state["cash"])
    equity = cash + shares * close_price
    current_position = (shares * close_price / equity) if equity > 0 else 0.0

    if shares > 0 and state["avg_cost"] is not None:
        avg_cost = float(state["avg_cost"])
        unrealized_return = close_price / avg_cost - 1 if avg_cost > 0 else 0.0
        state["holding_days"] = int(state["holding_days"]) + 1
        avg_cost_text = f"{avg_cost:.4f}"
        unrealized_text = f"{unrealized_return:.4f}"
    else:
        unrealized_return = 0.0
        avg_cost_text = "none"
        unrealized_text = "0.0000"
        state["holding_days"] = 0

    state["equity"] = equity

    return {
        "current_position": current_position,
        "shares": shares,
        "cash": cash,
        "equity": equity,
        "avg_cost": avg_cost_text,
        "unrealized_return": unrealized_text,
        "holding_days": int(state["holding_days"]),
        "last_signal": str(state["last_signal"]),
        "last_target_position": float(state["last_target_position"]),
    }


def generate_signals(args: argparse.Namespace) -> list[dict]:
    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)
    bars = load_daily_bars(args.vt_symbol, start, end)

    if len(bars) < args.window + 1:
        raise ValueError("Not enough daily bars for signal generation")

    api_key = str(args.api_key or LLM_SETTINGS["api_key"]).strip()
    model = str(args.model or LLM_SETTINGS["model"]).strip()
    base_url = str(args.base_url or LLM_SETTINGS["base_url"]).strip()
    temperature = float(args.temperature if args.temperature is not None else LLM_SETTINGS["temperature"])

    if not api_key:
        raise ValueError("LLM api_key is empty. Fill LLM_SETTINGS['api_key'] in scripts/generate_llm_signals.py")
    if not model:
        raise ValueError("LLM model is empty. Fill LLM_SETTINGS['model'] in scripts/generate_llm_signals.py")

    client = OpenAICompatibleClient(
        api_key=api_key,
        model=model,
        base_url=base_url,
        temperature=temperature,
    )

    rows: list[dict] = []
    state: dict[str, float | int | str | None] = {
        "cash": float(args.capital),
        "shares": 0,
        "avg_cost": None,
        "holding_days": 0,
        "equity": float(args.capital),
        "last_signal": "hold",
        "last_target_position": 0.0,
    }
    pending_target_position: float | None = None
    existing_by_signal_date: dict[str, dict] = {}

    if args.resume:
        output_path = Path(args.output)
        if output_path.exists():
            with output_path.open("r", encoding="utf-8-sig", newline="") as f:
                existing_rows = list(csv.DictReader(f))
            existing_by_signal_date = {str(row["signal_date"]): row for row in existing_rows}

    for index in range(args.window - 1, len(bars) - 1):
        visible_bars = bars[index - args.window + 1:index + 1]
        today_bar = visible_bars[-1]
        signal_date = visible_bars[-1]["date"]
        trade_date = bars[index + 1]["date"]

        if pending_target_position is not None:
            apply_target_on_open(state, pending_target_position, today_bar["open"], args.lot_size)

        position_state = build_position_state(state, today_bar["close"])

        existing_row = existing_by_signal_date.get(signal_date)
        if existing_row:
            target_position = float(existing_row["target_position"])
            signal = str(existing_row["signal"])
            rows.append(existing_row)
            pending_target_position = target_position
            state["last_signal"] = signal
            state["last_target_position"] = target_position
            continue

        payload = {
            "vt_symbol": args.vt_symbol,
            "signal_date": signal_date,
            "bars": visible_bars,
            "indicators": calc_indicators(visible_bars),
            "position_state": position_state,
        }

        system_prompt, user_prompt = build_daily_bar_prompt(payload)
        decision: LlmDecision = client.complete(system_prompt, user_prompt)
        decision.target_position = snap_position(decision.target_position)
        current_position = float(position_state["current_position"])
        decision.signal = infer_signal(decision.target_position, current_position)

        row = {
            "signal_date": signal_date,
            "trade_date": trade_date,
            "vt_symbol": args.vt_symbol,
            "signal": decision.signal,
            "target_position": f"{decision.target_position:.4f}",
            "confidence": f"{decision.confidence:.4f}",
            "reason": decision.reason,
        }
        rows.append(row)
        if args.resume:
            append_row(row, args.output)
        print(f"[{len(rows)}] {signal_date} -> {trade_date} {row['signal']} {row['target_position']}")

        pending_target_position = decision.target_position
        state["last_signal"] = decision.signal
        state["last_target_position"] = decision.target_position

    return rows


def save_rows(rows: list[dict], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "signal_date",
        "trade_date",
        "vt_symbol",
        "signal",
        "target_position",
        "confidence",
        "reason",
    ]

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_row(row: dict[str, str], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    fieldnames = [
        "signal_date",
        "trade_date",
        "vt_symbol",
        "signal",
        "target_position",
        "confidence",
        "reason",
    ]
    with path.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    rows = generate_signals(args)
    if not args.resume:
        save_rows(rows, args.output)
    print(f"Generated {len(rows)} rows -> {args.output}")


if __name__ == "__main__":
    main()
