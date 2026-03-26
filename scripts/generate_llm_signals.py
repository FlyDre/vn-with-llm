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
from llm_backtest.prompt import build_daily_bar_prompt


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
    parser.add_argument("--model", default=None, help="Optional override for the model in LLM_SETTINGS")
    parser.add_argument("--base-url", default=None, help="Optional override for the base_url in LLM_SETTINGS")
    parser.add_argument("--api-key", default=None, help="Optional override for the api_key in LLM_SETTINGS")
    parser.add_argument("--temperature", type=float, default=None, help="Optional override for the temperature in LLM_SETTINGS")
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

    for index in range(args.window - 1, len(bars) - 1):
        visible_bars = bars[index - args.window + 1:index + 1]
        signal_date = visible_bars[-1]["date"]
        trade_date = bars[index + 1]["date"]

        payload = {
            "vt_symbol": args.vt_symbol,
            "signal_date": signal_date,
            "bars": visible_bars,
            "indicators": calc_indicators(visible_bars),
        }

        system_prompt, user_prompt = build_daily_bar_prompt(payload)
        decision: LlmDecision = client.complete(system_prompt, user_prompt)

        rows.append(
            {
                "signal_date": signal_date,
                "trade_date": trade_date,
                "vt_symbol": args.vt_symbol,
                "signal": decision.signal,
                "target_position": f"{decision.target_position:.4f}",
                "confidence": f"{decision.confidence:.4f}",
                "reason": decision.reason,
            }
        )

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


def main() -> None:
    args = parse_args()
    rows = generate_signals(args)
    save_rows(rows, args.output)
    print(f"Generated {len(rows)} rows -> {args.output}")


if __name__ == "__main__":
    main()
