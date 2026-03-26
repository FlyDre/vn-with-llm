from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

from vnpy.trader.constant import Direction, Offset
from vnpy_ctastrategy import (
    BarData,
    BarGenerator,
    CtaTemplate,
    OrderData,
    StopOrder,
    TickData,
    TradeData,
)


class LlmSignalStrategy(CtaTemplate):
    """Daily-bar long-only strategy driven by offline LLM signals."""

    author = "Codex"

    signal_path: str = "data/llm_signals/000001.SZSE.csv"
    min_confidence: float = 0.0
    max_position: float = 1.0
    lot_size: int = 100

    current_signal: str = "hold"
    current_target_position: float = 0.0
    current_confidence: float = 0.0
    estimated_cash: float = 0.0
    estimated_equity: float = 0.0

    parameters = ["signal_path", "min_confidence", "max_position", "lot_size"]
    variables = [
        "current_signal",
        "current_target_position",
        "current_confidence",
        "estimated_cash",
        "estimated_equity",
    ]

    def on_init(self) -> None:
        self.write_log("LLM signal strategy initialized")
        self.bg: BarGenerator = BarGenerator(self.on_bar)
        self.signal_map: dict[date, dict[str, str]] = self._load_signal_map()
        self.estimated_cash = float(getattr(self.cta_engine, "capital", 0))
        self.estimated_equity = self.estimated_cash
        self.write_log(
            f"Initial capital={self.estimated_cash:.2f}, signals={len(self.signal_map)}, vt_symbol={self.vt_symbol}"
        )
        self.load_bar(1)

    def on_start(self) -> None:
        self.write_log("LLM signal strategy started")
        self.put_event()

    def on_stop(self) -> None:
        self.write_log("LLM signal strategy stopped")
        self.put_event()

    def on_tick(self, tick: TickData) -> None:
        self.bg.update_tick(tick)

    def on_bar(self, bar: BarData) -> None:
        self.cancel_all()

        signal_row: dict[str, str] | None = self.signal_map.get(bar.datetime.date())
        if not signal_row:
            self._mark_to_market(bar)
            self.put_event()
            return

        self.current_signal = signal_row["signal"]
        self.current_target_position = min(max(float(signal_row["target_position"]), 0.0), self.max_position)
        self.current_confidence = float(signal_row["confidence"])
        self.write_log(
            f"Signal hit on {bar.datetime.date()}: signal={self.current_signal}, "
            f"target_position={self.current_target_position:.4f}, confidence={self.current_confidence:.4f}, pos={self.pos}"
        )

        if self.current_confidence < self.min_confidence:
            self._mark_to_market(bar)
            self.put_event()
            return

        self._rebalance(bar)
        self._mark_to_market(bar)
        self.put_event()

    def on_trade(self, trade: TradeData) -> None:
        contract_size: int = self.get_size()
        turnover: float = trade.price * trade.volume * contract_size

        if trade.direction == Direction.LONG and trade.offset == Offset.OPEN:
            self.estimated_cash -= turnover
        elif trade.direction == Direction.SHORT and trade.offset == Offset.CLOSE:
            self.estimated_cash += turnover

        self.put_event()

    def on_order(self, order: OrderData) -> None:
        pass

    def on_stop_order(self, stop_order: StopOrder) -> None:
        pass

    def _rebalance(self, bar: BarData) -> None:
        contract_size: int = self.get_size()
        self.estimated_equity = self.estimated_cash + self.pos * bar.close_price * contract_size

        raw_target_volume: float = (
            self.estimated_equity * self.current_target_position / (bar.close_price * contract_size)
        )
        target_volume: int = self._round_down_to_lot(raw_target_volume)
        current_volume: int = int(self.pos)
        delta: int = target_volume - current_volume
        self.write_log(
            f"Rebalance on {bar.datetime.date()}: equity={self.estimated_equity:.2f}, "
            f"price={bar.close_price:.4f}, target_volume={target_volume}, current_volume={current_volume}, delta={delta}"
        )

        if delta > 0:
            self.buy(bar.close_price, delta)
        elif delta < 0:
            self.sell(bar.close_price, abs(delta))

    def _mark_to_market(self, bar: BarData) -> None:
        contract_size: int = self.get_size()
        self.estimated_equity = self.estimated_cash + self.pos * bar.close_price * contract_size

    def _round_down_to_lot(self, volume: float) -> int:
        lots: int = int(volume // self.lot_size)
        return lots * self.lot_size

    def _load_signal_map(self) -> dict[date, dict[str, str]]:
        path: Path = Path(self.signal_path)
        if not path.is_absolute():
            path = Path.cwd().joinpath(path)

        signal_map: dict[date, dict[str, str]] = {}
        if not path.exists():
            self.write_log(f"Signal file not found: {path}")
            return signal_map

        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("vt_symbol") != self.vt_symbol:
                    continue

                trade_date: date = date.fromisoformat(str(row["trade_date"]))
                signal_map[trade_date] = row

        self.write_log(f"Loaded {len(signal_map)} signals from {path}")
        return signal_map
