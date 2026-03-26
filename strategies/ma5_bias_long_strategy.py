from vnpy_ctastrategy import (
    CtaTemplate,
    StopOrder,
    TickData,
    BarData,
    TradeData,
    OrderData,
    BarGenerator,
    ArrayManager,
)


class Ma5BiasLongStrategy(CtaTemplate):
    """A-share long-only strategy based on MA5 breakout."""

    author = "Codex"

    ma_window: int = 5
    entry_multiplier: float = 1.01
    fixed_size: int = 1

    ma_value: float = 0.0
    last_close: float = 0.0

    parameters = ["ma_window", "entry_multiplier", "fixed_size"]
    variables = ["ma_value", "last_close"]

    def on_init(self) -> None:
        """Callback when strategy is inited."""
        self.write_log("策略初始化")

        self.bg: BarGenerator = BarGenerator(self.on_bar)
        self.am: ArrayManager = ArrayManager()

        self.load_bar(self.ma_window + 5)

    def on_start(self) -> None:
        """Callback when strategy is started."""
        self.write_log("策略启动")
        self.put_event()

    def on_stop(self) -> None:
        """Callback when strategy is stopped."""
        self.write_log("策略停止")
        self.put_event()

    def on_tick(self, tick: TickData) -> None:
        """Callback of new tick data update."""
        self.bg.update_tick(tick)

    def on_bar(self, bar: BarData) -> None:
        """Callback of new bar data update."""
        self.cancel_all()

        am: ArrayManager = self.am
        am.update_bar(bar)
        if not am.inited:
            return

        self.ma_value = am.sma(self.ma_window)
        self.last_close = bar.close_price

        if self.pos == 0:
            if self.last_close > self.entry_multiplier * self.ma_value:
                self.buy(bar.close_price, self.fixed_size)
        elif self.pos > 0:
            if self.last_close < self.ma_value:
                self.sell(bar.close_price, abs(self.pos))

        self.put_event()

    def on_order(self, order: OrderData) -> None:
        """Callback of new order data update."""
        pass

    def on_trade(self, trade: TradeData) -> None:
        """Callback of new trade data update."""
        self.put_event()

    def on_stop_order(self, stop_order: StopOrder) -> None:
        """Callback of stop order update."""
        pass
