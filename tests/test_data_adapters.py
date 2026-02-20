from bot_cripto.data.adapters import YFinanceAdapter, build_adapter


def test_build_adapter_binance() -> None:
    adapter = build_adapter("binance")
    assert adapter.name == "binance"


def test_build_adapter_bybit() -> None:
    adapter = build_adapter("bybit")
    assert adapter.name == "bybit"


def test_yfinance_symbol_mapping_and_timeframe() -> None:
    assert YFinanceAdapter._to_yf_symbol("BTC/USD") == "BTC-USD"
    assert YFinanceAdapter._to_yf_symbol("EUR/USD") == "EURUSD=X"
    assert YFinanceAdapter.parse_timeframe("5m") == 300
