"""Market-domain helpers (crypto vs forex)."""

from __future__ import annotations


_FOREX_CODES = {
    "USD",
    "EUR",
    "GBP",
    "JPY",
    "CHF",
    "AUD",
    "NZD",
    "CAD",
    "SEK",
    "NOK",
    "DKK",
}


def market_domain(symbol: str) -> str:
    """Return market domain for symbol: `crypto` or `forex`."""
    if "/" not in symbol:
        return "crypto"
    base, quote = symbol.upper().split("/", maxsplit=1)
    if base in _FOREX_CODES and quote in _FOREX_CODES:
        return "forex"
    return "crypto"
