"""NSE symbol normalisation/validation and the `SYMBOL.NS` ticker mapping."""
from __future__ import annotations

import re

from app.core.errors import InvalidSymbolError

_SYMBOL_RE = re.compile(r"^[A-Z0-9&\-_]{1,20}$")


def normalize_symbol(raw: str) -> str:
    """Upper-case, strip whitespace and a trailing `.NS`; validate the characters."""
    sym = (raw or "").strip().upper()
    if sym.endswith(".NS"):
        sym = sym[:-3]
    if not _SYMBOL_RE.match(sym):
        raise InvalidSymbolError(
            f"'{raw}' is not a valid NSE symbol (allowed: A-Z, 0-9, '&', '-', '_', max 20 chars)."
        )
    return sym


def to_yahoo_ticker(symbol: str) -> str:
    return f"{normalize_symbol(symbol)}.NS"
