"""Which stocks were the forecast ranges validated on?

The calibrated ranges, the volatility model and the live scorecard were tested only on the Nifty 200.
Other stocks still get ranges, but with an explicit warning:
  nifty200          validated universe: no warning
  nifty500          in the Nifty 500 but outside the Nifty 200: caution (never tested there)
  outside_nifty500  never tested on stocks like this: strong warning
A stock trading in a restricted segment (BE trade-for-trade, BZ, SME) is escalated to a warning.
Index membership comes from committed snapshots of NSE's constituent lists (`nifty200.csv`,
`nifty500.csv`); refresh them when the indices are rebalanced.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

DATA = Path(__file__).parent
LIST_SNAPSHOT = "2026-10-12"
RESTRICTED_SERIES = {"BE", "BZ", "SM", "ST"}

MESSAGES = {
    "nifty200": "In the Nifty 200, the universe the forecast ranges were validated on "
                "(80% ranges contained the outcome 79.9% of the time, out-of-sample).",
    "nifty500": "Outside the Nifty 200. The forecast ranges were validated only on Nifty 200 stocks. This "
                "stock is in the Nifty 500, but mid-sized stocks can behave differently, so its ranges may "
                "cover less often than the stated 80%. This has not been tested.",
    "outside_nifty500": "Outside the Nifty 500. The forecast ranges have never been tested on stocks like "
                        "this. Smaller, less liquid stocks jump more, can hit price limits and have patchier "
                        "data, so real coverage may be well below the stated 80%. Treat every range and "
                        "risk figure here as rough.",
}
SERIES_NOTE = (" It also trades in the '{series}' segment (restricted / trade-for-trade / SME), where "
               "trading is thinner and price limits are tighter.")


@lru_cache
def _symbols(filename: str) -> frozenset[str]:
    return frozenset(pd.read_csv(DATA / filename)["Symbol"].astype(str).str.upper())


def validation_status(symbol: str, series: str | None = None) -> dict:
    s = symbol.upper()
    if s in _symbols("nifty200.csv"):
        tier, level = "nifty200", "ok"
    elif s in _symbols("nifty500.csv"):
        tier, level = "nifty500", "caution"
    else:
        tier, level = "outside_nifty500", "warning"
    msg = MESSAGES[tier]
    series = (series or "").upper() or None
    if series in RESTRICTED_SERIES:
        msg += SERIES_NOTE.format(series=series)
        level = "warning"
    return {"tier": tier, "level": level, "validated": tier == "nifty200", "series": series,
            "validated_universe": "Nifty 200", "message": msg, "index_lists_as_of": LIST_SNAPSHOT}
