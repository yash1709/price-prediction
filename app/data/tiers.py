"""Which stocks were the forecast ranges validated on?

The ranges were developed and tested on the Nifty 200, then tested out-of-sample on stocks the
method never saw (scripts/interval_midcap_check.py). Banner levels:
  nifty200          validated universe: no banner                                   (level "ok")
  nifty500          Nifty 201-500: measured out-of-sample coverage, mild note       (level "info")
  outside_nifty500  never tested on stocks like this: strong warning                (level "warning")
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
    "nifty200": "In the Nifty 200, the universe the forecast ranges were developed on "
                "(80% ranges contained the outcome 79.9% of the time, out-of-sample).",
    "nifty500": "Nifty 201-500 stock. The ranges were fitted on Nifty 200 stocks and then tested "
                "out-of-sample on 276 Nifty 201-500 stocks (2022-2026): the 80% range contained the "
                "outcome 79.3% of the time at 1 day and 79.0% at 5 days, in line with the target. "
                "Ranges are about 17% wider than for Nifty 200 stocks, as expected. Stocks outside "
                "the Nifty 500 have not been tested.",
    "outside_nifty500": "Outside the Nifty 500. The forecast ranges have never been tested on stocks like "
                        "this. Smaller, less liquid stocks jump more, can hit price limits and have patchier "
                        "data, so real coverage may be well below the stated 80%. Treat every range and "
                        "risk figure here as rough.",
}
TITLES = {
    "ok": "",
    "info": "Ranges tested out-of-sample for this stock group.",
    "warning": "Ranges not validated for this stock.",
}
ESCALATED_TITLE = "Ranges may be less reliable for this stock."
SERIES_NOTE = (" It also trades in the '{series}' segment (restricted / trade-for-trade / SME), where "
               "trading is thinner and price limits are tighter, which the tests did not cover.")


@lru_cache
def _symbols(filename: str) -> frozenset[str]:
    return frozenset(pd.read_csv(DATA / filename)["Symbol"].astype(str).str.upper())


def validation_status(symbol: str, series: str | None = None) -> dict:
    s = symbol.upper()
    if s in _symbols("nifty200.csv"):
        tier, level = "nifty200", "ok"
    elif s in _symbols("nifty500.csv"):
        tier, level = "nifty500", "info"
    else:
        tier, level = "outside_nifty500", "warning"
    msg, title = MESSAGES[tier], TITLES[level]
    series = (series or "").upper() or None
    if series in RESTRICTED_SERIES:
        msg += SERIES_NOTE.format(series=series)
        if tier != "outside_nifty500":
            title = ESCALATED_TITLE
        level = "warning"
    return {"tier": tier, "level": level, "title": title, "validated": tier != "outside_nifty500",
            "series": series, "validated_universe": "Nifty 500 (developed on Nifty 200)",
            "message": msg, "index_lists_as_of": LIST_SNAPSHOT}
