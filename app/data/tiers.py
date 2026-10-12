"""Which stocks were the forecast ranges validated on?

The ranges were developed and tested on the Nifty 200, then tested out-of-sample on stocks the
method never saw (scripts/interval_midcap_check.py). Banner levels:
  nifty200          validated universe: no banner                                   (level "ok")
  nifty500          Nifty 201-500: measured out-of-sample coverage, mild note       (level "info")
  smallcap          market-cap ranks 501-1500: band-calibrated ranges, measured     (level "info")
                    out-of-sample (scripts/interval_tail_check.py, app/data/smallcap_ranks.csv)
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
    "smallcap": "Small stock, ranked about 501-1500 by market cap (outside the Nifty 500). The ranges for these "
                "stocks use a calibration fitted on stocks of this size and were tested out-of-sample on 827 "
                "of them (2022-2026): the 80% range contained the outcome 79.8% / 80.0% of the time at 1 day "
                "and 80.0% / 80.2% at 5 days (ranks 501-1000 / 1001-1500). Ranges are about 25-35% wider "
                "than for Nifty 200 stocks. Not tested: stocks ranked below 1500, listings with under 400 "
                "sessions of history, and stocks delisted before today.",
    "outside_nifty500": "Outside the Nifty 500 and not in the tested market-cap ranks (501-1500). The forecast "
                        "ranges have never been tested on stocks like this. Smaller, less liquid stocks jump "
                        "more, can hit price limits and have patchier data, so real coverage may be well "
                        "below the stated 80%. Treat every range and risk figure here as rough.",
}
TITLES = {
    "ok": "",
    "info": "Ranges tested out-of-sample for this stock group.",
    "warning": "Ranges not validated for this stock.",
}
ESCALATED_TITLE = "Ranges may be less reliable for this stock."
SERIES_NOTE = (" It also trades in the '{series}' segment (restricted / trade-for-trade / SME), where "
               "trading is thinner and price limits are tighter, which the tests did not analyse separately.")


@lru_cache
def _symbols(filename: str) -> frozenset[str]:
    return frozenset(pd.read_csv(DATA / filename)["Symbol"].astype(str).str.upper())


@lru_cache
def _bands() -> dict[str, str]:
    df = pd.read_csv(DATA / "smallcap_ranks.csv")
    return dict(zip(df["symbol"].astype(str).str.upper(), df["band"]))


def smallcap_band(symbol: str | None) -> str | None:
    """'r501_1000' / 'r1001_1500' for tested stocks ranked 501-1500 by market cap outside the Nifty 500."""
    s = (symbol or "").upper()
    if s in _symbols("nifty500.csv"):
        return None
    return _bands().get(s)


def validation_status(symbol: str, series: str | None = None) -> dict:
    s = symbol.upper()
    if s in _symbols("nifty200.csv"):
        tier, level = "nifty200", "ok"
    elif s in _symbols("nifty500.csv"):
        tier, level = "nifty500", "info"
    elif smallcap_band(s):
        tier, level = "smallcap", "info"
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
            "series": series, "validated_universe": "Nifty 500 + market-cap ranks 501-1500 (developed on Nifty 200)",
            "message": msg, "index_lists_as_of": LIST_SNAPSHOT}
