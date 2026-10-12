"""Validation tiers: warn when a stock is outside the universe the ranges were tested on."""
import json
from pathlib import Path

import pandas as pd
import pytest

from app.data.tiers import DATA, LIST_SNAPSHOT, RESTRICTED_SERIES, validation_status
from tests.conftest import make_prices

REPORTS = Path(__file__).resolve().parents[1] / "reports"

N200 = set(pd.read_csv(DATA / "nifty200.csv")["Symbol"])
N500 = set(pd.read_csv(DATA / "nifty500.csv")["Symbol"])
MID = sorted(N500 - N200)[0]          # a real Nifty 500 stock that is not in the Nifty 200


def test_committed_lists_are_sane():
    assert len(N200) == 200 and 450 <= len(N500) <= 520
    assert N200 <= N500 and len(N500 - N200) > 250          # Nifty 200 is nested inside the Nifty 500
    assert "RELIANCE" in N200 and MID not in N200 and MID in N500


def test_tiers_and_messages():
    ok = validation_status("RELIANCE")
    assert (ok["tier"], ok["level"], ok["validated"]) == ("nifty200", "ok", True)
    assert "79.9%" in ok["message"]
    assert ok["title"] == ""
    mid = validation_status(MID.lower())                      # case-insensitive
    assert (mid["tier"], mid["level"], mid["validated"]) == ("nifty500", "info", True)
    assert "tested out-of-sample" in mid["title"] and "out-of-sample on 276 Nifty 201-500" in mid["message"]
    assert "have not been tested" in mid["message"] and "never" not in mid["message"].lower()
    out = validation_status("ZZZNOPE")
    assert (out["tier"], out["level"], out["validated"]) == ("outside_nifty500", "warning", False)
    assert out["title"] == "Ranges not validated for this stock."
    assert "never been tested" in out["message"] and "well below the stated 80%" in out["message"]
    assert out["index_lists_as_of"] == LIST_SNAPSHOT


def test_quoted_evidence_matches_the_committed_reports():
    """The messages quote measured numbers: they must equal what the reports say (rounded)."""
    mid = json.loads((REPORTS / "interval_midcap_check.json").read_text(encoding="utf-8"))
    g = {h: mid["horizons"][h]["groups"]["mid"]["methods"]["fhs_xgb"]["0.8"] for h in ("1", "5")}
    msg = validation_status(MID)["message"]
    assert f"{g['1']['coverage_pct']:.1f}% of the time at 1 day" in msg
    assert f"{g['5']['coverage_pct']:.1f}% at 5 days" in msg
    assert f"{mid['mid_stocks_used']} Nifty 201-500 stocks" in msg
    n200 = mid["horizons"]["1"]["groups"]["n200"]["methods"]["fhs_xgb"]["0.8"]["mean_width_pct"]
    wide = g["1"]["mean_width_pct"] / n200 - 1
    assert f"{round(wide * 100)}% wider" in msg
    exp = json.loads((REPORTS / "interval_experiment.json").read_text(encoding="utf-8"))
    cov = exp["horizons"]["1"]["methods"]["fhs_xgb"]["0.8"]["coverage_pct"]
    assert f"{cov:.1f}%" in validation_status("RELIANCE")["message"]


@pytest.mark.parametrize("series", sorted(RESTRICTED_SERIES))
def test_restricted_segments_escalate_to_a_warning(series):
    v = validation_status(MID, series)
    assert v["level"] == "warning" and v["tier"] == "nifty500" and f"'{series}' segment" in v["message"]
    assert v["title"] == "Ranges may be less reliable for this stock."
    assert validation_status("RELIANCE", series)["level"] == "warning"     # even a validated name
    assert validation_status("ZZZNOPE", series)["title"] == "Ranges not validated for this stock."
    assert validation_status(MID, "EQ")["level"] == "info" and validation_status(MID, None)["level"] == "info"


def test_api_reports_validation_on_history_and_risk(client, provider, prices):
    provider.frames["RELIANCE"] = prices.copy()
    provider.frames[MID] = make_prices(700, seed=9)
    h = client.get("/stocks/RELIANCE/history").json()["validation"]
    assert h["tier"] == "nifty200" and h["level"] == "ok"
    h = client.get(f"/stocks/{MID}/history").json()["validation"]
    assert h["tier"] == "nifty500" and h["level"] == "info" and h["validated"] is True
    h = client.get("/stocks/TESTCO/history").json()["validation"]          # synthetic symbol: in neither list
    assert h["tier"] == "outside_nifty500" and h["level"] == "warning"
    assert client.get("/stocks/TESTCO/risk").json()["validation"]["tier"] == "outside_nifty500"
    assert client.get(f"/stocks/{MID}/risk").json()["validation"]["tier"] == "nifty500"
    # a stock's trading segment is stored on import and escalates its level
    from app.database.models import Stock
    with client.app.state.ctx.repo() as repo:
        repo.s.get(Stock, MID).series = "BE"
    h = client.get(f"/stocks/{MID}/history").json()["validation"]
    assert h["level"] == "warning" and h["series"] == "BE" and "'BE' segment" in h["message"]
