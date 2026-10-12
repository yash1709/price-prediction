"""Validation tiers: warn when a stock is outside the universe the ranges were tested on."""
import pandas as pd
import pytest

from app.data.tiers import DATA, LIST_SNAPSHOT, RESTRICTED_SERIES, validation_status
from tests.conftest import make_prices

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
    mid = validation_status(MID.lower())                      # case-insensitive
    assert (mid["tier"], mid["level"], mid["validated"]) == ("nifty500", "caution", False)
    assert "validated only on Nifty 200" in mid["message"] and "Nifty 500" in mid["message"]
    out = validation_status("ZZZNOPE")
    assert (out["tier"], out["level"], out["validated"]) == ("outside_nifty500", "warning", False)
    assert "never been tested" in out["message"] and "well below the stated 80%" in out["message"]
    assert out["index_lists_as_of"] == LIST_SNAPSHOT and out["validated_universe"] == "Nifty 200"


@pytest.mark.parametrize("series", sorted(RESTRICTED_SERIES))
def test_restricted_segments_escalate_to_a_warning(series):
    v = validation_status(MID, series)
    assert v["level"] == "warning" and v["tier"] == "nifty500" and f"'{series}' segment" in v["message"]
    assert validation_status("RELIANCE", series)["level"] == "warning"     # even a validated name
    assert validation_status(MID, "EQ")["level"] == "caution" and validation_status(MID, None)["level"] == "caution"


def test_api_reports_validation_on_history_and_risk(client, provider, prices):
    provider.frames["RELIANCE"] = prices.copy()
    provider.frames[MID] = make_prices(700, seed=9)
    h = client.get("/stocks/RELIANCE/history").json()["validation"]
    assert h["tier"] == "nifty200" and h["level"] == "ok"
    h = client.get(f"/stocks/{MID}/history").json()["validation"]
    assert h["tier"] == "nifty500" and h["level"] == "caution"
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
