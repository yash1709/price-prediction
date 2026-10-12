"""Ship the band-specific range calibration for market-cap ranks 501-1500 (outside the Nifty 500).

scripts/interval_tail_check.py showed that the Nifty-200 calibration under-covers these stocks at 5 days
while a calibration fitted on the band's OWN stocks (fhs_xgb_grp) passes the pre-registered rule. This
script produces what the app ships for that result:
  1. app/data/smallcap_ranks.csv : which stocks are in which band (the same selection the test used:
     market-cap rank among stocks outside the Nifty 500, >= 400 cached sessions).
  2. `bands` section of app/models/pretrained/interval_calibration.json : z-quantiles of
     (realised return / predicted sigma) per band, horizon and method, measured on the SAME 252-session
     purged calibration window as the Nifty-200 calibration, using the SHIPPED volatility model (which was
     fitted on Nifty 200 data before that window, so there is no leakage).

Usage: python scripts/export_band_calibration.py      (needs the prices cached by interval_tail_check.py)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app.features.engineering import FEATURE_NAMES  # noqa: E402
from app.models.intervals import PRETRAINED, _model  # noqa: E402
from interval_experiment import LEVELS, vol_inputs  # noqa: E402
from interval_midcap_check import cached_rows  # noqa: E402
from interval_tail_check import BANDS  # noqa: E402
from pooled_experiment import build_panel, log  # noqa: E402

DATA = ROOT / "app" / "data"


def main() -> None:
    mc = pd.read_csv(ROOT / "reports" / "market_cap_snapshot.csv")
    ranked = mc[mc["market_cap_inr"].notna()].sort_values("market_cap_inr", ascending=False).reset_index(drop=True)
    ranked["rank"] = ranked.index + 1
    top = ranked[ranked["rank"] <= max(hi for _, hi in BANDS.values())].copy()
    rows = cached_rows(list(top["symbol"]))
    top["sessions"] = top["symbol"].map(rows)
    top = top[top["sessions"] >= 400]
    top["band"] = [next((b for b, (lo, hi) in BANDS.items() if lo <= r <= hi), None) for r in top["rank"]]
    top = top[top["band"].notna()]
    sel = top.rename(columns={"rank": "rank_outside_nifty500"})[
        ["symbol", "band", "rank_outside_nifty500", "market_cap_inr", "as_of"]].sort_values("symbol")
    sel.to_csv(DATA / "smallcap_ranks.csv", index=False)
    log("band members:", sel["band"].value_counts().to_dict())

    meta_path = PRETRAINED / "interval_calibration.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    csv500 = pd.read_csv(DATA / "nifty500.csv")
    panel = build_panel(list(sel["symbol"]), dict(zip(csv500["Symbol"], csv500["Industry"])), [1, 5],
                        cross_sectional=False)
    need = list(FEATURE_NAMES) + [c for c in panel.columns if c.startswith("nifty_") or c.startswith("vix")]
    panel = panel.dropna(subset=need)
    V = vol_inputs(panel, [1, 5])
    X = panel[need].astype("float32").copy()
    for nme in ("ewma", "roll60", "lr252"):
        X[f"log_{nme}"] = np.log(np.clip(V[nme], 1e-4, None)).astype("float32")
    band_of = dict(zip(sel["symbol"], sel["band"]))
    syms = panel.index.get_level_values("symbol")
    group = np.array([band_of[s] for s in syms])
    dates = panel.index.get_level_values("date")

    meta["bands"] = {}
    for h in ("1", "5"):
        hh = int(h)
        info = meta["horizons"][h]
        w0, w1 = (pd.Timestamp(d) for d in info["calibration_window"])
        y = panel[f"y{hh}"].values
        inwin = (dates >= w0) & (dates <= w1) & ~np.isnan(y)
        sq = np.sqrt(hh)
        sx = np.full(len(panel), np.nan)
        sx[inwin] = np.sqrt(np.exp(np.asarray(_model(hh).predict(X.loc[inwin, info["features"]]), dtype=float))) * sq
        se = V["ewma"] * sq
        for band in BANDS:
            m = inwin & (group == band)
            entry = {"n_rows": int(m.sum()), "n_stocks": int(pd.Series(syms[m]).nunique()),
                     "calibration_window": info["calibration_window"], "z_quantiles": {}}
            for method, sig in (("fhs_xgb", sx), ("fhs_ewma", se)):
                ok = m & np.isfinite(sig) & (sig > 0)
                z = y[ok] / sig[ok]
                entry["z_quantiles"][method] = {
                    str(lv): [float(np.quantile(z, (1 - lv) / 2)), float(np.quantile(z, 1 - (1 - lv) / 2))]
                    for lv in LEVELS}
            meta["bands"].setdefault(band, {})[h] = entry
            n200z = info["z_quantiles"]["fhs_xgb"]["0.8"]
            bz = entry["z_quantiles"]["fhs_xgb"]["0.8"]
            log(f"h={h} {band}: {entry['n_stocks']} stocks, {entry['n_rows']:,} rows; z80 band "
                f"{bz[0]:+.2f}/{bz[1]:+.2f} vs Nifty 200 {n200z[0]:+.2f}/{n200z[1]:+.2f}")
    meta["bands_note"] = ("Band-specific calibration for market-cap ranks 501-1500 outside the Nifty 500; "
                          "evidence in reports/interval_tail_check.json (fhs_xgb_grp).")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log("wrote", meta_path)


if __name__ == "__main__":
    main()
