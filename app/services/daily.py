"""Daily forward-testing job: publish calibrated forecast ranges and score them as outcomes arrive.

For every stock in the universe, after the NSE close:
  1. fetch recent daily bars (validated, incomplete session dropped);
  2. record a forecast for each horizon: point forecast = latest close ("no change", the only
     point forecast that survived out-of-sample testing) + calibrated 50/80/95 % ranges;
  3. resolve earlier forecasts whose target session has now happened: did the actual move land
     inside each range? (evaluated on adjusted returns so splits/dividends do not distort it);
  4. rewrite small CSV/JSON files that the dashboard reads.
State lives in plain files (committed to a data branch by the GitHub Actions workflow), so no
database or persistent disk is required. Results are genuine forward tests: every forecast is
stored before its outcome exists.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from app.core.config import Settings
from app.core.errors import AppError, ProviderRateLimitError
from app.data.providers.base import DataProvider
from app.data.validation import clean_ohlcv, verify_symbol
from app.models.intervals import Z80, IntervalService

log = logging.getLogger(__name__)

FORECAST_COLS = ["as_of_date", "symbol", "horizon", "reference_close", "point_forecast",
                 "lo50", "hi50", "lo80", "hi80", "lo95", "hi95", "implied_daily_vol", "method",
                 "created_at_utc"]
OUTCOME_COLS = ["as_of_date", "symbol", "horizon", "actual_date", "actual_close", "reference_close",
                "ret_pct", "inside50", "inside80", "inside95", "width80_pct", "method",
                "resolved_at_utc"]


@dataclass
class DailyConfig:
    horizons: tuple[int, ...] = (1, 5)
    history_days: int = 1100          # ~3 years: enough for the 252-session volatility features
    stale_days: int = 6               # skip a stock whose latest bar is older than this
    pause_seconds: float = 0.3        # politeness delay between provider requests
    max_minutes: float | None = None  # soft deadline: remaining symbols are skipped and reported


def _read(path: Path, cols: list[str]) -> pd.DataFrame:
    if path.exists() and path.stat().st_size > 0:
        df = pd.read_csv(path, dtype={"as_of_date": str, "actual_date": str, "symbol": str})
        if "sigma_daily" in df.columns:  # legacy column: raw model scale, not a volatility. Replace it
            df["implied_daily_vol"] = ((df["hi80"] - df["lo80"]) / df["reference_close"]
                                       / (2 * Z80) / np.sqrt(df["horizon"].astype(float)))
            df = df.drop(columns="sigma_daily")
        return df
    return pd.DataFrame(columns=cols)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def fetch_clean(provider: DataProvider, settings: Settings, symbol: str, now: datetime,
                cfg: DailyConfig) -> pd.DataFrame:
    start = (now - timedelta(days=cfg.history_days)).date()
    for attempt in range(2):
        try:
            res = provider.fetch_daily(symbol, start, None)
            break
        except ProviderRateLimitError:
            if attempt == 1:
                raise
            time.sleep(30)
    verify_symbol(res, symbol)
    df, _ = clean_ohlcv(res.frame, now=now, complete_hour_ist=settings.session_complete_hour_ist)
    return df


def resolve_pending(forecasts: pd.DataFrame, outcomes: pd.DataFrame, frames: dict[str, pd.DataFrame],
                    now: datetime) -> list[dict]:
    """Outcomes for forecasts whose h-th following trading bar now exists in the fetched data."""
    done = set(zip(outcomes["as_of_date"], outcomes["symbol"], outcomes["horizon"].astype(int)))
    rows = []
    for f in forecasts.itertuples(index=False):
        key = (f.as_of_date, f.symbol, int(f.horizon))
        df = frames.get(f.symbol)
        if key in done or df is None:
            continue
        asof = pd.Timestamp(f.as_of_date)
        if asof not in df.index:
            continue
        pos, h = df.index.get_loc(asof), int(f.horizon)
        if pos + h >= len(df):
            continue  # the target session has not happened yet
        ref = float(f.reference_close)
        ret = float(df["adj_close"].iloc[pos + h] / df["adj_close"].iloc[pos] - 1)
        inside = {lv: (float(getattr(f, f"lo{lv}")) / ref - 1) <= ret <= (float(getattr(f, f"hi{lv}")) / ref - 1)
                  for lv in (50, 80, 95)}
        rows.append({
            "as_of_date": f.as_of_date, "symbol": f.symbol, "horizon": h,
            "actual_date": str(df.index[pos + h].date()), "actual_close": float(df["close"].iloc[pos + h]),
            "reference_close": ref, "ret_pct": ret * 100, "inside50": inside[50], "inside80": inside[80],
            "inside95": inside[95], "width80_pct": (float(f.hi80) - float(f.lo80)) / ref * 100,
            "method": f.method, "resolved_at_utc": now.isoformat(timespec="seconds"),
        })
    return rows


def coverage_stats(o: pd.DataFrame) -> dict:
    if o.empty:
        return {"n": 0}
    def block(d: pd.DataFrame) -> dict:
        return {"n": int(len(d)),
                **{f"inside{lv}_pct": float(d[f"inside{lv}"].astype(bool).mean() * 100) for lv in (50, 80, 95)},
                "mean_width80_pct": float(d["width80_pct"].mean()),
                "no_change_mape_pct": float(d["ret_pct"].abs().mean())}
    out = block(o)
    recent_dates = sorted(o["actual_date"].unique())[-60:]
    out["last_60_sessions"] = block(o[o["actual_date"].isin(recent_dates)])
    return out


def run_daily(provider: DataProvider, symbols: list[tuple[str, str | None]], out_dir: Path,
              intervals: IntervalService, settings: Settings, now: datetime | None = None,
              cfg: DailyConfig | None = None, say: Callable[[str], None] = print) -> dict:
    cfg = cfg or DailyConfig()
    now = (now or datetime.now(timezone.utc)).replace(microsecond=0)
    out_dir.mkdir(parents=True, exist_ok=True)
    forecasts = _read(out_dir / "forecasts.csv", FORECAST_COLS)
    outcomes = _read(out_dir / "outcomes.csv", OUTCOME_COLS)
    have = set(zip(forecasts["as_of_date"], forecasts["symbol"], forecasts["horizon"].astype(int)))
    names = dict(symbols)
    frames: dict[str, pd.DataFrame] = {}
    failed: list[dict] = []
    new_rows: list[dict] = []
    t0 = time.time()
    for i, (sym, _name) in enumerate(symbols, 1):
        if cfg.max_minutes and (time.time() - t0) / 60 > cfg.max_minutes:
            failed += [{"symbol": s, "reason": "skipped: time budget exhausted"} for s, _ in symbols[i - 1:]]
            say(f"time budget reached after {i - 1} symbols; skipping the rest")
            break
        try:
            df = fetch_clean(provider, settings, sym, now, cfg)
            if df.empty or (now.date() - df.index[-1].date()).days > cfg.stale_days:
                failed.append({"symbol": sym, "reason": "stale or empty data"})
                continue
            frames[sym] = df
            as_of = str(df.index[-1].date())
            for h in cfg.horizons:
                if (as_of, sym, h) in have:
                    continue
                r = intervals.compute(df, h)
                if r is None:
                    failed.append({"symbol": sym, "reason": f"no range for horizon {h} (short history?)"})
                    continue
                ref = float(df["close"].iloc[-1])
                row = {"as_of_date": as_of, "symbol": sym, "horizon": h, "reference_close": ref,
                       "point_forecast": ref, "implied_daily_vol": r.implied_daily_vol, "method": r.method,
                       "created_at_utc": now.isoformat(timespec="seconds")}
                for lv in (0.5, 0.8, 0.95):
                    lo, hi = r.ranges[lv]
                    row[f"lo{int(lv * 100)}"], row[f"hi{int(lv * 100)}"] = ref * (1 + lo), ref * (1 + hi)
                new_rows.append(row)
        except AppError as exc:
            failed.append({"symbol": sym, "reason": f"{exc.code}: {exc.message}"[:200]})
        except Exception as exc:  # noqa: BLE001 - one bad stock must not abort the run; it is reported
            log.exception("daily job failed for %s", sym)
            failed.append({"symbol": sym, "reason": f"{type(exc).__name__}: {exc}"[:200]})
        if i % 20 == 0:
            say(f"[{i}/{len(symbols)}] ok={len(frames)} failed={len(failed)} ({(time.time() - t0) / 60:.1f} min)")
        time.sleep(cfg.pause_seconds)

    if new_rows:
        forecasts = pd.concat([forecasts, pd.DataFrame(new_rows)[FORECAST_COLS]], ignore_index=True)
    resolved = resolve_pending(forecasts, outcomes, frames, now)
    if resolved:
        outcomes = pd.concat([outcomes, pd.DataFrame(resolved)[OUTCOME_COLS]], ignore_index=True)

    _write_csv(forecasts, out_dir / "forecasts.csv")
    _write_csv(outcomes, out_dir / "outcomes.csv")
    if not forecasts.empty:
        latest = forecasts.sort_values("as_of_date").groupby(["symbol", "horizon"], as_index=False).tail(1)
        latest = latest.assign(name=latest["symbol"].map(names)).sort_values(["symbol", "horizon"])
        _write_csv(latest, out_dir / "latest_forecasts.csv")
    if not outcomes.empty:
        o = outcomes.assign(i50=outcomes["inside50"].astype(bool), i80=outcomes["inside80"].astype(bool),
                            i95=outcomes["inside95"].astype(bool))
        daily = (o.groupby(["actual_date", "horizon"])
                 .agg(n=("symbol", "size"), in50=("i50", "sum"), in80=("i80", "sum"), in95=("i95", "sum"))
                 .reset_index())
        _write_csv(daily, out_dir / "coverage_by_day.csv")

    n_ok = len(frames)
    ratio = n_ok / max(len(symbols), 1)
    status = "ok" if ratio >= 0.9 else ("degraded" if ratio >= 0.5 else "failed")
    summary = {
        "status": status, "last_run_utc": now.isoformat(timespec="seconds"),
        "as_of_date": max((str(d.index[-1].date()) for d in frames.values()), default=None),
        "symbols_requested": len(symbols), "symbols_ok": n_ok,
        "failed": failed[:50], "n_failed": len(failed),
        "new_forecasts": len(new_rows), "new_outcomes": len(resolved),
        "forecasts_total": int(len(forecasts)), "outcomes_total": int(len(outcomes)),
        "point_forecast": "no change (latest close); ranges are the validated product",
        "coverage": {str(h): coverage_stats(outcomes[outcomes["horizon"].astype(int) == h])
                     for h in cfg.horizons},
        "targets": {"inside50_pct": 50, "inside80_pct": 80, "inside95_pct": 95},
        "runtime_seconds": round(time.time() - t0, 1),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    say(f"status={status} ok={n_ok}/{len(symbols)} new forecasts={len(new_rows)} "
        f"new outcomes={len(resolved)} total outcomes={len(outcomes)}")
    return summary
