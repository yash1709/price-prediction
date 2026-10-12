"""Streamlit dashboard. Talks to the FastAPI backend only through `api_client.ApiClient`."""
from __future__ import annotations

import os
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import charts  # noqa: E402
from api_client import ApiClient, ApiError  # noqa: E402

DISCLAIMER = ("Research and forecasting tool — **not investment advice**. Forecasts are model "
              "estimates; historical backtest results do not guarantee future accuracy. "
              "Prices are end-of-day (delayed), in Indian Rupees (₹).")
DEFAULT_API = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="NSE Stock Forecasts", page_icon="📈", layout="wide")


def inr(v: float | None) -> str:
    return "—" if v is None else f"₹{v:,.2f}"


def pct(v: float | None, signed: bool = False) -> str:
    return "—" if v is None else (f"{v:+.2f}%" if signed else f"{v:.2f}%")


# --------------------------------------------------------------------------------------------
# Sidebar / connection
# --------------------------------------------------------------------------------------------
with st.sidebar:
    st.title("📈 NSE Forecasts")
    api_url = st.text_input("API URL", DEFAULT_API)
    embedded_note = ""
    try:
        health = ApiClient(api_url).health()
    except ApiError:
        # No external API: run the backend inside this Streamlit process (single-process hosting).
        try:
            import embedded
            api_url = embedded.ensure_started()
            health = ApiClient(api_url).health()
            embedded_note = " · embedded API"
        except Exception as e:  # noqa: BLE001 - shown to the user, then stop
            st.error(f"No API at the configured URL and the embedded API failed to start: {e}")
            st.stop()
    api = ApiClient(api_url)
    st.caption(f"API v{health['version']} · provider: {health['data_provider']}{embedded_note}")


@st.cache_data(ttl=300, show_spinner=False)
def load_universe(url: str) -> dict:
    return ApiClient(url).stocks()


@st.cache_data(ttl=300, show_spinner=False)
def search_universe(url: str, q: str) -> dict:
    return ApiClient(url).stocks(q=q, limit=50)


with st.sidebar:
    uni = load_universe(api_url)
    q = st.text_input("Search any NSE stock", placeholder="Type a name or symbol: Infosys, DMART, M&M",
                      help="Press Enter. A forecast is prepared automatically for the stock you pick.")
    q = q.strip()
    symbol = ""
    if q:
        matches = search_universe(api_url, q)["stocks"]
        if matches:
            options = {f"{s['symbol']} — {s['name'] or ''}": s["symbol"] for s in matches}
            symbol = options[st.selectbox(f"{len(matches)} match(es)", list(options))]
        else:
            symbol = q.upper().removesuffix(".NS")
            st.caption(f"'{q}' is not in the stock list; trying it as the NSE symbol **{symbol}**.")
    else:
        options = {f"{s['symbol']} — {s['name'] or ''}": s["symbol"] for s in uni["stocks"]}
        labels = list(options)
        default_idx = next((i for i, k in enumerate(labels) if k.startswith("RELIANCE")), 0)
        choice = st.selectbox("…or pick from the list", labels, index=default_idx if labels else None)
        symbol = options.get(choice, "")
    st.caption(f"Universe: {uni['total']} symbols. {uni['note']}")
    years = st.slider("History (years)", 1, 15, 8)
    start_d = date.today() - timedelta(days=int(365.25 * years))
    d_range = st.date_input("Chart date range", value=(start_d, date.today()),
                            max_value=date.today())
    st.info(DISCLAIMER)

if not symbol:
    st.warning("Select or enter an NSE symbol in the sidebar.")
    st.stop()

if isinstance(d_range, tuple) and len(d_range) == 2:
    r_start, r_end = d_range
else:
    r_start, r_end = start_d, date.today()


@st.cache_data(ttl=120, show_spinner=False)
def load_history(url: str, sym: str, start: str, end: str, nonce: int) -> dict:
    return ApiClient(url).history(sym, start, end)


if "nonce" not in st.session_state:
    st.session_state.nonce = 0


def bump() -> None:
    st.session_state.nonce += 1


try:
    with st.spinner(f"Loading {symbol} price history…"):
        hist_resp = load_history(api_url, symbol, r_start.isoformat(), r_end.isoformat(),
                                 st.session_state.nonce)
except ApiError as e:
    st.error(f"Could not load data for **{symbol}**: {e.message}")
    st.stop()

hist = pd.DataFrame(hist_resp["rows"])
if hist.empty:
    st.warning("No price rows in the selected range.")
    st.stop()
hist["date"] = pd.to_datetime(hist["date"])
hist = hist.set_index("date")
dinfo = hist_resp["data"]

st.header(f"{symbol} {('· ' + hist_resp['name']) if hist_resp.get('name') else ''}")
for w in dinfo.get("warnings", []):
    st.warning(w)
_val = hist_resp.get("validation")
if _val and _val["level"] == "warning":      # shown above every tab, incl. Risk Tools and forecasts
    st.warning(f"⚠️ **{_val['title']}** " + _val["message"]
               + f" (Index lists as of {_val['index_lists_as_of']}.)")
elif _val and _val["level"] == "info":
    st.info(f"ℹ️ **{_val['title']}** " + _val["message"]
            + f" (Index lists as of {_val['index_lists_as_of']}.)")

# --- automatic forecast: no stored forecast for the latest data -> prepare one ------------------
AUTO_HORIZONS = [1, 5]
tried = st.session_state.setdefault("auto_tried", set())
auto_key = (symbol, dinfo["last_date"])
if auto_key not in tried:
    try:
        have_h = {m["horizon"] for m in api.models(symbol)["models"] if m["is_selected"]}
        stored = api.predictions(symbol, 50)["predictions"]
    except ApiError:
        have_h, stored = set(), []
    fresh = bool(stored) and max(p["as_of_date"] for p in stored) >= dinfo["last_date"]
    if not (set(AUTO_HORIZONS) <= have_h and fresh):
        tried.add(auto_key)  # one automatic attempt per stock per data date (no retry loops)
        box = st.container()
        bar = box.progress(0.0, text=f"Preparing the forecast for {symbol}…")
        try:
            try:
                task_id = api.quick_forecast(symbol, {"horizons": AUTO_HORIZONS})["task_id"]
            except ApiError as e:  # someone is already training this stock: just follow that task
                if e.code != "conflict" or not e.details.get("task_id"):
                    raise
                task_id = e.details["task_id"]
            final = api.wait_for_task(task_id, lambda t: bar.progress(
                float(t["progress"]), text=t["message"] or t["status"]))
            if final["status"] == "succeeded":
                bump()
                st.rerun()
            box.error(f"Automatic forecast failed: {(final['error'] or {}).get('message')}")
        except ApiError as e:
            box.error(f"Automatic forecast failed: {e.message}")

tabs = st.tabs(["Overview", "Stock Explorer", "Model Training", "Price Prediction",
                "Model Performance", "Prediction History", "Nifty 200 Benchmark", "Live Scorecard",
                "Risk Tools"])


def get_models(h: int | None = None) -> list[dict]:
    try:
        return api.models(symbol, h)["models"]
    except ApiError as e:
        st.error(e.message)
        return []


# --------------------------------------------------------------------------------------------
# A. Overview
# --------------------------------------------------------------------------------------------
with tabs[0]:
    last = hist.iloc[-1]
    prev = hist["close"].iloc[-2] if len(hist) > 1 else None
    c1, c2, c3 = st.columns(3)
    c1.metric("Latest close", inr(last["close"]),
              None if prev is None else f"{(last['close'] / prev - 1) * 100:+.2f}% vs prev. session")
    c2.metric("Last trading day in data", hist.index[-1].strftime("%d %b %Y"))
    c3.metric("Data source", dinfo["source"])
    st.caption(f"Cache refreshed {dinfo['fetched_at']} UTC · {dinfo['data_note']}"
               + (" · ⚠️ stale cache" if dinfo.get("stale") else ""))

    st.subheader("Latest forecast")
    try:
        preds = api.predictions(symbol, 50)["predictions"]
    except ApiError as e:
        preds, _ = [], st.error(e.message)
    if not preds:
        st.info("No forecast stored yet. Train models, then generate a forecast in "
                "**Price Prediction**.")
    else:
        latest_asof = max(p["as_of_date"] for p in preds)
        cur = sorted((p for p in preds if p["as_of_date"] == latest_asof
                      and p.get("model_is_selected", True)), key=lambda p: p["horizon"])
        exp_by_h: dict[int, dict] = {}
        for p in preds:  # forecasts from non-selected, non-baseline models (unvalidated)
            if (p["as_of_date"] == latest_asof and not p.get("model_is_selected", True)
                    and not p["model_is_baseline"]):
                exp_by_h[p["horizon"]] = max(p, exp_by_h.get(p["horizon"], p), key=lambda x: x["id"])
        cols = st.columns(min(len(cur), 4))
        for col, p in zip(cols, cur):
            col.metric(f"{p['horizon']}-day forecast (≈ {p['target_date_estimate']})",
                       inr(p["predicted_price"]),
                       f"{p['predicted_change']:+.2f} ₹ ({p['predicted_change_pct']:+.2f}%)")
            rng = ((p.get("interval") or {}).get("ranges") or {}).get("80")
            col.caption(f"Model: {p['algorithm']}" + (" (baseline)" if p["model_is_baseline"] else ""))
            if rng:
                col.markdown(f"**80% range:** {inr(rng['lower'])} – {inr(rng['upper'])}  \n"
                             f"({rng['lower_pct']:+.1f}% … {rng['upper_pct']:+.1f}%)")
        st.caption(f"Forecasts made from data through {latest_asof}. Not actual prices. The 80% range "
                   "is where the close should land about 4 times out of 5 (calibrated: 79.9% "
                   "out-of-sample over 2022-2026); it describes uncertainty, not direction.")
        if any(p["model_is_baseline"] and p["algorithm"] == "naive" for p in cur):
            st.info("**Why is the forecast 'no change'?** For this stock no trained model beat the "
                    "naive baseline (tomorrow = today) in validation, so the validated forecast is "
                    "flat. That is the honest answer, not a missing prediction. The **range** below "
                    "each forecast is the informative part: it shows how far the close is likely to "
                    "move, and it is calibrated and tested out-of-sample.")
        if exp_by_h:
            with st.expander("Best model's price estimate (NOT validated — treat as unreliable)"):
                st.warning("These come from the best-scoring trained model, which did **not** beat "
                           "the naive baseline on out-of-sample data. Shown only so you can see what "
                           "the model would say; its direction calls are close to a coin flip.")
                ecols = st.columns(min(len(exp_by_h), 4))
                for col, h in zip(ecols, sorted(exp_by_h)):
                    p = exp_by_h[h]
                    col.metric(f"{h}-day estimate ({p['algorithm']})", inr(p["predicted_price"]),
                               f"{p['predicted_change']:+.2f} ₹ ({p['predicted_change_pct']:+.2f}%)")

    st.subheader("Selected model (validation metrics)")
    sel = [m for m in get_models() if m["is_selected"]]
    if not sel:
        st.info("No trained models for this stock yet.")
    else:
        st.dataframe(pd.DataFrame([{
            "Horizon (d)": m["horizon"], "Model": m["algorithm"], "MAE ₹": m["metrics"]["mae"],
            "RMSE ₹": m["metrics"]["rmse"], "MAPE %": m["metrics"]["mape"],
            "Directional acc.": m["metrics"]["directional_accuracy"],
            "MAE vs naive %": m["metrics"]["vs_naive"]["mae_improvement_pct"],
            "Trained through": m["train_end"], "OOS rows": m["metrics"]["n"],
        } for m in sorted(sel, key=lambda x: x["horizon"])]), hide_index=True, width="stretch")
        if any(m["algorithm"] == "naive" for m in sel):
            st.warning("For at least one horizon no model beat the naive baseline out-of-sample, "
                       "so 'no change' is the selected forecast there.")

# --------------------------------------------------------------------------------------------
# B. Stock Explorer
# --------------------------------------------------------------------------------------------
with tabs[1]:
    c1, c2 = st.columns([1, 3])
    kind = c1.radio("Chart type", ["Candlestick", "Line"], horizontal=True)
    smas = tuple(c2.multiselect("Moving averages", [5, 10, 20, 50, 100, 200], default=[20, 50]))
    st.plotly_chart(charts.price_chart(hist, kind, smas, symbol), width="stretch")
    if st.button("Force refresh data from provider"):
        with st.spinner("Refreshing…"):
            try:
                api.history(symbol, r_start.isoformat(), r_end.isoformat(), refresh=True)
                bump()
                st.rerun()
            except ApiError as e:
                st.error(e.message)
    show = hist[["open", "high", "low", "close", "adj_close", "volume"]].sort_index(ascending=False)
    st.dataframe(show, width="stretch")
    st.download_button("Download CSV", show.to_csv().encode(), f"{symbol}_ohlcv.csv", "text/csv")

# --------------------------------------------------------------------------------------------
# C. Model Training
# --------------------------------------------------------------------------------------------
with tabs[2]:
    algos = list(health["algorithms"])
    st.caption("Models predict the h-day return; metrics come from chronological expanding-window "
               "walk-forward validation (no shuffling, no leakage). 'naive' is always included.")
    c1, c2 = st.columns(2)
    sel_algos = c1.multiselect("Algorithms", algos, default=algos,
                               help="; ".join(f"{k}: {v}" for k, v in health["algorithms"].items()))
    sel_h = c2.multiselect("Forecast horizons (trading days)", [1, 2, 3, 5, 10, 15, 20], default=[1])
    t_range = st.date_input("Training date range (leave default for all cached data)",
                            value=(r_start, date.today()), key="train_range", max_value=date.today())
    if st.button("Start training", type="primary", disabled=not (sel_algos and sel_h)):
        body = {"algorithms": sel_algos, "horizons": sel_h}
        if isinstance(t_range, tuple) and len(t_range) == 2:
            body["start"], body["end"] = t_range[0].isoformat(), t_range[1].isoformat()
        try:
            acc = api.train(symbol, body)
            bar, msg = st.progress(0.0), st.empty()

            def on_update(t: dict) -> None:
                bar.progress(float(t["progress"]))
                msg.info(f"{t['status']}: {t['message']}")

            final = api.wait_for_task(acc["task_id"], on_update)
            if final["status"] == "succeeded":
                msg.success("Training finished.")
                bump()
            else:
                msg.error(f"Training failed: {(final['error'] or {}).get('message')}")
        except ApiError as e:
            st.error(e.message)

    models_all = get_models()
    if not models_all:
        st.info("No trained models yet. Pick algorithms and press **Start training**.")
    else:
        latest_run = models_all[0]["run_id"]
        run_models = [m for m in models_all if m["run_id"] == latest_run]
        st.subheader("Latest training run")
        for h in sorted({m["horizon"] for m in run_models}):
            ms = [m for m in run_models if m["horizon"] == h]
            st.markdown(f"**Horizon: {h} trading day(s)**")
            st.dataframe(pd.DataFrame([{
                "Model": m["algorithm"] + (" ✅ selected" if m["is_selected"] else ""),
                "MAE ₹": m["metrics"]["mae"], "RMSE ₹": m["metrics"]["rmse"],
                "MAPE %": m["metrics"]["mape"],
                "Directional acc.": m["metrics"]["directional_accuracy"],
                "MAE vs naive %": m["metrics"]["vs_naive"]["mae_improvement_pct"],
                "RMSE vs naive %": m["metrics"]["vs_naive"]["rmse_improvement_pct"],
                "Eligible": m["eligible"], "Why": m["selection_note"],
            } for m in ms]), hide_index=True, width="stretch")
            b1, b2 = st.columns(2)
            b1.plotly_chart(charts.metric_bars(ms, "rmse", f"RMSE by model (h={h})", "₹"),
                            width="stretch", key=f"rmse{h}")
            b2.plotly_chart(charts.metric_bars(ms, "mae", f"MAE by model (h={h})", "₹"),
                            width="stretch", key=f"mae{h}")

# --------------------------------------------------------------------------------------------
# D. Price Prediction
# --------------------------------------------------------------------------------------------
with tabs[3]:
    st.caption("Direct multi-horizon forecasts: each horizon has its own model trained on that "
               "horizon's target. No predicted price is fed back as an input. Uncertainty grows "
               "with horizon.")
    trained_h = sorted({m["horizon"] for m in get_models() if m["is_selected"]})
    if not trained_h:
        st.info("Train models first (Model Training tab).")
    else:
        horizons = st.multiselect("Horizons to forecast (trading days)", trained_h,
                                  default=trained_h[:1])
        refresh = st.checkbox("Refresh market data first", value=True)
        if st.button("Generate forecast", type="primary", disabled=not horizons):
            try:
                with st.spinner("Forecasting…"):
                    st.session_state["fc"] = api.predict(
                        symbol, {"horizons": horizons, "refresh_data": refresh})
                bump()
            except ApiError as e:
                st.error(e.message)
        fc = st.session_state.get("fc")
        if fc and fc["symbol"] == symbol:
            for w in fc["warnings"]:
                st.warning(w)
            cols = st.columns(min(len(fc["predictions"]), 4))
            for col, p in zip(cols, fc["predictions"]):
                col.metric(f"{p['horizon']}-day forecast", inr(p["predicted_price"]),
                           f"{p['predicted_change']:+.2f} ₹ ({p['predicted_change_pct']:+.2f}%)")
                col.caption(f"≈ {p['target_date_estimate']} · {p['algorithm']} · "
                            f"ref close {inr(p['reference_close'])} on {p['as_of_date']}")
                rg = ((p.get("interval") or {}).get("ranges") or {})
                if rg:
                    col.markdown(f"**50%:** {inr(rg['50']['lower'])} – {inr(rg['50']['upper'])}  \n"
                                 f"**80%:** {inr(rg['80']['lower'])} – {inr(rg['80']['upper'])}  \n"
                                 f"**95%:** {inr(rg['95']['lower'])} – {inr(rg['95']['upper'])}")
                else:
                    col.caption("No range: ranges are calibrated for 1- and 5-day horizons only.")
            if any(p["model_is_baseline"] for p in fc["predictions"]):
                st.warning("A baseline model (naive/drift) was selected for some horizons because "
                           "no trained model beat it out-of-sample.")
            st.plotly_chart(charts.forecast_chart(hist, fc["predictions"], symbol), width="stretch")
            for p in fc["predictions"]:
                if p["horizon"] > 1:
                    st.caption(f"⚠️ {p['uncertainty_note']}")
            st.caption(fc["disclaimer"])

# --------------------------------------------------------------------------------------------
# E. Model Performance
# --------------------------------------------------------------------------------------------
with tabs[4]:
    mods = get_models()
    if not mods:
        st.info("No models to evaluate yet.")
    else:
        labels_m = {
            f"#{m['id']} · {m['algorithm']} · h={m['horizon']} · trained through {m['train_end']}"
            + (" · selected" if m["is_selected"] else ""): m for m in mods}
        pick = labels_m[st.selectbox("Model version", list(labels_m))]
        st.warning("**Backtest** — historical out-of-sample predictions from walk-forward "
                   "validation. These are not future forecasts.")
        try:
            bt = pd.DataFrame(api.backtest(symbol, pick["id"])["rows"])
        except ApiError as e:
            bt = pd.DataFrame()
            st.error(e.message)
        if bt.empty:
            st.info("No backtest rows stored.")
        else:
            bt["target_date"] = pd.to_datetime(bt["target_date"])
            title = f"{symbol} · {pick['algorithm']} · {pick['horizon']}-day backtest"
            st.plotly_chart(charts.backtest_chart(bt, title), width="stretch")
            st.plotly_chart(charts.error_chart(bt, "Backtest error over time"), width="stretch")
        m = pick["metrics"]
        mc = st.columns(5)
        mc[0].metric("MAE", inr(m["mae"]))
        mc[1].metric("RMSE", inr(m["rmse"]))
        mc[2].metric("MAPE", pct(m["mape"]))
        mc[3].metric("Directional acc.",
                     "n/a" if m["directional_accuracy"] is None else f"{m['directional_accuracy']:.1%}")
        mc[4].metric("MAE vs naive", pct(m["vs_naive"]["mae_improvement_pct"], signed=True))
        st.caption("Directional accuracy: share of out-of-sample rows where the predicted move "
                   "direction (vs. the reference close) matched the actual one; undefined (n/a) "
                   "for the naive model. MAPE excludes non-positive actuals.")
        st.subheader("Comparison at this horizon")
        same_h = [x for x in mods if x["horizon"] == pick["horizon"] and x["run_id"] == pick["run_id"]]
        if same_h:
            st.plotly_chart(charts.metric_bars(same_h, "rmse", "RMSE (lower is better)", "₹"),
                            width="stretch", key="perf_rmse")
        with st.expander("Training metadata"):
            st.json({k: pick[k] for k in ("version", "algorithm", "horizon", "run_id", "train_start",
                                          "train_end", "n_train", "created_at", "n_features",
                                          "eligible", "selection_note", "is_selected")})
            st.write("Features:", ", ".join(api.models(symbol)["feature_names"]))

# --------------------------------------------------------------------------------------------
# G. Nifty 200 benchmark (static report committed in reports/; independent of the selected stock)
# --------------------------------------------------------------------------------------------
REPORTS = Path(__file__).resolve().parents[1] / "reports"


@st.cache_data(show_spinner=False)
def load_benchmark() -> tuple[dict | None, pd.DataFrame | None]:
    import json
    try:
        summary = json.loads((REPORTS / "benchmark_summary.json").read_text(encoding="utf-8"))
        forecasts = pd.read_csv(REPORTS / "forecasts.csv")
        return summary, forecasts
    except (OSError, ValueError):
        return None, None


with tabs[6]:
    bsum, bfc = load_benchmark()
    if bsum is None:
        st.info("No benchmark report found. Run `python scripts/benchmark_universe.py` to create one.")
    else:
        st.warning("Held-out evaluation over the Nifty 200: models were trained and selected on "
                   f"data *before* the last {bsum['holdout_sessions']} sessions and scored on those "
                   "unseen sessions. Constituents are today's index members (survivorship bias).")
        st.caption(f"{bsum['n_ok']} stocks evaluated, {bsum['n_skipped']} skipped (too little "
                   "history), snapshot of the last benchmark run — not live.")
        rows = []
        for h, s in bsum["horizons"].items():
            sysm, nn = s["system_all_stocks_holdout"], s["system_when_non_naive_selected_holdout"]
            rows.append({
                "Horizon (days)": int(h), "Stocks": s["n_stocks"],
                "Naive median MAPE %": s["per_algorithm_holdout"]["naive"]["median_mape_pct"],
                "System median MAPE %": sysm["median_mape_pct"],
                "Naive selected": s["selected_counts"].get("naive", 0),
                "Non-naive picks": s["selected_non_naive"],
                "…that beat naive": nn["stocks_beating_naive"] if nn else 0,
                "Directional acc. % (non-naive picks)": nn["directional_accuracy_pct"] if nn else None,
            })
        st.subheader("Accuracy of the full system")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("The system falls back to the naive 'no change' forecast unless a model beats it "
                   "in validation. Most picks are naive, and non-naive picks mostly lost to naive on "
                   "unseen data; direction calls are close to a coin flip (50%).")
        h_pick = st.radio("Algorithm comparison for horizon (days)", list(bsum["horizons"]),
                          horizontal=True)
        st.dataframe(pd.DataFrame([{
            "Algorithm": a, "Median MAPE %": m["median_mape_pct"],
            "Median MAE vs naive %": m["median_mae_vs_naive_pct"],
            "Wins vs naive": m["stocks_beating_naive"], "Losses vs naive": m["stocks_losing_to_naive"],
            "Directional acc. %": m["directional_accuracy_pct"],
        } for a, m in bsum["horizons"][h_pick]["per_algorithm_holdout"].items()]),
            hide_index=True, width="stretch")
        st.caption("Each algorithm applied to every stock. Negative 'MAE vs naive' = worse than naive.")

        try:
            import json as _json
            iv = _json.loads((REPORTS / "interval_experiment.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            iv = None
        if iv:
            st.subheader("Forecast-range accuracy (out-of-sample)")
            st.caption(f"{iv['stocks']} stocks, {iv['oos_start']} → {iv['oos_end']}. 'Coverage' = how often the "
                       "actual return landed inside the range; the target equals the nominal level.")
            rows = []
            for h in ("1", "5"):
                for m, label in (("fhs_xgb", "Shipped (volatility model, calibrated)"),
                                 ("uncond", "Baseline (historical quantiles)"),
                                 ("normal_ewma", "Normal curve, EWMA volatility")):
                    mm = iv["horizons"][h]["methods"][m]
                    rows.append({"Horizon (d)": int(h), "Method": label,
                                 "50% range hit %": mm["0.5"]["coverage_pct"],
                                 "80% range hit %": mm["0.8"]["coverage_pct"],
                                 "95% range hit %": mm["0.95"]["coverage_pct"],
                                 "Mean 80% width %": mm["0.8"]["mean_width_pct"]})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.caption("The shipped method hits its nominal levels closely and stays calibrated in calm "
                       "and volatile markets; the normal-curve version over-covers when volatility is "
                       "high. Gains in sharpness over the simple baseline are modest (about 1–2.5%).")

        st.subheader("Latest stored forecasts (all stocks)")
        c1, c2, c3 = st.columns([1, 1, 2])
        hz = c1.radio("Horizon", sorted(bfc["horizon_days"].unique()), horizontal=True, key="bm_h")
        only_nonflat = c2.checkbox("Hide flat (naive) forecasts", value=False)
        q = c3.text_input("Filter by symbol/name", key="bm_q")
        view = bfc[bfc["horizon_days"] == hz]
        if only_nonflat:
            view = view[view["predicted_change_pct"] != 0]
        if q.strip():
            mask = (view["symbol"].str.contains(q.strip(), case=False, na=False)
                    | view["name"].str.contains(q.strip(), case=False, na=False))
            view = view[mask]
        st.dataframe(view.rename(columns={
            "horizon_days": "Horizon (d)", "as_of_date": "As of", "target_date_estimate": "Target (est.)",
            "reference_close": "Reference ₹", "predicted_price": "Forecast ₹",
            "predicted_change_pct": "Change %", "model": "Model",
            "holdout_selected_vs_naive_mae_pct": "Model skill vs naive % (held-out)"}),
            hide_index=True, width="stretch")
        st.caption("Forecasts are model estimates, not actual prices or advice. Rows where the model "
                   "is 'naive' are simply 'no change'.")
        st.download_button("Download forecasts CSV", bfc.to_csv(index=False).encode(),
                           "nifty200_forecasts.csv", "text/csv")

# --------------------------------------------------------------------------------------------
# H. Live scorecard: forward results published by the daily GitHub Actions job (live-data branch)
# --------------------------------------------------------------------------------------------
LIVE_BASE = os.getenv("LIVE_DATA_BASE_URL",
                      "https://raw.githubusercontent.com/yash1709/price-prediction/live-data/live")
LIVE_LOCAL = Path(__file__).resolve().parents[1] / "live"


@st.cache_data(ttl=900, show_spinner=False)
def load_live(base: str) -> dict | None:
    import io
    import json as _json

    import httpx

    def get(name: str) -> str | None:
        try:
            r = httpx.get(f"{base}/{name}", timeout=15, follow_redirects=True)
            if r.status_code == 200:
                return r.text
        except httpx.HTTPError:
            pass
        p = LIVE_LOCAL / name
        return p.read_text(encoding="utf-8") if p.exists() else None

    summary = get("summary.json")
    if summary is None:
        return None
    out = {"summary": _json.loads(summary), "latest": None, "daily": None}
    for key, name in (("latest", "latest_forecasts.csv"), ("daily", "coverage_by_day.csv")):
        text = get(name)
        out[key] = pd.read_csv(io.StringIO(text)) if text else None
    return out


with tabs[7]:
    live = load_live(LIVE_BASE)
    if live is None:
        st.info("The daily job has not published any results yet. It runs on weekdays after the NSE "
                "close (GitHub Actions); the first results appear after its first run.")
    else:
        ls = live["summary"]
        banner = {"ok": st.success, "degraded": st.warning, "failed": st.error}[ls["status"]]
        banner(f"Last run: {ls['last_run_utc']} UTC · data as of {ls['as_of_date']} · "
               f"{ls['symbols_ok']}/{ls['symbols_requested']} stocks fetched · status: {ls['status']}")
        st.markdown(
            "Every weekday after the close, the job stores a forecast for each Nifty 200 stock "
            "**before** its outcome exists, then scores it when the date arrives. The point forecast is "
            "'no change' (the only one that survived testing); the **ranges** are what is being "
            "tested. This is a genuine forward test, not a backtest.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Forecasts recorded", f"{ls['forecasts_total']:,}")
        c2.metric("Matured and scored", f"{ls['outcomes_total']:,}")
        c3.metric("Added in last run", f"{ls['new_forecasts']:,} / {ls['new_outcomes']:,}")

        rows = []
        for h, cv in ls["coverage"].items():
            if cv.get("n"):
                rows.append({"Horizon (d)": int(h), "Scored forecasts": cv["n"],
                             "50% range hit %": cv["inside50_pct"], "80% range hit %": cv["inside80_pct"],
                             "95% range hit %": cv["inside95_pct"],
                             "Mean 80% width %": cv["mean_width80_pct"],
                             "Typical move %": cv["no_change_mape_pct"],
                             "Last 60 sessions, 80% hit %": cv["last_60_sessions"]["inside80_pct"]})
        st.subheader("Live range accuracy (targets: 50 / 80 / 95 %)")
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.caption("Stocks tend to move together, so these samples are not independent and early "
                       "numbers are noisy; judge the trend over weeks, not a single day. 'Typical move' "
                       "is the average absolute return, i.e. the error of the 'no change' forecast.")
        else:
            st.info("No forecast has matured yet. The first 1-day outcomes arrive after the next "
                    "trading session; 5-day outcomes after five sessions.")
        daily = live["daily"]
        if daily is not None and not daily.empty:
            import plotly.graph_objects as go
            fig = go.Figure()
            for h, g in daily.groupby("horizon"):
                g = g.sort_values("actual_date")
                fig.add_trace(go.Scatter(x=g["actual_date"], y=g["in80"].cumsum() / g["n"].cumsum() * 100,
                                         mode="lines", name=f"{h}-day"))
            fig.add_hline(y=80, line_dash="dash", line_color="grey", annotation_text="target 80%")
            fig.update_layout(template="plotly_white", height=340, yaxis_title="% inside 80% range (cumulative)",
                              margin=dict(l=10, r=10, t=30, b=10), legend_orientation="h")
            st.plotly_chart(fig, width="stretch")

        latest = live["latest"]
        if latest is not None and not latest.empty:
            st.subheader("Latest forecasts, all stocks")
            f1, f2 = st.columns([1, 2])
            hz = f1.radio("Horizon (days)", sorted(latest["horizon"].unique()), horizontal=True, key="live_h")
            q = f2.text_input("Filter by symbol or name", key="live_q")
            v = latest[latest["horizon"] == hz].copy()
            if q.strip():
                v = v[v["symbol"].str.contains(q.strip(), case=False, na=False)
                      | v["name"].fillna("").str.contains(q.strip(), case=False, na=False)]
            v["80% low %"] = (v["lo80"] / v["reference_close"] - 1) * 100
            v["80% high %"] = (v["hi80"] / v["reference_close"] - 1) * 100
            st.dataframe(v[["symbol", "name", "as_of_date", "reference_close", "lo80", "hi80", "80% low %",
                            "80% high %", "method"]].rename(columns={
                "symbol": "Symbol", "name": "Name", "as_of_date": "As of", "reference_close": "Latest close ₹",
                "lo80": "80% low ₹", "hi80": "80% high ₹", "method": "Method"}),
                hide_index=True, width="stretch")
            st.download_button("Download latest forecasts CSV", latest.to_csv(index=False).encode(),
                               "live_latest_forecasts.csv", "text/csv")
        if ls.get("n_failed"):
            with st.expander(f"{ls['n_failed']} stock(s) could not be processed in the last run"):
                st.dataframe(pd.DataFrame(ls["failed"]), hide_index=True, width="stretch")

# --------------------------------------------------------------------------------------------
# I. Risk tools (built only on validated pieces: calibrated ranges and volatility; no price calls)
# --------------------------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def load_risk(url: str, sym: str, nonce: int) -> dict:
    return ApiClient(url).risk(sym)


with tabs[8]:
    st.info("These tools describe **how far** a stock or portfolio may move, not which direction. "
            "They are built on the calibrated ranges and volatility estimates tested earlier. "
            "Not investment advice.")
    t_card, t_size, t_port, t_screen = st.tabs(
        ["Stock risk card", "Position size", "Portfolio risk", "Risk screener"])

    with t_card:
        try:
            with st.spinner(f"Computing risk for {symbol}…"):
                rk = load_risk(api_url, symbol, st.session_state.nonce)
        except ApiError as e:
            rk = None
            st.error(e.message)
        if rk:
            v = rk["volatility"]
            st.caption(f"{symbol} · data as of {rk['as_of_date']} · last close {inr(rk['last_close'])}")
            c = st.columns(5)
            c[0].metric("Forecast daily volatility", pct(v["forecast_daily_pct"]))
            c[1].metric("Forecast annualised", pct(v["forecast_annualised_pct"]),
                        help="Daily volatility x √252. Method: " + str(v["forecast_method"]))
            c[2].metric("Realised, last year", pct(v["realised_annualised_pct"].get("252d")))
            c[3].metric("Beta vs Nifty (1y)", "—" if rk["beta_vs_nifty_1y"] is None
                        else f"{rk['beta_vs_nifty_1y']:.2f}",
                        help="1.0 moves like the Nifty on average; above 1 swings more.")
            c[4].metric("Down from 1-year peak", pct(rk["drawdown"]["current_from_1y_peak_pct"]))
            st.subheader("Expected move ranges (calibrated)")
            rows = []
            for h, m in rk["expected_moves"].items():
                for lv in ("50", "80", "95"):
                    x = m[lv]
                    rows.append({"Horizon (days)": int(h), "Likelihood inside": f"{lv}%",
                                 "Low ₹": x["lower"], "High ₹": x["upper"],
                                 "Low %": x["lower_pct"], "High %": x["upper_pct"]})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.caption("Tested out-of-sample on ~190 stocks: the 80% range contained the outcome 79.9% of "
                       "the time; the 50% and 95% ranges 49.8% and 94.9%.")
            bd = rk["bad_day_thresholds"]
            if bd:
                st.subheader("Bad-day thresholds (1-day)")
                b1, b2 = st.columns(2)
                for col, key, title in ((b1, "1_in_10", "1-in-10 bad day"), (b2, "1_in_40", "1-in-40 bad day")):
                    col.metric(title, f"{bd[key]['threshold_pct']:+.2f}%", inr(bd[key]["threshold_price"]),
                               delta_color="off")
                    col.caption(bd[key]["meaning"])
            d = rk["drawdown"]
            e = rk["extremes_1y"]
            r52 = rk["range_52w"]
            st.subheader("History at a glance")
            st.dataframe(pd.DataFrame([
                {"Measure": "Worst / best single day, last year", "Value": f"{e['worst_day_pct']:+.1f}% / {e['best_day_pct']:+.1f}%"},
                {"Measure": "Largest peak-to-trough fall: 1 year / 3 years", "Value": f"{d['max_1y_pct']:.1f}% / {d['max_3y_pct']:.1f}%"},
                {"Measure": "52-week low – high", "Value": f"{inr(r52['low'])} – {inr(r52['high'])}"},
                {"Measure": "Position in 52-week range", "Value": pct(r52["position_pct"])},
                {"Measure": "Average daily traded value (20d)", "Value": f"₹{rk['avg_daily_traded_value_inr'] / 1e7:,.1f} crore"},
            ]), hide_index=True, width="stretch")
            for n in rk["notes"]:
                st.caption("• " + n)

    with t_size:
        st.markdown("How big a position can **" + symbol + "** be if you only want to lose a set share of "
                    "your portfolio on a bad move? The bad-move size comes from the calibrated ranges.")
        s1, s2 = st.columns(2)
        pv = s1.number_input("Portfolio value (₹)", min_value=10_000.0, value=1_000_000.0, step=50_000.0)
        rp = s2.slider("Maximum loss on a bad move (% of portfolio)", 0.1, 5.0, 1.0, 0.1)
        s3, s4 = st.columns(2)
        hz = s3.radio("Horizon", [1, 5], horizontal=True, format_func=lambda x: f"{x} day(s)", key="ps_h")
        tl = s4.radio("How bad?", ["1_in_10", "1_in_40"], horizontal=True, key="ps_t",
                      format_func=lambda x: "1-in-10 bad move" if x == "1_in_10" else "1-in-40 bad move")
        try:
            ps = api.position_size({"symbol": symbol, "portfolio_value": pv, "risk_pct": rp,
                                    "horizon": hz, "tail": tl})
            m1, m2, m3 = st.columns(3)
            m1.metric("Largest position", inr(ps["position_value_inr"]),
                      f"{ps['position_value_inr'] / pv * 100:.1f}% of portfolio", delta_color="off")
            m2.metric("Shares", f"{ps['shares']:,}")
            m3.metric("Loss if the bad move happens", inr(ps["loss_if_bad_move_inr"]),
                      f"{ps['loss_if_bad_move_pct_of_portfolio']:.2f}% of portfolio", delta_color="off")
            st.caption(f"Bad move = {ps['bad_move_pct']:+.2f}% over {hz} day(s), which happens "
                       f"{ps['bad_move_frequency']}. Formula: {ps['formula']}.")
            if ps["capped_at_portfolio"]:
                st.warning("Your risk budget is large enough that the position is capped at the whole "
                           "portfolio (no leverage).")
            with st.expander("Caveats"):
                for cv in ps["caveats"]:
                    st.write("• " + cv)
        except ApiError as e:
            st.error(e.message)

    with t_port:
        st.markdown("Enter holdings as `SYMBOL, amount in ₹` (one per line, up to 25).")
        txt = st.text_area("Holdings", "RELIANCE, 100000\nTCS, 100000\nHDFCBANK, 100000\nITC, 100000",
                           height=140)
        if st.button("Analyse portfolio", type="primary"):
            import re
            hs, bad = [], []
            for ln in [x for x in txt.splitlines() if x.strip()]:
                mt = re.match(r"^\s*([A-Za-z0-9&\-_.]+)[,\s:]+([\d,\.]+)\s*$", ln)
                try:
                    hs.append({"symbol": mt.group(1), "amount": float(mt.group(2).replace(",", ""))})
                except (AttributeError, ValueError):
                    bad.append(ln)
            if bad:
                st.error("Could not read these lines: " + "; ".join(bad))
            else:
                try:
                    with st.spinner("Fetching data and computing…"):
                        st.session_state["pr"] = api.portfolio_risk(hs)
                except ApiError as e:
                    st.session_state.pop("pr", None)
                    st.error(e.message)
        pr = st.session_state.get("pr")
        if pr:
            p = pr["portfolio"]
            st.caption(f"Total {inr(pr['total_amount_inr'])} · data as of {pr['as_of_date']}")
            c = st.columns(5)
            c[0].metric("Daily volatility", pct(p["daily_volatility_pct"]))
            c[1].metric("Annualised", pct(p["annualised_volatility_pct"]))
            c[2].metric("Diversification ratio", f"{p['diversification_ratio']:.2f}",
                        help="Average holding volatility ÷ portfolio volatility. 1.0 = no diversification benefit.")
            c[3].metric("Effective # of holdings", f"{p['effective_number_of_holdings']:.1f}")
            c[4].metric("Avg. pairwise correlation",
                        "—" if p["average_pairwise_correlation"] is None else f"{p['average_pairwise_correlation']:.2f}")
            for w in pr["warnings"]:
                st.warning(w)
            st.plotly_chart(charts.risk_contrib_chart(pr["holdings"]), width="stretch")
            st.dataframe(pd.DataFrame([{
                "Symbol": h["symbol"], "Amount ₹": h["amount_inr"], "Weight %": h["weight_pct"],
                "Daily vol %": h["daily_volatility_pct"], "Annual vol %": h["annualised_volatility_pct"],
                "Share of risk %": h["risk_contribution_pct"]} for h in pr["holdings"]]),
                hide_index=True, width="stretch")
            if len(pr["holdings"]) > 1:
                st.plotly_chart(charts.corr_heatmap(pr["correlation_matrix"]), width="stretch")
            st.info("**No portfolio forecast range is shown.** I tested one on 300 random portfolios; it "
                    "did not stay calibrated at the 5-day horizon in every year, so it is deliberately not "
                    "offered. The figures above are volatility estimates (method: " + pr["method"] + ").")

    with t_screen:
        lv = load_live(LIVE_BASE)
        if lv is None or lv["latest"] is None:
            st.info("The risk screener uses the daily live data, which is not published yet.")
        else:
            lt = lv["latest"]
            lt = lt[lt["horizon"] == 1].copy()
            lt["Range width % (80%)"] = (lt["hi80"] - lt["lo80"]) / lt["reference_close"] * 100
            lt["Bad day % (1-in-10)"] = (lt["lo80"] / lt["reference_close"] - 1) * 100
            lt["Daily vol % (fcst)"] = lt["Range width % (80%)"] / (2 * 1.2816)  # implied by the range
            st.caption(f"Nifty 200 as of {lv['summary']['as_of_date']}, ranked by the calibrated 1-day "
                       "80% range width: wider = the stock is expected to move more.")
            cols = ["symbol", "name", "reference_close", "Daily vol % (fcst)", "Range width % (80%)",
                    "Bad day % (1-in-10)"]
            ren = {"symbol": "Symbol", "name": "Name", "reference_close": "Close ₹"}
            a, b = st.columns(2)
            a.markdown("**Calmest 15**")
            a.dataframe(lt.nsmallest(15, "Range width % (80%)")[cols].rename(columns=ren),
                        hide_index=True, width="stretch")
            b.markdown("**Most volatile 15**")
            b.dataframe(lt.nlargest(15, "Range width % (80%)")[cols].rename(columns=ren),
                        hide_index=True, width="stretch")
            qq = st.text_input("Search the full table", key="scr_q")
            full = lt.sort_values("Range width % (80%)")
            if qq.strip():
                full = full[full["symbol"].str.contains(qq.strip(), case=False, na=False)
                            | full["name"].fillna("").str.contains(qq.strip(), case=False, na=False)]
            st.dataframe(full[cols].rename(columns=ren), hide_index=True, width="stretch")

# --------------------------------------------------------------------------------------------
# F. Prediction History
# --------------------------------------------------------------------------------------------
with tabs[5]:
    if st.button("Record actual closes for matured forecasts"):
        try:
            with st.spinner("Updating outcomes…"):
                res = api.update_outcomes(symbol)
            st.success(f"Recorded {res['newly_resolved']} new outcome(s); "
                       f"{res['still_pending']} still pending.")
            for w in res["warnings"]:
                st.warning(w)
        except ApiError as e:
            st.error(e.message)
    try:
        hist_preds = api.predictions(symbol, 1000)["predictions"]
        perf = api.performance(symbol)
    except ApiError as e:
        hist_preds, perf = [], {"by_model": [], "outcomes": []}
        st.error(e.message)
    if not hist_preds:
        st.info("No stored forecasts yet.")
    else:
        rows = [{
            "Made from (as of)": p["as_of_date"], "Horizon (d)": p["horizon"],
            "Target (est.)": p["target_date_estimate"], "Model": p["algorithm"],
            "Model version": p["model_version"], "Reference ₹": p["reference_close"],
            "Forecast ₹": p["predicted_price"],
            "80% range ₹": (lambda r: f"{r['lower']:,.2f} – {r['upper']:,.2f}" if r else None)(
                ((p.get("interval") or {}).get("ranges") or {}).get("80")),
            "Actual ₹": (p["outcome"] or {}).get("actual_close"),
            "Inside 80%?": (p["outcome"] or {}).get("within_range_80"),
            "Error ₹": (p["outcome"] or {}).get("error"),
            "Error %": (p["outcome"] or {}).get("pct_error"),
            "Direction ✓": (p["outcome"] or {}).get("direction_correct"),
            "Status": "resolved" if p["outcome"] else "pending",
        } for p in hist_preds]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        resolved = [dict(as_of_date=p["as_of_date"], horizon=p["horizon"],
                         pct_error=p["outcome"]["pct_error"]) for p in hist_preds if p["outcome"]]
        if resolved:
            st.plotly_chart(charts.live_error_chart(pd.DataFrame(resolved)), width="stretch")
        cov_rows = [r for r in perf["by_model"] if r.get("live_range_coverage")]
        if cov_rows:
            n_iv = sum(r["live_range_coverage"]["n"] for r in cov_rows)
            in80 = sum(r["live_range_coverage"]["inside_80_pct"] * r["live_range_coverage"]["n"]
                       for r in cov_rows) / n_iv
            st.metric("Live range check: actual close inside the 80% range", f"{in80:.0f}%",
                      help=f"{n_iv} matured forecast(s). Target ≈ 80%; small samples are noisy.")
        if perf["by_model"]:
            st.subheader("Realised accuracy by model version")
            st.dataframe(pd.DataFrame([{
                "Model version": r["model_version"], "Horizon": r["horizon"],
                "N": r["metrics"]["n"], "MAE ₹": r["metrics"]["mae"],
                "Naive MAE ₹ (same rows)": r["naive_metrics"]["mae"],
                "Beats naive": r["beats_naive_mae"], "Note": r["note"],
            } for r in perf["by_model"]]), hide_index=True, width="stretch")
        elif hist_preds:
            st.caption("No forecast has matured yet; realised accuracy appears once actual "
                       "closes are available.")
