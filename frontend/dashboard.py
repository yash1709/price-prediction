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
                "Model Performance", "Prediction History", "Nifty 200 Benchmark"])


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
            col.caption(f"Model: {p['algorithm']}" + (" (baseline)" if p["model_is_baseline"] else ""))
        st.caption(f"Forecasts made from data through {latest_asof}. Not actual prices.")
        if any(p["model_is_baseline"] and p["algorithm"] == "naive" for p in cur):
            st.info("**Why is the forecast 'no change'?** For this stock no trained model beat the "
                    "naive baseline (tomorrow = today) in validation, so the validated forecast is "
                    "flat. That is the honest answer, not a missing prediction.")
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
            "Forecast ₹": p["predicted_price"], "Actual ₹": (p["outcome"] or {}).get("actual_close"),
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
