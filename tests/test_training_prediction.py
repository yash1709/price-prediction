"""Training, persistence, prediction, outcomes and database behaviour (mocked market data)."""
from datetime import date

import joblib
import numpy as np
import pytest
from sqlalchemy.exc import IntegrityError

from app.core.errors import InsufficientDataError, NoModelError
from app.data.ingestion import ensure_data
from app.database.models import Prediction
from app.models.prediction import estimate_target_date, predict_for_symbol
from app.models.registry import ALGORITHMS
from app.models.training import train_symbol
from app.services.outcomes import performance_summary, resolve_outcomes
from tests.conftest import make_prices


@pytest.fixture
def loaded(ctx, provider):
    with ctx.repo() as repo:
        ensure_data(repo, provider, ctx.settings, "TESTCO", start=date(2020, 1, 1))
    return ctx


def test_train_persists_models_artifacts_metrics_and_backtests(loaded):
    with loaded.repo() as repo:
        run = train_symbol(repo, loaded.settings, "TESTCO", horizons=[1, 5])
        assert run.status == "succeeded"
        mvs = repo.list_model_versions("TESTCO", run_id=run.id)
        assert {m.algorithm for m in mvs} == set(ALGORITHMS)
        assert len(mvs) == len(ALGORITHMS) * 2
        for h in (1, 5):
            assert sum(m.is_selected for m in mvs if m.horizon == h) == 1
        for m in mvs:
            art = joblib.load(m.artifact_path)          # fitted model + pipeline persisted
            assert art["feature_names"] == m.feature_names and art["horizon"] == m.horizon
            assert {"mae", "rmse", "mape", "directional_accuracy", "vs_naive"} <= set(m.metrics)
            assert m.train_start < m.train_end and m.n_train > 0 and m.version
            bt = repo.backtest_rows(m.id)
            assert len(bt) == m.metrics["n"] and all(b.target_date > b.as_of_date for b in bt)
        naive = next(m for m in mvs if m.algorithm == "naive" and m.horizon == 1)
        assert naive.metrics["vs_naive"]["mae_improvement_pct"] == pytest.approx(0.0)


def test_training_without_data_or_with_too_little_data_fails_cleanly(ctx, provider):
    with ctx.repo() as repo:
        repo.ensure_stock("EMPTY")
        with pytest.raises(InsufficientDataError):
            train_symbol(repo, ctx.settings, "EMPTY")
    provider.frames["SHORT"] = make_prices(120)
    with ctx.repo() as repo:
        ensure_data(repo, provider, ctx.settings, "SHORT", start=date(2020, 1, 1))
        with pytest.raises(InsufficientDataError):
            train_symbol(repo, ctx.settings, "SHORT", algorithms=["ridge"])
        from app.database.models import TrainingRun
        assert repo.s.query(TrainingRun).filter_by(symbol="SHORT").one().status == "failed"


def test_predict_requires_a_model(loaded):
    with loaded.repo() as repo:
        with pytest.raises(NoModelError):
            predict_for_symbol(repo, "TESTCO", [1])


def test_prediction_is_valid_stored_and_idempotent(loaded, prices):
    with loaded.repo() as repo:
        train_symbol(repo, loaded.settings, "TESTCO", algorithms=["ridge", "random_forest"],
                     horizons=[1, 5])
        out, _ = predict_for_symbol(repo, "TESTCO", [1, 5], today=date(2024, 7, 1))
        assert [p["horizon"] for p in out] == [1, 5]
        last_close = float(prices["close"].iloc[-1])
        for p in out:
            assert np.isfinite(p["predicted_price"]) and p["predicted_price"] > 0
            assert p["reference_close"] == pytest.approx(last_close)
            assert p["as_of_date"] == prices.index[-1].date().isoformat()
            assert p["predicted_change"] == pytest.approx(p["predicted_price"] - last_close)
            assert p["kind"] == "forecast" and p["method"] == "direct"
        assert "uncertainty" in out[1]["uncertainty_note"]
        again, _ = predict_for_symbol(repo, "TESTCO", [1, 5], today=date(2024, 7, 1))
        assert [p["id"] for p in again] == [p["id"] for p in out]
        assert repo.s.query(Prediction).count() == 2


def test_stale_data_warning(loaded):
    with loaded.repo() as repo:
        train_symbol(repo, loaded.settings, "TESTCO", algorithms=["ridge"], horizons=[1])
        _, warnings = predict_for_symbol(repo, "TESTCO", [1], today=date(2024, 8, 30))
        assert any("refresh data" in w for w in warnings)


def test_predictions_survive_retraining_and_keep_their_model_version(loaded):
    with loaded.repo() as repo:
        train_symbol(repo, loaded.settings, "TESTCO", algorithms=["ridge"], horizons=[1])
        first, _ = predict_for_symbol(repo, "TESTCO", [1])
        old_mv = first[0]["model_version_id"]
        train_symbol(repo, loaded.settings, "TESTCO", algorithms=["ridge"], horizons=[1])
        second, _ = predict_for_symbol(repo, "TESTCO", [1])
        assert second[0]["model_version_id"] != old_mv
        preds = repo.list_predictions("TESTCO")
        assert {p.model_version_id for p in preds} == {old_mv, second[0]["model_version_id"]}
        assert repo.get_model_version(old_mv) is not None              # old version still stored
        assert not repo.get_model_version(old_mv).is_selected
        assert repo.selected_model("TESTCO", 1).id == second[0]["model_version_id"]


def test_duplicate_prediction_rejected_by_db(loaded):
    with loaded.repo() as repo:
        train_symbol(repo, loaded.settings, "TESTCO", algorithms=["ridge"], horizons=[1])
        p, _ = predict_for_symbol(repo, "TESTCO", [1])
        repo.s.commit()
        dup = Prediction(symbol="TESTCO", model_version_id=p[0]["model_version_id"], horizon=1,
                         as_of_date=date.fromisoformat(p[0]["as_of_date"]),
                         target_date_est=date(2030, 1, 1), reference_close=1, predicted_price=1,
                         predicted_change=0, predicted_change_pct=0)
        with pytest.raises(IntegrityError):
            repo.add_prediction(dup)
        repo.s.rollback()


def test_bar_upsert_prevents_duplicates(loaded, prices):
    with loaded.repo() as repo:
        n = len(repo.load_bars("TESTCO"))
        repo.upsert_bars("TESTCO", prices.tail(10), "fake")
        assert len(repo.load_bars("TESTCO")) == n


def test_outcomes_are_recorded_once_actual_close_exists(ctx, provider):
    full = make_prices(710)
    provider.frames["TESTCO"] = full.iloc[:-10]
    with ctx.repo() as repo:
        ensure_data(repo, provider, ctx.settings, "TESTCO", start=date(2020, 1, 1))
        train_symbol(repo, ctx.settings, "TESTCO", algorithms=["ridge"], horizons=[1, 5])
        preds, _ = predict_for_symbol(repo, "TESTCO", [1, 5])
        as_of = full.index[-11]
        assert resolve_outcomes(repo, "TESTCO") == 0                    # nothing has happened yet
        # market moves on: 10 more sessions arrive
        provider.frames["TESTCO"] = full
        ensure_data(repo, provider, ctx.settings, "TESTCO", start=date(2020, 1, 1), force_refresh=True)
        assert resolve_outcomes(repo, "TESTCO") == 2
        assert resolve_outcomes(repo, "TESTCO") == 0                    # idempotent
        by_h = {p.horizon: p for p in repo.list_predictions("TESTCO")}
        for h, p in by_h.items():
            o = p.outcome
            assert o.actual_date == full.index[full.index.get_loc(as_of) + h].date()
            assert o.actual_close == pytest.approx(full["close"].iloc[full.index.get_loc(as_of) + h])
            assert o.error == pytest.approx(p.predicted_price - o.actual_close)
            assert o.pct_error == pytest.approx(o.error / o.actual_close * 100)
        summary = performance_summary(repo, "TESTCO")
        assert {s["horizon"] for s in summary} == {1, 5} and summary[0]["metrics"]["n"] == 1


def test_target_date_estimate_skips_weekends():
    assert estimate_target_date(date(2024, 6, 28), 1) == date(2024, 7, 1)    # Fri -> Mon
    assert estimate_target_date(date(2024, 6, 26), 5) == date(2024, 7, 3)
