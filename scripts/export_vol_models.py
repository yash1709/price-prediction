"""Export the volatility models and interval calibration produced by interval_experiment.py into
`app/models/pretrained/` (portable XGBoost JSON + calibration JSON) for the application to ship.

Usage: python scripts/export_vol_models.py     (run after scripts/interval_experiment.py)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402

REPORTS = ROOT / "reports"
OUT = ROOT / "app" / "models" / "pretrained"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    calib = json.loads((REPORTS / "interval_calibration_candidates.json").read_text(encoding="utf-8"))
    exp = json.loads((REPORTS / "interval_experiment.json").read_text(encoding="utf-8"))
    meta = {"created": calib["created"], "levels": calib["levels"], "oos_start": exp["oos_start"],
            "oos_end": exp["oos_end"], "stocks": exp["stocks"], "selection": exp["selection"],
            "horizons": {}}
    for h, info in calib["horizons"].items():
        art = joblib.load(REPORTS / f"vol_model_h{h}_experiment.joblib")
        art["model"].save_model(str(OUT / f"vol_model_h{h}.json"))
        meta["horizons"][h] = {
            "features": art["features"], "eps": art["eps"],
            "calibration_window": info["calibration_window"], "z_quantiles": info["z_quantiles"],
            "validation": {m: {lv: {"coverage_pct": exp["horizons"][h]["methods"][m][lv]["coverage_pct"]}
                               for lv in ("0.5", "0.8", "0.95")} for m in exp["horizons"][h]["methods"]},
        }
        print(f"h={h}: exported model ({(OUT / f'vol_model_h{h}.json').stat().st_size // 1024} KB), "
              f"{len(art['features'])} features")
    (OUT / "interval_calibration.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("wrote", OUT / "interval_calibration.json")


if __name__ == "__main__":
    main()
