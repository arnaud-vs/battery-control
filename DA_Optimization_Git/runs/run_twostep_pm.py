# run_twostep_pm.py
# Per-minute version of run_twostep.py: decisions every minute on the refined per-minute ensemble
# forecast, settled at the quarter-hour imbalance price (price_period_minutes=15 decouples the
# 1-min decision step from the 15-min price/horizon period -- the optimizer/LP is unchanged).
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import pandas as pd


def run_one(terminal_penalty: float, model: str, granularity: str = "pm") -> None:
    """granularity='pm'  -> re-decide every minute on the refined forecast (per-minute re-dispatch).
       granularity='qh'  -> QH-MPC baseline: decide once at the QH start and hold for the whole QH.
    Both use the SAME forecast model and realized QH prices, so the difference is the value of
    intra-QH information."""
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    import src.config_loader as config_loader
    import util.read_price as read_price
    import src.ip.optimizations as optimizations

    battery, markets = config_loader.load_config(PROJECT_ROOT / "configs" / "battery_config.yaml")
    ip_cfg = markets["ip"]  # dt_minutes stays 15 -> LP plans QH energy; per-minute is the decision step

    # Per-minute ensemble forecast (index = per-minute decision times, cols qh1..qh8 = QH-horizon
    # prices refined each minute). Built by two_step_forecast.py and synced to results_forecast/.
    pm_path = PROJECT_ROOT.parent / "results_forecast" / \
        "ensembleTOT_forecast_DATAbasic+market+other_pm_CW8736_RF720_LB1_HZ2_1.csv"
    ip_det = pd.read_csv(pm_path, index_col=0)
    ip_det.index = pd.to_datetime(ip_det.index)
    ip_det = ip_det[[f"qh{i}" for i in range(1, 9)]]  # keep the 8 horizon columns

    if granularity == "qh":
        # QH-MPC baseline: only the QH-start rows (minute 0 = before any intra-QH reveal -> the pure
        # QH forecast). Classic path (price_period_minutes=None): decide at QH start, commit full QH.
        ip_det = ip_det[(ip_det.index.minute % 15) == 0]
        price_period = None
    else:
        price_period = 15  # decouple: per-minute decisions, per-QH prices

    # Realized QH imbalance price (same series the QH case study settles on). Worker floors each
    # per-minute decision timestamp to its QH for settlement.
    ip_real = read_price.read_ip_real_prices(freq="15min", keep_extra_columns=False)["Price"]

    start_date, end_date = "2023-01-01", "2023-12-30"
    print(f"[START] {granularity} | model={model} | TP={terminal_penalty}")

    optimizations.run_ip_rolling_ce_models(
        battery=battery,
        market=ip_cfg,
        forecasts={model: ip_det},
        real_price_series=ip_real,
        price_source=["forecast"],
        start=start_date,
        end=end_date,
        terminal_target_kwh=battery.energy_kwh * 0.5,
        terminal_penalty=terminal_penalty,
        terminal_penalty_mode="L2",
        solver_name="scip",
        save=True,
        tag=f"{granularity}_v1",
        cycle_penalty_eur_per_mwh=0,
        price_period_minutes=price_period,
    )
    print(f"[DONE ] {granularity} | model={model} | TP={terminal_penalty}")


if __name__ == "__main__":
    for tp in [0.01]:
        for m in ["ensembleTOT_pm"]:
            run_one(tp, m, granularity="qh")  # baseline: decide at QH start, hold
            run_one(tp, m, granularity="pm")  # per-minute re-dispatch