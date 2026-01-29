# simple.py
import os
import sys
from pathlib import Path
from joblib import Parallel, delayed


# Keep each process single-threaded internally (prevents oversubscription)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def run_one(price_source: str, terminal_penalty: float) -> None:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    import src.config_loader as config_loader
    import util.read_price as read_price
    import src.ip.optimizations as optimizations

    battery, markets = config_loader.load_config(PROJECT_ROOT / "configs" / "battery_config.yaml")
    ip_cfg = markets["ip"]

    ip_det_qr = read_price.read_ip_forecast(model="QR", kind="deterministic", freq="15min")
    ip_real = read_price.read_ip_real_prices(freq="15min", keep_extra_columns=False)["Price"]

    start_date = "2023-01-01"
    end_date = "2023-12-30"

    print(f"[START] {price_source} | TP={terminal_penalty}")

    optimizations.run_ip_rolling_ce_models(
        battery=battery,
        market=ip_cfg,
        forecasts={"qr": ip_det_qr},
        real_price_series=ip_real,
        price_source=[price_source],  # single source => unique filename
        start=start_date,
        end=end_date,
        terminal_target_kwh=battery.energy_kwh * 0.5,
        terminal_penalty=terminal_penalty,
        terminal_penalty_mode="L2",
        solver_name="gurobi",
        save=True,
        tag="v1",
        cycle_penalty_eur_per_mwh=0,
    )

    print(f"[DONE ] {price_source} | TP={terminal_penalty}")


if __name__ == "__main__":
    penalties = [0.001, 0.003, 0.01, 0.03]
    sources = ["forecast", "perfect_foresight"]

    jobs = [(s, tp) for s in sources for tp in penalties]  # 8 independent runs

    Parallel(n_jobs=8, backend="loky", verbose=10)(
        delayed(run_one)(s, tp) for (s, tp) in jobs
    )
