# simple_prob.py
import os
import sys
from pathlib import Path
from joblib import Parallel, delayed

os.environ["GRB_LICENSE_FILE"] = r"C:\Program Files\gurobi\license\gurobi.lic"

def run_one(alpha: float, lambda_cvar: float, *, gurobi_threads: int) -> tuple[float, float]:
    """
    One probabilistic rolling run for a given (alpha, lambda_cvar).
    """

    # Limit numeric libs; let Gurobi use the threads we choose.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    import src.config_loader as config_loader
    import util.read_price as read_price
    import src.ip.optimizations as optimizations

    battery, markets = config_loader.load_config(PROJECT_ROOT / "configs" / "battery_config.yaml")
    ip_cfg = markets["ip"]

    ip_prob_qr = read_price.read_ip_forecast(model="QR", kind="probabilistic", freq="15min")
    ip_real = read_price.read_ip_real_prices(freq="15min", keep_extra_columns=False)["Price"]

    start_date = "2023-01-01"
    end_date = "2023-12-30"
    terminal_penalty = 0.1

    print(f"[START] alpha={alpha} lambda_cvar={lambda_cvar} threads={gurobi_threads}")

    # If your run_* function exposes solver options, use them.
    # Common patterns are solver_options=..., gurobi_options=..., or solver_kwds=...
    # Try solver_options first; if your function doesn't accept it, I'll show a fallback below.
    optimizations.run_ip_rolling_prob_models(
        battery=battery,
        market=ip_cfg,
        forecasts={"qr": ip_prob_qr},
        real_price_series=ip_real,
        risks=[{"alpha": alpha, "lambda_cvar": lambda_cvar}],
        start=start_date,
        end=end_date,
        terminal_target_kwh=battery.energy_kwh * 0.5,
        terminal_penalty=terminal_penalty,
        terminal_penalty_mode="L1",
        scenario_method="quantile_paths",
        save=True,
        tag="v1",
        cycle_penalty_eur_per_mwh=0,
        solver_options={"Threads": gurobi_threads},
        solver_name = "gurobi",
    )

    print(f"[DONE ] alpha={alpha} lambda_cvar={lambda_cvar}")
    return alpha, lambda_cvar


def main():
    alpha = 0.95
    configs = [
        (alpha, 0.1),
        (alpha, 0.3),
    ]

    # 10 vCPUs -> 2 jobs × 5 threads each
    gurobi_threads_per_job = 5

    Parallel(n_jobs=2, backend="loky", verbose=10)(
        delayed(run_one)(a, lam, gurobi_threads=gurobi_threads_per_job) for (a, lam) in configs
    )


if __name__ == "__main__":
    main()
