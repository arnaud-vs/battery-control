# single_copula.py
import os
import sys
from pathlib import Path
# Keep numpy/BLAS from spawning extra threads; let Gurobi have the CPU.
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

# Pick ONE risk configuration:
alpha = 0.95
lambda_cvar = 0.1   # or 0.1

print(f"Running copula: alpha={alpha}, lambda_cvar={lambda_cvar}")

res = optimizations.run_ip_rolling_prob_models(
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
    scenario_method="copula",
    n_scenarios=100,
    lam_corr=0.995,
    save=True,
    tag="v1",
    cycle_penalty_eur_per_mwh=0,
    solver_options={"Threads": 8},
    solver_name="gurobi",
)

