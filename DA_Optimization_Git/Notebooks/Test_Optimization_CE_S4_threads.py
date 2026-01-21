import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
# Always resolve project root relative to this file
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.config_loader as config_loader
import util.read_price as read_price
import src.ip.optimizations as optimizations

start_date = "2023-01-01"
end_date = "2023-12-31"

cycle_penalty_grid = [5.0, 10.0, 20.0, 40.0]
fixed_L1_terminal_penalty = 0.01

def run_one(cyc: float):
    # Load inside the worker to avoid pickling large DataFrames
    battery, markets = config_loader.load_config(PROJECT_ROOT / "configs" / "battery_config.yaml")
    ip_cfg = markets["ip"]

    ip_det_qr = read_price.read_ip_forecast(model="QR", kind="deterministic", freq="15min")
    ip_real = read_price.read_ip_real_prices(freq="15min", keep_extra_columns=False)["Price"]

    common_kwargs = dict(
        battery=battery,
        market=ip_cfg,
        forecasts={"qr": ip_det_qr},
        real_price_series=ip_real,
        price_source=["forecast"],
        start=start_date,
        end=end_date,
        terminal_target_kwh=battery.energy_kwh * 0.5,
        solver_name="gurobi",
        save=True,
        solver_options={"Threads": 4, "OutputFlag": 0},  # each process uses 5 threads
    )

    optimizations.run_ip_rolling_ce_models(
        **common_kwargs,
        terminal_penalty=float(fixed_L1_terminal_penalty),
        terminal_penalty_mode="L1",
        cycle_penalty_eur_per_mwh=float(cyc),
        # tag=tag,
    )
    return cyc

if __name__ == "__main__":
    results = {}
    with ProcessPoolExecutor(max_workers=2) as ex:
        futures = [ex.submit(run_one, cyc) for cyc in cycle_penalty_grid]
        for f in as_completed(futures):
            f.result()

    # Example access:
    # df = results["L1_tp=0.01_cyc=10.0"].history
