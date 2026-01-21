import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp


# ------------------------------------------------------------
# Robust project root (relative to THIS file, not cwd)
#   .../Notebooks/your_script.py -> parents[1] == project root
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.config_loader as config_loader
import util.read_price as read_price
import src.ip.optimizations as optimizations


start_date = "2023-01-01"
end_date = "2023-12-31"

# grids
terminal_penalty_L1 = [0.01, 0.03]     # €/kWh
terminal_penalty_L2 = [1e-5, 3e-5]     # L2 scale (small!)
cycle_penalty = 0.0                   # fixed here


def _make_jobs():
    jobs = []
    # (A) L1 sweep
    for tp in terminal_penalty_L1:
        jobs.append(
            {
                "mode": "L1",
                "terminal_penalty": float(tp),
                "cycle_penalty": float(cycle_penalty),
                "label": f"L1_tp={tp}_cyc=0",
            }
        )
    # (B) L2 sweep
    for tp in terminal_penalty_L2:
        jobs.append(
            {
                "mode": "L2",
                "terminal_penalty": float(tp),
                "cycle_penalty": float(cycle_penalty),
                "label": f"L2_tp={tp}_cyc=0",
            }
        )
    return jobs


def run_one(job: dict, *, threads_per_worker: int = 1) -> dict:
    """
    Worker: load inputs locally, run one configuration, save parquet via optimizations.py.
    Return a small summary dict to parent.
    """
    pid = os.getpid()

    # Load inside worker (avoid pickling big frames)
    battery, markets = config_loader.load_config(PROJECT_ROOT / "configs" / "battery_config.yaml")
    ip_cfg = markets["ip"]

    ip_det_qr = read_price.read_ip_forecast(model="QR", kind="deterministic", freq="15min")
    ip_real = read_price.read_ip_real_prices(freq="15min", keep_extra_columns=False)["Price"]

    # Unique tag avoids collisions across processes and repeated runs
    tag = f"pid{pid}_{job['label']}".replace(".", "p").replace("-", "m").replace("=", "_")

    common_kwargs = dict(
        battery=battery,
        market=ip_cfg,
        forecasts={"qr": ip_det_qr},
        real_price_series=ip_real,
        price_source=["forecast"],
        start=start_date,
        end=end_date,
        terminal_target_kwh=battery.energy_kwh * 0.5,
        solver_name="gurobi",  # or "gurobi_direct"
        solver_options={"Threads": int(threads_per_worker), "OutputFlag": 0},
        save=True,
        tag=tag,
    )

    res = optimizations.run_ip_rolling_ce_models(
        **common_kwargs,
        terminal_penalty=float(job["terminal_penalty"]),
        terminal_penalty_mode=str(job["mode"]),
        cycle_penalty_eur_per_mwh=float(job["cycle_penalty"]),
    )

    return {
        "label": job["label"],
        "mode": job["mode"],
        "terminal_penalty": float(job["terminal_penalty"]),
        "cycle_penalty": float(job["cycle_penalty"]),
        "pid": pid,
        "final_energy_kwh": float(res.final_energy_kwh),
        "tag": tag,
    }


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    jobs = _make_jobs()

    # Start conservative; increase later if stable
    max_workers = 2
    threads_per_worker = 4

    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("optimizations:", optimizations.__file__)
    print(f"Running {len(jobs)} jobs with max_workers={max_workers}, Threads={threads_per_worker}")

    summaries = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(run_one, job, threads_per_worker=threads_per_worker) for job in jobs]
        for f in as_completed(futures):
            s = f.result()  # raises the true exception if a worker fails
            summaries.append(s)
            print("Done:", s)

    print("\nAll jobs finished. Summaries:")
    for s in summaries:
        print(s)
