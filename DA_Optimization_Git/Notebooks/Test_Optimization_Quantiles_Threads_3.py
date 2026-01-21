# Experiment runner: 2 parallel jobs, each solver run uses 4 Gurobi threads
# - Parallelism: joblib (n_jobs=2) -> two runs at a time
# - Solver threads: pass Threads=4 to Gurobi (works for Pyomo GUROBI / gurobi_direct)


from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import json
import time

from joblib import Parallel, delayed
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

# ----------------------------
# 1) Define an experiment spec
# ----------------------------
@dataclass(frozen=True)
class RiskSpec:
    alpha: float
    lambda_cvar: float

@dataclass(frozen=True)
class ProbRunSpec:
    # scenario
    scenario_method: str                      # "quantile_paths" | "copula"
    n_scenarios: Optional[int] = None         # copula only
    lam_corr: Optional[float] = None          # copula only
    lookback_days: Optional[int] = None       # copula only
    base_seed: Optional[int] = None           # copula only (or any stochastic generator)

    # risk
    alpha: float = 0.9
    lambda_cvar: float = 0.1

    # costs / penalties
    terminal_penalty: float = 0.05
    terminal_penalty_mode: str = "L1"
    cycle_penalty_eur_per_mwh: float = 3.0

    # bookkeeping
    name: str = ""                            # optional human-readable label


def _slug(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in s).strip("-")


# -----------------------------------------
# 2) Build a compact, high-information grid
# -----------------------------------------
def build_experiment_grid() -> List[ProbRunSpec]:
    runs: List[ProbRunSpec] = []

    # A) Scenario method comparison (RN vs RA) for both methods
    for scenario_method in ["quantile_paths"]:
        for lam in [0.0, 0.1]:
            if scenario_method == "quantile_paths":
                runs.append(ProbRunSpec(
                    scenario_method="quantile_paths",
                    alpha=0.9, lambda_cvar=lam,
                    name=f"qp_a0.9_l{lam}"
                ))
            else:
                runs.append(ProbRunSpec(
                    scenario_method="copula",
                    n_scenarios=500, lam_corr=0.98, lookback_days=30, base_seed=1,
                    alpha=0.9, lambda_cvar=lam,
                    name=f"cop_n500_r0.98_lb30_seed1_a0.9_l{lam}"
                ))

    return runs


# ----------------------------------------------------
# 3) Single-run wrapper (inject Gurobi Threads=4)
# ----------------------------------------------------
def run_one_experiment(
    spec: ProbRunSpec,
    *,
    battery: Any,
    ip_cfg: Any,
    ip_prob_qr: Any,              # your probabilistic forecast df
    ip_real: Any,                 # real price series
    start_date: str,
    end_date: str,
    tag_prefix: str = "exp",
    save: bool = True,
    solver_name: str = "gurobi_direct",
    gurobi_threads: int = 4,
    results_dir: Optional[Path] = None,
) -> Tuple[str, Optional[Any], Dict[str, Any]]:
    """
    Returns: (tag, res_or_none, info_dict)
    """

    # Unique-ish tag for your parquet naming
    tag = _slug(f"{tag_prefix}_{spec.name or spec.scenario_method}_{int(time.time())}")

    # Common args
    kwargs: Dict[str, Any] = dict(
        battery=battery,
        market=ip_cfg,
        forecasts={"qr": ip_prob_qr},
        real_price_series=ip_real,
        risks=[{"alpha": spec.alpha, "lambda_cvar": spec.lambda_cvar}],
        start=start_date,
        end=end_date,
        terminal_target_kwh=battery.energy_kwh * 0.5,
        terminal_penalty=spec.terminal_penalty,
        terminal_penalty_mode=spec.terminal_penalty_mode,
        scenario_method=spec.scenario_method,
        cycle_penalty_eur_per_mwh=spec.cycle_penalty_eur_per_mwh,
        save=save,
        tag=tag,
        solver_name=solver_name,

        # IMPORTANT: threads for Gurobi
        # If your run_ip_rolling_prob_models forwards this dict to Pyomo's solver.options,
        # this is the usual pattern.
        solver_options={"Threads": gurobi_threads},
    )

    # Add copula-only parameters
    if spec.scenario_method == "copula":
        kwargs.update(
            n_scenarios=spec.n_scenarios,
            lam_corr=spec.lam_corr,
            lookback_days=spec.lookback_days,
            base_seed=spec.base_seed,
        )

    # Persist the spec alongside results (optional but recommended)
    if results_dir is not None:
        results_dir.mkdir(parents=True, exist_ok=True)
        with open(results_dir / f"{tag}__spec.json", "w", encoding="utf-8") as f:
            json.dump(asdict(spec), f, indent=2)

    # Execute
    try:
        res = optimizations.run_ip_rolling_prob_models(**kwargs)
        info = {"status": "ok", "tag": tag}
        return tag, res, info
    except Exception as e:
        info = {"status": "error", "tag": tag, "error": repr(e)}
        return tag, None, info


# ----------------------------------------------------
# 4) Run the full grid with 2 jobs (parallel processes)
# ----------------------------------------------------
def run_all_experiments(
    *,
    battery: Any,
    ip_cfg: Any,
    ip_prob_qr: Any,
    ip_real: Any,
    start_date: str,
    end_date: str,
    tag_prefix: str = "v1",
    n_jobs: int = 2,
    gurobi_threads: int = 4,
    solver_name: str = "gurobi_direct",
    results_dir: Optional[Path] = None,
) -> List[Tuple[str, Optional[Any], Dict[str, Any]]]:

    grid = build_experiment_grid()

    # Note: if you use Gurobi, also consider setting the env var to avoid oversubscription:
    # export OMP_NUM_THREADS=1  (or set in your shell). Gurobi uses its own Threads.
    out = Parallel(n_jobs=n_jobs, prefer="processes", verbose=10)(
        delayed(run_one_experiment)(
            spec,
            battery=battery,
            ip_cfg=ip_cfg,
            ip_prob_qr=ip_prob_qr,
            ip_real=ip_real,
            start_date=start_date,
            end_date=end_date,
            tag_prefix=tag_prefix,
            save=True,
            solver_name=solver_name,
            gurobi_threads=gurobi_threads,
            results_dir=results_dir,
        )
        for spec in grid
    )

    # Optional: print a compact summary
    ok = sum(1 for _, _, info in out if info.get("status") == "ok")
    err = len(out) - ok
    print(f"Finished: {ok} ok, {err} errors")
    if err:
        for _, _, info in out:
            if info.get("status") == "error":
                print(info["tag"], info["error"])

    return out


if __name__ == "__main__":

    # Load inside the worker to avoid pickling large DataFrames
    battery, markets = config_loader.load_config(PROJECT_ROOT / "configs" / "battery_config.yaml")
    ip_cfg = markets["ip"]
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    ip_prob_qr = read_price.read_ip_forecast(model="QR", kind="probabilistic", freq="15min")
    ip_real = read_price.read_ip_real_prices(freq="15min", keep_extra_columns=False)["Price"]


    results = run_all_experiments(
        battery=battery,
        ip_cfg=ip_cfg,
        ip_prob_qr=ip_prob_qr,
        ip_real=ip_real,
        start_date=start_date,
        end_date=end_date,
        tag_prefix="prob_sweep",
        n_jobs=2,                 # <-- two runs at a time
        gurobi_threads=4,         # <-- each run uses 4 threads
        solver_name="gurobi",
        results_dir=Path(PROJECT_ROOT) / "results" / "experiment_specs",
    )