# src/ip/optimizations.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import sys

from src.battery import BatterySpec
from src.markets import MarketConfig
from src.ip.core_model import build_ip_core
from src.ip.forecast import deterministic_vector_from_row
from src.ip.objectives import add_objective_certainty_equivalent, ensure_scenario_price_param, add_objective_mean_cvar
from typing import Optional, Union, Sequence, Literal, Dict, Any
from pathlib import Path
from src.ip.scenarios import sigma_as_of, generate_copula_scenarios_at_t  # NEW

PriceSource = Literal["forecast", "perfect_foresight", "naive"]
ScenarioMethod = Literal["quantile_paths", "copula"]
PriceSources = Union[PriceSource, Sequence[PriceSource]]
DateLike = Union[str, pd.Timestamp]
TerminalPenaltyMode = Literal["none","L1","L2"]

UNIQUE_BENCH_SOURCES: set[str] = {"naive", "perfect_foresight"}

@dataclass(frozen=True)
class IPRollingResult:
    history: pd.DataFrame
    final_energy_kwh: float

################################ CE  ########################################################
def _row_with_prefix(prefix: str, **kwargs) -> dict:
    return {f"{prefix}.{k}": v for k, v in kwargs.items()}

def naive_horizon_from_real(
    real_price_full: pd.Series,
    ts: pd.Timestamp,
    horizon_steps: int,
    dt: pd.Timedelta,
) -> np.ndarray:
    # previous 8 quarters: ts-8dt ... ts-1dt
    idx = [ts - (horizon_steps - k) * dt for k in range(horizon_steps)]
    missing = [t for t in idx if t not in real_price_full.index]
    if missing:
        raise KeyError(
            f"Naive horizon missing {len(missing)} timestamps (e.g. {missing[0]}) around ts={ts}."
        )
    return real_price_full.loc[idx].astype(float).to_numpy()

def _run_ip_rolling_ce_single(
    *,
    battery: BatterySpec,
    market: MarketConfig,
    ip_det_forecast: pd.DataFrame,                 # index timestamps; cols qh1..qh8
    real_price_series: Optional[pd.Series] = None, # optional: realized price €/MWh at each timestamp
    horizon_steps: int = 8,
    enforce_no_simultaneous: bool = True,
    start_soc: Optional[float] = None,             # SOC fraction (0..1)
    solver_name: str = "gurobi_direct",
    solver_options: Optional[dict] = None,
    start: Optional[DateLike] = None,              # NEW (inclusive)
    end: Optional[DateLike] = None,                # NEW (inclusive)
    price_source: PriceSource = "forecast",
    terminal_target_kwh: Optional[float] = None,
    terminal_penalty: float = 0.0,
    terminal_penalty_mode: TerminalPenaltyMode = "none",
    cycle_penalty_eur_per_mwh: float = 0.0,
    prefix: str,

) -> IPRollingResult:
    if ip_det_forecast.empty:
        raise ValueError("ip_det_forecast is empty")

    ip_det_forecast = ip_det_forecast.sort_index()

    # --- Apply optional date filtering ---
    if start is not None:
        start_ts = pd.to_datetime(start)
        ip_det_forecast = ip_det_forecast.loc[ip_det_forecast.index >= start_ts]
    if end is not None:
        end_ts = pd.to_datetime(end)
        ip_det_forecast = ip_det_forecast.loc[ip_det_forecast.index <= end_ts]

    if ip_det_forecast.empty:
        raise ValueError("After applying start/end filtering, ip_det_forecast is empty.")

    real_price_full = None
    if real_price_series is not None:
        real_price_full = real_price_series.sort_index()

    # Initial energy
    bd0 = battery.discretize(market.dt_minutes)
    E = bd0.e_init_kwh if start_soc is None else float(start_soc) * battery.energy_kwh


    # Build model once
    core = build_ip_core(
        bd=bd0,
        market=market,
        horizon_steps=horizon_steps,
        enforce_no_simultaneous=enforce_no_simultaneous,
    )

    m = core.model

    # Must have mutable E_init for rolling
    if not getattr(m.E_init, "mutable", False):
        raise ValueError("E_init must be mutable. Set mutable=True in core_model.py")

    add_objective_certainty_equivalent(
        m,
        grid_fee_eur_per_mwh=market.grid_fee_eur_per_mwh,
        terminal_target_kwh=terminal_target_kwh,
        terminal_penalty=terminal_penalty,
        terminal_penalty_mode=terminal_penalty_mode,
        cycle_penalty_eur_per_mwh=cycle_penalty_eur_per_mwh,
    )
    # Solver
    solver = pyo.SolverFactory(solver_name)
    if solver is None or not solver.available():
        raise RuntimeError(f"Solver '{solver_name}' not available.")

    # Options (Gurobi)
    default_opts = {"OutputFlag": 0}
    opts = {}
    opts.update(default_opts)
    if solver_options:
        opts.update(solver_options)
    for k, v in opts.items():
        solver.options[k] = v

    e_min = float(bd0.e_min_kwh)
    e_max = float(bd0.e_max_kwh)
    fee = float(getattr(market, "grid_fee_eur_per_mwh", 0.0))

    rows = []
    a=0
    for ts, f_row in ip_det_forecast.iterrows():
        progress = 100 * a / len(ip_det_forecast)
        print(f"\rSolving {price_source}: {progress:5.1f} %", end="", flush=True)
        # Update initial SOC
        m.E_init.set_value(float(E))

        dt = pd.Timedelta(minutes=15)
        if real_price_full is not None and len(real_price_full.index) >= 2:
            dt = real_price_full.index[1] - real_price_full.index[0]

        # Update price parameters for horizon
        if price_source == "forecast":
            p = deterministic_vector_from_row(f_row, horizon_steps=horizon_steps)

        elif price_source == "perfect_foresight":
            if real_price_full is None:
                raise ValueError("price_source='perfect_foresight' requires real_price_series.")

            idx = [ts + k * dt for k in range(horizon_steps)]
            missing = [t for t in idx if t not in real_price_full.index]
            if missing:
                raise KeyError(f"Perfect foresight horizon missing {len(missing)} timestamps (e.g. {missing[0]}) around ts={ts}.")


            p = real_price_full.loc[idx].astype(float).to_numpy()

        elif price_source == "naive":
            if real_price_full is None:
                raise ValueError("price_source='naive' requires real_price_series.")
            p = naive_horizon_from_real(real_price_full, ts, horizon_steps, dt)

        else:
            raise ValueError(f"Unknown price_source: {price_source}")
        
        for t in range(horizon_steps):
            m.price[t].set_value(float(p[t]))

        # Solve
        solver.solve(m, tee=False)

        # Implement first action only
        e_ch0 = float(pyo.value(m.e_ch[0]))
        e_dis0 = float(pyo.value(m.e_dis[0]))
        E_start = float(E)

        E = E + float(battery.eta_charge) * e_ch0 - (1.0 / float(battery.eta_discharge)) * e_dis0
        E = min(max(E, e_min), e_max)

        real_p = (
            float(real_price_full.loc[ts])
            if (real_price_full is not None and ts in real_price_full.index)
            else np.nan
        )
        forecasted_profit = ((p[0] - fee) / 1000.0) * (e_dis0 - e_ch0) 
        realized_profit = ((real_p - fee) / 1000.0) * (e_dis0 - e_ch0) if not np.isnan(real_p) else np.nan

        profit_forecast = forecasted_profit if price_source != "perfect_foresight" else np.nan

        rows.append(
            {
                "DateTime": ts,
                **_row_with_prefix(
                    prefix,
                    E_start_kwh=E_start,
                    e_ch_kwh=e_ch0,
                    e_dis_kwh=e_dis0,
                    E_end_kwh=float(E),
                    price_qh1_eur_per_mwh=float(p[0]),
                    profit_forecast_eur=profit_forecast,
                    profit_realized_eur=realized_profit,
                ),
            }
        )

        a=a+1

    hist = pd.DataFrame(rows).set_index("DateTime")
    return IPRollingResult(history=hist, final_energy_kwh=float(E))

def run_ip_rolling_ce_models(
    *,
    battery,
    market,
    forecasts: Dict[str, pd.DataFrame],   # keys: "lear","xgb","qr"
    real_price_series: Optional[pd.Series],
    price_source: PriceSources = "forecast",
    horizon_steps: int = 8,
    enforce_no_simultaneous: bool = True,
    start_soc: Optional[float] = None,
    solver_name: str = "gurobi_direct",
    solver_options: Optional[dict] = None,
    start: Optional[DateLike] = None,
    end: Optional[DateLike] = None,
    terminal_target_kwh: Optional[float] = None,
    terminal_penalty: float = 0.0,
    cycle_penalty_eur_per_mwh: float = 0.0,
    terminal_penalty_mode: TerminalPenaltyMode = "none",
    save: bool = True,
    tag: Optional[str] = None,
    code_versions: Optional[Dict[str, str]] = None,
) -> IPRollingResult:

    # normalize sources
    if isinstance(price_source, (list, tuple, set)):
        sources = list(price_source)
    else:
        sources = [price_source]
    seen = set()
    sources = [s for s in sources if not (s in seen or seen.add(s))]

    idx0 = None
    for model, dfm in forecasts.items():
        idx = dfm.sort_index().index
        if idx0 is None:
            idx0 = idx
        elif not idx.equals(idx0):
            raise ValueError(f"Forecast indices are not identical across models (mismatch at '{model}').")
        
    dfs: list[pd.DataFrame] = []
    final_E: dict[str, float] = {}
    
    bench_sources = [s for s in sources if s in UNIQUE_BENCH_SOURCES]
    model_sources = [s for s in sources if s not in UNIQUE_BENCH_SOURCES]

    def _run(prefix: str, ip_det: pd.DataFrame, src: PriceSource) -> None:
        res = _run_ip_rolling_ce_single(
            battery=battery,
            market=market,
            ip_det_forecast=ip_det,
            real_price_series=real_price_series,
            horizon_steps=horizon_steps,
            enforce_no_simultaneous=enforce_no_simultaneous,
            start_soc=start_soc,
            solver_name=solver_name,
            solver_options=solver_options,
            start=start,
            end=end,
            price_source=src,
            prefix=prefix,
            terminal_target_kwh=terminal_target_kwh,
            terminal_penalty=terminal_penalty,
            terminal_penalty_mode=terminal_penalty_mode,
            cycle_penalty_eur_per_mwh = cycle_penalty_eur_per_mwh,
        )
        dfs.append(res.history)
        final_E[prefix] = res.final_energy_kwh

    # 1) model-dependent sources for each model
    for model, ip_det in forecasts.items():
        model_l = str(model).lower()
        for src in model_sources:
            _run(f"{model_l}.{src}", ip_det, src)

    # 2) unique benchmarks once (driven by first model's index)
    if bench_sources:
        if real_price_series is None:
            raise ValueError("Unique benchmarks require real_price_series (naive/perfect_foresight).")
        driver_forecast = forecasts[next(iter(forecasts))]
        for src in bench_sources:
            _run(f"benchmark.{src}", driver_forecast, src)


    hist = dfs[0]
    for d in dfs[1:]:
        hist = hist.join(d, how="inner")

    out = IPRollingResult(history=hist, final_energy_kwh=list(final_E.values())[-1])

    if save:
        project_root = Path(__file__).resolve().parents[2]
        safe_tag = f"_{tag}" if tag else ""
        models_str = "-".join(sorted([str(k).lower() for k in forecasts.keys()]))
        sources_str = "-".join(sources)
        fname = f"ip_rolling_ce_{models_str}_deterministic_{sources_str}_TP_{terminal_penalty_mode}_{terminal_penalty}_CP_{cycle_penalty_eur_per_mwh}_{safe_tag}.parquet"
        out_path = project_root / "results" / "ip_rolling_ce" / fname

        from src.analysis import ip_results
        meta = ip_results.build_ip_run_metadata(
            forecast_model_name=models_str,
            forecast_type="deterministic",
            start_date=str(start) if start is not None else "",
            end_date=str(end) if end is not None else "",
            freq=f"{market.dt_minutes}min" if hasattr(market, "dt_minutes") else "",
            battery=battery,
            market=market,
            terminal_target_kwh=float(terminal_target_kwh) if terminal_target_kwh is not None else float("nan"),
            terminal_penalty=float(terminal_penalty),
            terminal_penalty_mode=str(terminal_penalty_mode),
            solver_name=solver_name,
            price_source_forecast=sources_str,
            price_source_benchmark="perfect_foresight" if "perfect_foresight" in sources else "",
            code_versions=code_versions or {},
            cycle_penalty_eur_per_mwh=float(cycle_penalty_eur_per_mwh)
        )
        meta["final_energy_by_mode_kwh"] = final_E
        meta["models"] = sorted([str(k).lower() for k in forecasts.keys()])
        meta["sources"] = sources
        meta["unique_benchmarks"] = sorted(UNIQUE_BENCH_SOURCES)
        meta["benchmark_prefixes"] = [f"benchmark.{s}" for s in UNIQUE_BENCH_SOURCES if s in sources]

        ip_results.save_parquet_with_metadata(hist, out_path, meta)

    return out

################################ Scenarios: Each quantile a probability  ########################################################
def _risk_tag(*, alpha: float, lambda_cvar: float) -> str:
    if float(lambda_cvar) == 0.0:
        return "rn"
    # stable, filename-safe
    return f"cvar_a{alpha:.2f}_l{lambda_cvar:.3g}".replace(".", "p")

def _extract_quantiles_sorted(ip_prob_forecast: pd.DataFrame) -> list[float]:
    # columns look like qh1_q0.1 ... qh8_q0.9
    qs = set()
    for c in ip_prob_forecast.columns:
        if "_q" in c:
            try:
                qs.add(float(c.split("_q")[-1]))
            except Exception:
                pass
    out = sorted(qs)
    if not out:
        raise ValueError("Could not infer quantiles from columns (expected *_q0.x).")
    return out

def _cdf_bin_weights_renorm(quantiles: list[float]) -> np.ndarray:
    q = np.array(quantiles, dtype=float)
    if not (np.all(q > 0) and np.all(q < 1) and np.all(np.diff(q) > 0)):
        raise ValueError(f"Invalid quantiles list: {quantiles}")
    w = np.empty_like(q)
    w[0] = q[0]
    w[1:] = q[1:] - q[:-1]
    w = w / w.sum()
    return w

def _run_ip_rolling_prob_single(
    *,
    battery: BatterySpec,
    market: MarketConfig,
    ip_prob_forecast: pd.DataFrame,
    real_price_series: Optional[pd.Series] = None,
    horizon_steps: int = 8,
    enforce_no_simultaneous: bool = True,
    start_soc: Optional[float] = None,
    solver_name: str = "gurobi_direct",
    solver_options: Optional[dict] = None,
    start: Optional[DateLike] = None,
    end: Optional[DateLike] = None,
    terminal_target_kwh: Optional[float] = None,
    terminal_penalty: float = 0.0,
    terminal_penalty_mode: TerminalPenaltyMode = "none",
    alpha: float = 0.9,
    lambda_cvar: float = 0.0,
    prefix: str,   # NEW
    scenario_method: ScenarioMethod = "quantile_paths",
    n_scenarios: int = 500,                 # only used when scenario_method="copula"
    lam_corr: float = 0.995,                # EWMA lambda for Sigma_t
    lookback_days: int = 30,                # limit history used for Sigma_t
    base_seed: int = 123,                   # reproducible scenarios
    cycle_penalty_eur_per_mwh: float = 0.0,  
) -> IPRollingResult:
    
    if ip_prob_forecast.empty:
        raise ValueError("ip_prob_forecast is empty")

    ip_prob_forecast = ip_prob_forecast.sort_index()

    if start is not None:
        start_ts = pd.to_datetime(start)
        ip_prob_forecast = ip_prob_forecast.loc[ip_prob_forecast.index >= start_ts]
    if end is not None:
        end_ts = pd.to_datetime(end)
        ip_prob_forecast = ip_prob_forecast.loc[ip_prob_forecast.index <= end_ts]

    if ip_prob_forecast.empty:
        raise ValueError("After applying start/end filtering, ip_prob_forecast is empty.")

    real_price_full = real_price_series.sort_index() if real_price_series is not None else None

    quantiles = np.array(_extract_quantiles_sorted(ip_prob_forecast), dtype=float)

    if scenario_method == "quantile_paths":
        S = len(quantiles)   
        weights = _cdf_bin_weights_renorm(list(quantiles))

    elif scenario_method == "copula":
        S = int(n_scenarios)
        if S <= 0:
            raise ValueError("n_scenarios must be > 0 when scenario_method='copula'")
        # Monte Carlo weights: equal
        weights = np.ones(S, dtype=float) / S

    else:
        raise ValueError(f"Unknown scenario_method: {scenario_method}")

    if lambda_cvar < 0.0:
        raise ValueError("lambda_cvar must be >= 0")
    if lambda_cvar > 0.0 and not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1) when lambda_cvar > 0")

    # Battery init
    bd0 = battery.discretize(market.dt_minutes)
    E = bd0.e_init_kwh if start_soc is None else float(start_soc) * battery.energy_kwh

    # Build model once
    core = build_ip_core(
        bd=bd0,
        market=market,
        horizon_steps=horizon_steps,
        enforce_no_simultaneous=enforce_no_simultaneous,
    )
    m = core.model

    if not getattr(m.E_init, "mutable", False):
        raise ValueError("E_init must be mutable. Set mutable=True in core_model.py")

    # Add mutable scenario price param once
    ensure_scenario_price_param(m, num_scenarios=S)

    # Choose objective once (RN if lambda=0, else Mean–CVaR)
    if lambda_cvar == 0.0:
        mode_label = "RN"
    else:
        mode_label = f"CVaR(a={alpha},lam={lambda_cvar})"

    add_objective_mean_cvar(
        m,
        scenario_weights=weights,
        grid_fee_eur_per_mwh=market.grid_fee_eur_per_mwh,
        alpha=alpha,
        lambda_cvar=lambda_cvar, #if 0 => RN
        terminal_target_kwh=terminal_target_kwh,
        terminal_penalty=terminal_penalty,
        terminal_penalty_mode=terminal_penalty_mode,
        cycle_penalty_eur_per_mwh = cycle_penalty_eur_per_mwh,
    )
    

    # Solver
    solver = pyo.SolverFactory(solver_name)
    if solver is None or not solver.available():
        raise RuntimeError(f"Solver '{solver_name}' not available.")

    default_opts = {"OutputFlag": 0}
    opts = {**default_opts, **(solver_options or {})}
    for k, v in opts.items():
        solver.options[k] = v

    e_min = float(bd0.e_min_kwh)
    e_max = float(bd0.e_max_kwh)
    fee = float(getattr(market, "grid_fee_eur_per_mwh", 0.0))

    rows = []
    n = len(ip_prob_forecast)

    for a, (ts, f_row) in enumerate(ip_prob_forecast.iterrows()):
        progress = 100 * a / n
        print(f"\rSolving prob {mode_label}: {progress:5.1f} %", end="", flush=True)

        m.E_init.set_value(float(E))

        dt = pd.Timedelta(minutes=getattr(market, "dt_minutes", 15))
        lookback = pd.Timedelta(days=int(lookback_days))

        if scenario_method == "quantile_paths":
            # Existing behavior: scenario s = same quantile level across all horizons
            for t_h in range(horizon_steps):
                qh = t_h + 1
                for s_idx, q in enumerate(quantiles):
                    # robust column name formatting
                    col = f"qh{qh}_q{q}"
                    if col not in f_row.index:
                        col2 = f"qh{qh}_q{q:.1f}"
                        col3 = f"qh{qh}_q{q:.2f}"
                        if col2 in f_row.index:
                            col = col2
                        elif col3 in f_row.index:
                            col = col3
                        else:
                            raise KeyError(f"Missing column '{col}' for ts={ts}")
                    m.scen_price[t_h, s_idx].set_value(float(f_row[col]))

        elif scenario_method == "copula":
            if real_price_full is None:
                raise ValueError("scenario_method='copula' requires real_price_series (to estimate Sigma causally).")

            # 1) strictly-causal Sigma_t
            Sigma_t = sigma_as_of(
                ip_prob_hist=ip_prob_forecast,     # full history, but sigma_as_of will cut it causally using t_now
                ip_real=real_price_full,
                t_now=ts,
                K=horizon_steps,
                quantiles=quantiles,
                lam=lam_corr,
                dt=dt,
                lookback=lookback,
            )

            # 2) scenario paths (S x K), correlated across horizons
            scen_paths = generate_copula_scenarios_at_t(
                f_row,
                Sigma=Sigma_t,
                K=horizon_steps,
                quantiles=quantiles,
                n_scenarios=S,
                seed=int(base_seed) + int(a),  # deterministic but changes each timestep
            )

            # 3) write into Pyomo parameters
            for t_h in range(horizon_steps):
                for s_idx in range(S):
                    m.scen_price[t_h, s_idx].set_value(float(scen_paths[s_idx, t_h]))

        solver.solve(m, tee=False)

        # Implement first action only
        e_ch0 = float(pyo.value(m.e_ch[0]))
        e_dis0 = float(pyo.value(m.e_dis[0]))
        E_start = float(E)

        E = E + float(battery.eta_charge) * e_ch0 - (1.0 / float(battery.eta_discharge)) * e_dis0
        E = min(max(E, e_min), e_max)

        real_p = (
            float(real_price_full.loc[ts])
            if (real_price_full is not None and ts in real_price_full.index)
            else np.nan
        )

        # expected first-step price under weights (for logging)
        p0_s = np.array([pyo.value(m.scen_price[0, s]) for s in range(S)], dtype=float)
        exp_p0 = float(np.dot(weights, p0_s))

        forecasted_profit = ((exp_p0 - fee) / 1000.0) * (e_dis0 - e_ch0)
        realized_profit = ((real_p - fee) / 1000.0) * (e_dis0 - e_ch0) if not np.isnan(real_p) else np.nan

        # Optionally compute CVaR stats for logging (only if CVaR is enabled)
        # Keep it lightweight: only compute profit_s for the chosen action for t=0
        # (Full-horizon scenario profits are possible but more expensive to compute here)
        rows.append(
            {
                "DateTime": ts,
                **_row_with_prefix(
                    prefix,
                    E_start_kwh=E_start,
                    e_ch_kwh=e_ch0,
                    e_dis_kwh=e_dis0,
                    E_end_kwh=float(E),
                    price_qh1_eur_per_mwh=exp_p0,
                    profit_forecast_eur=forecasted_profit,
                    profit_realized_eur=realized_profit,
                ),
            }
        )
    
    hist = pd.DataFrame(rows).set_index("DateTime")
    return IPRollingResult(history=hist, final_energy_kwh=float(E))

def run_ip_rolling_prob_models(
    *,
    battery,
    market,
    forecasts: Dict[str, pd.DataFrame],   # {"lear": df, "xgb": df, "qr": df}
    real_price_series: Optional[pd.Series],
    risks: Sequence[Dict[str, Any]],      # e.g. [{"alpha":0.9,"lambda_cvar":0.0}, {"alpha":0.9,"lambda_cvar":2.0}]
    horizon_steps: int = 8,
    enforce_no_simultaneous: bool = True,
    start_soc: Optional[float] = None,
    solver_name: str = "gurobi_direct",
    solver_options: Optional[dict] = None,
    start: Optional[DateLike] = None,
    end: Optional[DateLike] = None,
    terminal_target_kwh: Optional[float] = None,
    terminal_penalty: float = 0.0,
    terminal_penalty_mode: TerminalPenaltyMode = "none",
    save: bool = True,
    tag: Optional[str] = None,
    code_versions: Optional[Dict[str, str]] = None,
    scenario_method: ScenarioMethod = "quantile_paths",
    n_scenarios: int = 500,
    lam_corr: float = 0.995,
    lookback_days: int = 30,
    base_seed: int = 123,
    cycle_penalty_eur_per_mwh: float = 0.0,  
) -> IPRollingResult:

    # sanity: identical indices across models (same as deterministic)
    idx0 = None
    for model, dfm in forecasts.items():
        idx = dfm.sort_index().index
        if idx0 is None:
            idx0 = idx
        elif not idx.equals(idx0):
            raise ValueError(f"Prob forecast indices are not identical across models (mismatch at '{model}').")

    # de-dup risks while preserving order
    seen = set()
    risks_uniq = []
    for r in risks:
        a = float(r.get("alpha", 0.9))
        lam = float(r.get("lambda_cvar", 0.0))
        key = (round(a, 6), round(lam, 12))
        if key not in seen:
            seen.add(key)
            risks_uniq.append({"alpha": a, "lambda_cvar": lam})

    dfs: list[pd.DataFrame] = []
    final_E: dict[str, float] = {}

    def _run(prefix: str, ip_prob: pd.DataFrame, *, alpha: float, lambda_cvar: float) -> None:
        res = _run_ip_rolling_prob_single(
            battery=battery,
            market=market,
            ip_prob_forecast=ip_prob,
            real_price_series=real_price_series,
            horizon_steps=horizon_steps,
            enforce_no_simultaneous=enforce_no_simultaneous,
            start_soc=start_soc,
            solver_name=solver_name,
            solver_options=solver_options,
            start=start,
            end=end,
            terminal_target_kwh=terminal_target_kwh,
            terminal_penalty=terminal_penalty,
            terminal_penalty_mode=terminal_penalty_mode,
            alpha=alpha,
            lambda_cvar=lambda_cvar,
            prefix=prefix,
            scenario_method=scenario_method,
            n_scenarios=n_scenarios,
            lam_corr=lam_corr,
            lookback_days=lookback_days,
            base_seed=base_seed,
            cycle_penalty_eur_per_mwh = cycle_penalty_eur_per_mwh,
        )
        dfs.append(res.history)
        final_E[prefix] = res.final_energy_kwh

    for model, ip_prob in forecasts.items():
        model_l = str(model).lower()
        for r in risks_uniq:
            alpha = float(r["alpha"])
            lam = float(r["lambda_cvar"])
            rtag = _risk_tag(alpha=alpha, lambda_cvar=lam)
            prefix = f"{model_l}.prob.{rtag}"
            _run(prefix, ip_prob, alpha=alpha, lambda_cvar=lam)

    hist = dfs[0]
    for d in dfs[1:]:
        hist = hist.join(d, how="inner")

    out = IPRollingResult(history=hist, final_energy_kwh=list(final_E.values())[-1])

    if save:
        project_root = Path.cwd().parent
        safe_tag = f"_{tag}" if tag else ""
        copula_info = f"_lambCorr_{lam_corr}" if scenario_method=="copula" else ""
        models_str = "-".join(sorted([str(k).lower() for k in forecasts.keys()]))

        risk_tags = []
        for r in risks_uniq:
            risk_tags.append(_risk_tag(alpha=r["alpha"], lambda_cvar=r["lambda_cvar"]))
        risks_str = "-".join(risk_tags)

        fname = f"ip_rolling_prob_{models_str}_{scenario_method}{copula_info}_{risks_str}_TP_{terminal_penalty_mode}_{terminal_penalty}_CP_{cycle_penalty_eur_per_mwh}{safe_tag}.parquet"
        out_path = project_root / "results" / "ip_rolling_prob" / fname

        from src.analysis import ip_results
        meta = ip_results.build_ip_run_metadata(
            forecast_model_name=models_str,
            forecast_type="probabilistic",
            start_date=str(start) if start is not None else "",
            end_date=str(end) if end is not None else "",
            freq=f"{market.dt_minutes}min" if hasattr(market, "dt_minutes") else "",
            battery=battery,
            market=market,
            terminal_target_kwh=float(terminal_target_kwh) if terminal_target_kwh is not None else float("nan"),
            terminal_penalty=float(terminal_penalty),
            terminal_penalty_mode=str(terminal_penalty_mode),
            solver_name=solver_name,
            price_source_forecast="prob",
            price_source_benchmark="",
            code_versions=code_versions or {},
            scenario_method=scenario_method,
            n_scenarios=n_scenarios,
            lam_corr=lam_corr,
            lookback_days=lookback_days,
            base_seed=base_seed,
            cycle_penalty_eur_per_mwh=float(cycle_penalty_eur_per_mwh)

        )
        meta["models"] = sorted([str(k).lower() for k in forecasts.keys()])
        meta["risk_configs"] = risks_uniq
        meta["risk_tags"] = risk_tags
        meta["final_energy_by_mode_kwh"] = final_E

        ip_results.save_parquet_with_metadata(hist, out_path, meta)

    return out
