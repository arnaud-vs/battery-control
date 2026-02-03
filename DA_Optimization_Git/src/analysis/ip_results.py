# src/analysis/ip_results.py
from __future__ import annotations

from dataclasses import dataclass, asdict, is_dataclass
from typing import Optional, Any, Dict, Optional, Tuple, Union
from datetime import datetime, timezone
from pathlib import Path
import json
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re




PathLike = Union[str, Path]


################### Save and Load Results ###################


def _jsonable(obj: Any) -> Any:
    """Best-effort conversion to JSON-serializable objects."""
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
        # pydantic-like
        return obj.dict()
    if hasattr(obj, "__dict__"):
        # fallback: keep only simple fields
        d = {}
        for k, v in obj.__dict__.items():
            if k.startswith("_"):
                continue
            if isinstance(v, (str, int, float, bool)) or v is None:
                d[k] = v
        if d:
            return d
    return str(obj)

def build_ip_run_metadata(
    *,
    forecast_model_name: str,
    forecast_type: str,
    start_date: str,
    end_date: str,
    freq: str,
    battery: Any,
    market: Any,
    terminal_target_kwh: float,
    terminal_penalty: float,
    terminal_penalty_mode: str,
    solver_name: str,
    price_source_forecast: str = "forecast",
    price_source_benchmark: str = "perfect_foresight",
    code_versions: Optional[Dict[str, str]] = None,
    scenario_method: Optional[str] = None,
    cycle_penalty_eur_per_mwh: float,
    n_scenarios=Optional[float],
    lam_corr=Optional[float],
    lookback_days=Optional[float],
    base_seed=Optional[float],
) -> Dict[str, Any]:
    meta = {
        "forecast_model": forecast_model_name,
        "forecast_type": forecast_type,
        "price_source_forecast": price_source_forecast,
        "price_source_benchmark": price_source_benchmark,
        "frequency": freq,
        "start_date": start_date,
        "end_date": end_date,
        "battery": _jsonable(battery),
        "market": _jsonable(market),
        "terminal_conditions": {
            "terminal_target_kwh": float(terminal_target_kwh),
            "terminal_penalty": float(terminal_penalty),
            "terminal_penalty_mode": str(terminal_penalty_mode),
        },
        "cycle_penalty_eur_per_mwh": cycle_penalty_eur_per_mwh,
        "solver": str(solver_name),
        "code_versions": code_versions or {},
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    # ---------- only for probabilistic ----------
    if forecast_type == "probabilistic":
        meta["scenario_method"] = scenario_method

        # ---------- only for copula ----------
        if scenario_method == "copula":
            meta.update({
                "n_scenarios": n_scenarios,
                "lam_corr": lam_corr,
                "lookback_days": lookback_days,
                "base_seed": base_seed,
            })

    return meta

def save_parquet_with_metadata(
    df: pd.DataFrame,
    path: Path,
    metadata: Dict[str, Any],
    *,
    compression: str = "zstd",
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    table = pa.Table.from_pandas(df, preserve_index=True)

    existing_meta = table.schema.metadata or {}
    payload = json.dumps(metadata, indent=2, default=str).encode("utf-8")
    new_meta = {**existing_meta, b"run_metadata": payload}

    table = table.replace_schema_metadata(new_meta)

    pq.write_table(table, path, compression=compression)
    return path

def default_ip_results_path(
    project_root: Path,
    *,
    forecast_model_name: str,
    forecast_type: str,
    tag: Optional[str] = None,
) -> Path:
    safe_tag = f"_{tag}" if tag else ""
    fname = f"ip_rolling_ce_{forecast_model_name}_{forecast_type}_{safe_tag}.parquet"
    return project_root / "results" / "ip_rolling_ce" / fname

def load_results_parquet(path: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Loads a parquet written by save_parquet_with_metadata and returns (df, metadata).
    """
    path = Path(path)
    table = pq.read_table(path)
    df = table.to_pandas()

    meta = table.schema.metadata or {}
    run_meta = {}
    if b"run_metadata" in meta:
        try:
            run_meta = json.loads(meta[b"run_metadata"].decode("utf-8"))
        except Exception:
            run_meta = {"_raw_run_metadata": meta[b"run_metadata"].decode("utf-8", errors="ignore")}
    return df, run_meta
################### Result Analysis ###################

def _col(base: str, var: str) -> str:
    return f"{base}.{var}"


_REQUIRED_VARS = [
    "E_start_kwh",
    "E_end_kwh",
    "e_ch_kwh",
    "e_dis_kwh",
    "price_qh1_eur_per_mwh",
    "profit_forecast_eur",
    "profit_realized_eur",
]

def validate_strategy_block(df: pd.DataFrame, base: str, *, require_price: bool = True) -> None:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df index must be a DatetimeIndex")
    if df.index.has_duplicates:
        raise ValueError("df index has duplicates; expected unique timestamps")

    req = list(_REQUIRED_VARS)
    if not require_price:
        req = [v for v in req if v != "price_qh1_eur_per_mwh"]

    missing = [ _col(base, v) for v in req if _col(base, v) not in df.columns ]
    if missing:
        raise ValueError(f"Missing required columns for base='{base}': {missing}")


def cumulative(series: pd.Series) -> pd.Series:
    return series.fillna(0.0).cumsum()


def battery_capacity_kwh_from_meta(meta: Dict[str, Any]) -> Optional[float]:
    """
    Tries to extract usable battery energy capacity from run metadata.
    Supports common shapes produced by your _jsonable(battery).
    Returns None if not found.
    """
    if not meta:
        return None
    b = meta.get("battery", None)
    if not isinstance(b, dict):
        return None

    # common patterns you might have
    for k in ["energy_kwh", "capacity_kwh", "E_kwh", "e_max_kwh"]:
        if k in b:
            try:
                return float(b[k])
            except Exception:
                pass

    # sometimes nested
    for k in ["spec", "params", "battery"]:
        if k in b and isinstance(b[k], dict):
            for kk in ["energy_kwh", "capacity_kwh"]:
                if kk in b[k]:
                    try:
                        return float(b[k][kk])
                    except Exception:
                        pass

    return None



def _find_unique_col(df: pd.DataFrame, suffix: str) -> str:
    """
    Find a unique column that ends with `suffix` (e.g. ".profit_realized_eur").
    Raises a helpful error if none or multiple exist.
    """
    matches = [c for c in df.columns if str(c).endswith(suffix)]
    if len(matches) == 0:
        raise KeyError(
            f"Could not find any column ending with '{suffix}'. "
            f"Available columns (sample): {list(df.columns)[:20]}"
        )
    if len(matches) > 1:
        raise KeyError(
            f"Found multiple columns ending with '{suffix}': {matches}. "
            "This loader expects exactly one profit column per result parquet."
        )
    return matches[0]


def _safe_sum(s: pd.Series) -> float:
    """Sum robustly (treat all-NaN as 0.0)."""
    x = pd.to_numeric(s, errors="coerce")
    if x.notna().sum() == 0:
        return 0.0
    return float(x.fillna(0.0).sum())


@dataclass(frozen=True)
class ProfitSeries:
    """Per-timestep profit series (indexed by DateTime)."""
    forecast: pd.Series
    realized: pd.Series
    optimal: Optional[pd.Series] = None  # perfect-foresight realized (upper bound)


@dataclass(frozen=True)
class ProfitTotals:
    """Aggregate totals over the full horizon."""
    forecast_total_eur: float
    realized_total_eur: float
    optimal_total_eur: Optional[float] = None


def extract_profit_series(
    df: pd.DataFrame,
    *,
    require_datetime_index: bool = True,
) -> Tuple[pd.Series, pd.Series]:
    """
    Given a results df (one scenario), return (profit_forecast_eur, profit_realized_eur) as Series.
    Column prefixes are auto-detected.
    """
    if require_datetime_index:
        if df.index.name != "DateTime" and not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(
                "Expected df indexed by DateTime (DatetimeIndex). "
                f"Got index={type(df.index)} name={df.index.name}"
            )

    c_fore = _find_unique_col(df, ".profit_forecast_eur")
    c_real = _find_unique_col(df, ".profit_realized_eur")

    s_fore = pd.to_numeric(df[c_fore], errors="coerce").rename("profit_forecast_eur")
    s_real = pd.to_numeric(df[c_real], errors="coerce").rename("profit_realized_eur")

    return s_fore, s_real


def profit_summary_for_scenario(
    scenario_path: PathLike,
    *,
    perfect_foresight_path: Optional[PathLike] = None,
) -> Tuple[ProfitSeries, ProfitTotals, Dict]:
    """
    Load one scenario parquet and compute:
      - forecast profit (series + total)
      - realized profit (series + total)
      - optimal profit (perfect foresight realized) if perfect_foresight_path provided

    Returns:
      (series, totals, run_meta_of_scenario)
    """
    df, run_meta = load_results_parquet(str(scenario_path))
    s_fore, s_real = extract_profit_series(df)

    optimal_series = None
    optimal_total = None
    if perfect_foresight_path is not None:
        df_opt, _ = load_results_parquet(str(perfect_foresight_path))
        # perfect foresight parquet has realized profit in ".profit_realized_eur"
        _, s_opt_real = extract_profit_series(df_opt)
        optimal_series = s_opt_real.rename("profit_optimal_eur")
        optimal_total = _safe_sum(optimal_series)

    series = ProfitSeries(
        forecast=s_fore,
        realized=s_real,
        optimal=optimal_series,
    )
    totals = ProfitTotals(
        forecast_total_eur=_safe_sum(s_fore),
        realized_total_eur=_safe_sum(s_real),
        optimal_total_eur=optimal_total,
    )
    return series, totals, run_meta

def infer_perfect_foresight_path(scenario_path: PathLike) -> Path:

    """
    Convenience helper for your deterministic results:
      ip_rolling_ce_qr_deterministic_forecast_... .parquet
    ->ip_rolling_ce_qr_deterministic_perfect_foresight_... .parquet

    Raises if the filename doesn't include 'deterministic_forecast'.
    """
    p = Path(scenario_path)
    name = p.name
    if "deterministic_forecast" not in name:
        raise ValueError(
            f"Cannot infer perfect foresight path from filename: {name}\n"
            "Expected substring 'deterministic_forecast'."
        )
    pf_name = name.replace("deterministic_forecast", "deterministic_perfect_foresight")
    return p.with_name(pf_name)

def plot_forecast_vs_realized(
    series,
    *,
    title: str = "",
    ax=None,
    s: float = 5,
    alpha: float = 0.4,
):
    """
    Scatter plot of forecasted vs realized profit with y=x line.

    series: ProfitSeries
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    x = series.forecast
    y = series.realized

    ax.scatter(x, y, s=s, alpha=alpha)

    # x = y reference line
    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    ax.plot([lo, hi], [lo, hi])

    ax.set_xlabel("Forecasted profit [€]")
    ax.set_ylabel("Realized profit [€]")
    ax.set_title(title)
    ax.grid(True)

    return ax
