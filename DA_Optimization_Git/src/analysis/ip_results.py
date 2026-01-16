# src/analysis/ip_results.py
from __future__ import annotations

from dataclasses import dataclass, asdict, is_dataclass
from typing import Optional, Any, Dict, Optional, Tuple, List
from datetime import datetime, timezone
from pathlib import Path
import json
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re


# ---------------------------
# Data validation / KPIs
# ---------------------------


@dataclass(frozen=True)
class ProfitSeries:
    realized: pd.Series
    forecasted: pd.Series
    optimal: Optional[pd.Series] = None
    regret: Optional[pd.Series] = None
    naive: Optional[pd.Series] = None


# -----------------------------
# Column conventions (base + optional)
# -----------------------------
DEFAULT_PROFIT_COLS = {
    "realized": "realized_profit_eur",
    "forecasted": "forecasted_profit_eur",
    "optimal": "optimal_profit_eur",                 # optional
    "naive_realized": "naive_realized_profit_eur",   # optional
}

DEFAULT_ACTION_COLS = {
    "E_start": "E_start_kwh",
    "E_end": "E_end_kwh",
    "e_ch": "e_ch_kwh",
    "e_dis": "e_dis_kwh",
}


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
    weight_method: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "run_type": "rolling_ce_ip",
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
        "solver": str(solver_name),
        "code_versions": code_versions or {},
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "weight_method": weight_method or {},
    }


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
