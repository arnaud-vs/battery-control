"""
Utilities to read forecast outputs (DA/IP) and real prices.

Requirements from user
----------------------
- Input timestamps are already local wall-clock (naive). NO tz conversion, NO DST localization.
- If there are repeated timestamps: collapse them and print which timestamps were collapsed.
- If there are gaps: create a complete regular grid and fill gaps by interpolation; print timestamps added.

Outputs
-------
DA forecasts:
  - deterministic: single column named the model ("LEAR"/"XGB"/"QR")
    * QR deterministic = median quantile q0.5
  - probabilistic: q0.1..q0.9

DA real:
  - Date, Price (optionally extra columns) -> returns index DateTime + Price

IP forecasts (always multi-horizon):
  - deterministic: qh1..qh8 (LEAR/XGB), QR uses median per horizon
  - probabilistic: qh1_q0.1..qh8_q0.9 (8 * len(quantiles)), robust to column naming variants:
      * qh{h}_q{q}
      * q{q}_qh{h}
      * qh{q}_q{h}  (swapped)

IP real:
  - Date, Price (single series)

Folder structure (example)
--------------------------
Data/
  DA_CET/
    DA_LEAR.csv
    DA_XGB.csv
    DA_QR.csv
    DA_Real_Prices.csv
  IP_CET/
    IP_LEAR.csv
    IP_XGB.csv
    IP_QR.csv
    IP_Real_Prices.csv
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Optional, Sequence

import pandas as pd
from pathlib import Path

# project root = parent of util/


# ---------------- Types ----------------

Market = Literal["DA", "IP"]
Model = Literal["LEAR", "XGB", "QR"]
OutputKind = Literal["deterministic", "probabilistic"]


# ---------------- Defaults ----------------

DEFAULT_QUANTILES: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
DEFAULT_IP_HORIZONS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "Data"

# ---------------- Dataclass ----------------

@dataclass(frozen=True)
class ReadSpec:
    market: Market
    model: Model
    kind: OutputKind
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES
    data_dir: str | Path = DEFAULT_DATA_DIR


# ---------------- Reporting helpers ----------------

def _print_timestamps(action: str, stamps: pd.DatetimeIndex, *, max_show: int = 20) -> None:
    n = len(stamps)
    if n == 0:
        return
    shown = list(stamps[:max_show])
    more = "" if n <= max_show else f" (and {n - max_show} more)"
    print(f"[read_price] {action}: {n} timestamp(s). First {min(n, max_show)}: {shown}{more}")


# ---------------- Core time handling: parse + fix duplicates + fill gaps ----------------

def _ensure_datetime_local_index(
    df: pd.DataFrame,
    time_col: str,
    *,
    freq: str | None = None,              # e.g. "15min", "60min"; inferred if None
    fill_gaps: bool = True,
    fix_duplicates: bool = True,
    interpolate_numeric: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Parse time_col as naive timestamps, set as index, then:

    - If fix_duplicates: collapse duplicated timestamps (mean for numeric columns, first for non-numeric)
    - If fill_gaps: reindex to full regular grid and interpolate numeric columns

    Prints which timestamps were collapsed (duplicates) and which were added (missing) when verbose=True.
    """
    if time_col not in df.columns:
        raise ValueError(f"Expected datetime column '{time_col}' not found. Columns: {list(df.columns)}")

    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
    if out[time_col].isna().any():
        bad = out.loc[out[time_col].isna(), time_col]
        raise ValueError(f"Failed to parse some timestamps in '{time_col}'. Examples: {bad.head(5).tolist()}")

    out = out.sort_values(time_col)

    # --- Fix duplicates by collapsing ---
    if fix_duplicates and out[time_col].duplicated().any():
        dup_vals = pd.DatetimeIndex(out.loc[out[time_col].duplicated(keep=False), time_col].unique()).sort_values()
        if verbose:
            _print_timestamps("Collapsed duplicate timestamps (kept one row via aggregation)", dup_vals)

        numeric_cols = out.select_dtypes(include="number").columns.tolist()
        non_numeric_cols = [c for c in out.columns if c not in numeric_cols and c != time_col]

        grouped_num = out.groupby(time_col, as_index=False)[numeric_cols].mean()

        if non_numeric_cols:
            grouped_non = out.groupby(time_col, as_index=False)[non_numeric_cols].first()
            out = grouped_num.merge(grouped_non, on=time_col, how="left")
        else:
            out = grouped_num

        out = out.sort_values(time_col)

    # --- Set index ---
    out = out.set_index(time_col)
    out.index.name = "DateTime"
    out = out.sort_index()

    # --- Determine frequency ---
    use_freq = freq or pd.infer_freq(out.index)
    if use_freq is None:
        diffs = out.index.to_series().diff().dropna()
        if diffs.empty:
            # single point: nothing to fill
            return out
        # Use most common delta
        use_freq = diffs.value_counts().idxmax()

    # --- Fill gaps by reindex + interpolate ---
    if fill_gaps:
        start, end = out.index.min(), out.index.max()
        expected = pd.date_range(start=start, end=end, freq=use_freq)

        missing = expected.difference(out.index)
        if len(missing) > 0:
            if verbose:
                _print_timestamps("Added missing timestamps (will fill by interpolation)", missing)

            out = out.reindex(expected)
            out.index.name = "DateTime"

            if interpolate_numeric:
                num_cols = out.select_dtypes(include="number").columns
                if len(num_cols) > 0:
                    # time-based interpolation; fill edges too
                    out[num_cols] = out[num_cols].interpolate(method="time", limit_direction="both")

        # Note: if there are "unexpected" timestamps not on the grid, reindex(...) drops them.
        # That is intentional when we enforce a regular grid.
        extra = out.index.difference(expected)
        # after reindex, extra should be empty; kept here for clarity

    return out


# ---------------- Path helpers ----------------

def _path_for_forecast(market: Market, model: Model, data_dir: str | Path) -> Path:
    data_dir = Path(data_dir)
    folder = "DA_CET" if market == "DA" else "IP_CET"
    fname = f"{market}_{model}.csv"
    p = data_dir / folder / fname
    print(p)

    if not p.exists():
        raise FileNotFoundError(f"Forecast file not found: {p}")
    return p


def _path_for_real(market: Market, data_dir: str | Path) -> Path:
    data_dir = Path(data_dir)
    folder = "DA_CET" if market == "DA" else "IP_CET"
    fname = f"{market}_Real_Prices.csv"
    p = data_dir / folder / fname
    print(p)
    if not p.exists():
        raise FileNotFoundError(f"Real prices file not found: {p}")
    return p


# ---------------- DA column selection ----------------

def _select_da_forecast_columns(
    df: pd.DataFrame,
    model: Model,
    kind: OutputKind,
    quantiles: Iterable[float],
) -> pd.DataFrame:
    q_list = tuple(float(q) for q in quantiles)
    base = f"{model}"

    def q_col(q: float) -> str:
        return f"{base}_q{q:g}"

    if kind == "deterministic":
        if base in df.columns:
            return df[[base]].rename(columns={base: base})
        median = q_col(0.5)
        if median not in df.columns:
            raise ValueError(
                f"Deterministic requested but neither '{base}' nor median '{median}' found. "
                f"Available: {list(df.columns)}"
            )
        return df[[median]].rename(columns={median: base})

    if kind == "probabilistic":
        cols, ren = [], {}
        for q in q_list:
            c = q_col(q)
            if c not in df.columns:
                raise ValueError(f"Missing quantile column '{c}'. Available: {list(df.columns)}")
            cols.append(c)
            ren[c] = f"q{q:g}"
        return df[cols].rename(columns=ren)

    raise ValueError(f"Unknown kind: {kind}")


# ---------------- IP quantile column resolution (robust to naming) ----------------

def _fmt_q(q: float) -> str:
    return f"{float(q):g}"

def _resolve_ip_quantile_col(columns: Sequence[str], h: int, q: float) -> str:
    """
    Supports:
      - qh{h}_q{q}
      - q{q}_qh{h}
      - qh{q}_q{h}   (swapped)
    """
    q_str = _fmt_q(q)
    h_str = str(int(h))
    candidates = [
        f"qh{h_str}_q{q_str}",
        f"q{q_str}_qh{h_str}",
        f"qh{q_str}_q{h_str}",
    ]
    col_set = set(columns)
    for c in candidates:
        if c in col_set:
            return c
    raise ValueError(
        f"Could not find IP quantile column for (qh{h}, q{q_str}). "
        f"Tried {candidates}. Available (first 60): {list(columns)[:60]}"
    )


# ---------------- IP column selection ----------------

def _select_ip_forecast_columns(
    df: pd.DataFrame,
    model: Model,
    kind: OutputKind,
    quantiles: Iterable[float],
    *,
    horizons: Iterable[int] = DEFAULT_IP_HORIZONS,
) -> pd.DataFrame:
    """
    IP always returns all 8 horizons:
      - deterministic: qh1..qh8
      - probabilistic: qh{h}_q{q} for all horizons and quantiles (standardized naming)
    """
    q_list = tuple(float(q) for q in quantiles)
    h_list = tuple(int(h) for h in horizons)
    cols_available = list(df.columns)

    def det_col(h: int) -> str:
        return f"qh{h}"

    if kind == "deterministic":
        if model in ("LEAR", "XGB"):
            det_cols = [det_col(h) for h in h_list]
            missing = [c for c in det_cols if c not in df.columns]
            if missing:
                raise ValueError(
                    f"Missing deterministic IP columns: {missing}. Available (first 60): {cols_available[:60]}"
                )
            return df[det_cols].copy()

        if model == "QR":
            cols = []
            ren = {}
            for h in h_list:
                c = _resolve_ip_quantile_col(cols_available, h=h, q=0.5)
                cols.append(c)
                ren[c] = det_col(h)
            return df[cols].rename(columns=ren)

        raise ValueError(f"Unknown model: {model}")

    if kind == "probabilistic":
        cols = []
        ren = {}
        for h in h_list:
            for q in q_list:
                c = _resolve_ip_quantile_col(cols_available, h=h, q=q)
                cols.append(c)
                ren[c] = f"qh{h}_q{_fmt_q(q)}"
        return df[cols].rename(columns=ren)

    raise ValueError(f"Unknown kind: {kind}")


# ---------------- Public readers ----------------

def read_forecast(
    market: Market,
    model: Model,
    kind: OutputKind = "deterministic",
    *,
    quantiles: Iterable[float] = DEFAULT_QUANTILES,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    freq: str | None = None,
    fill_gaps: bool = True,
    fix_duplicates: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    p = _path_for_forecast(market, model, data_dir)
    df = pd.read_csv(p)

    time_col = "DateTime" if market == "DA" else "Date"
    df = _ensure_datetime_local_index(
        df,
        time_col=time_col,
        freq=freq,
        fill_gaps=fill_gaps,
        fix_duplicates=fix_duplicates,
        verbose=verbose,
    )

    if market == "DA":
        return _select_da_forecast_columns(df, model, kind, quantiles)

    return _select_ip_forecast_columns(df, model, kind, quantiles, horizons=DEFAULT_IP_HORIZONS)


def read_da_forecast(
    model: Model,
    kind: OutputKind = "deterministic",
    *,
    quantiles: Iterable[float] = DEFAULT_QUANTILES,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    freq: str | None = None,
    fill_gaps: bool = True,
    fix_duplicates: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    return read_forecast(
        market="DA",
        model=model,
        kind=kind,
        quantiles=quantiles,
        data_dir=data_dir,
        freq=freq,
        fill_gaps=fill_gaps,
        fix_duplicates=fix_duplicates,
        verbose=verbose,
    )


def read_ip_forecast(
    model: Model,
    kind: OutputKind = "deterministic",
    *,
    quantiles: Iterable[float] = DEFAULT_QUANTILES,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    freq: str | None = None,
    fill_gaps: bool = True,
    fix_duplicates: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    return read_forecast(
        market="IP",
        model=model,
        kind=kind,
        quantiles=quantiles,
        data_dir=data_dir,
        freq=freq,
        fill_gaps=fill_gaps,
        fix_duplicates=fix_duplicates,
        verbose=verbose,
    )


def read_real_prices(
    market: Market,
    *,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    freq: str | None = None,
    keep_extra_columns: bool = True,
    fill_gaps: bool = True,
    fix_duplicates: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    p = _path_for_real(market, data_dir)
    df = pd.read_csv(p)

    if "Date" not in df.columns:
        raise ValueError("Expected 'Date' column in real prices file.")
    if "Price" not in df.columns:
        raise ValueError("Expected 'Price' column in real prices file.")

    df = _ensure_datetime_local_index(
        df,
        time_col="Date",
        freq=freq,
        fill_gaps=fill_gaps,
        fix_duplicates=fix_duplicates,
        verbose=verbose,
    )

    return df if keep_extra_columns else df[["Price"]]


def read_da_real_prices(
    *,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    freq: str | None = None,
    keep_extra_columns: bool = False,
    fill_gaps: bool = True,
    fix_duplicates: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    return read_real_prices(
        market="DA",
        data_dir=data_dir,
        freq=freq,
        keep_extra_columns=keep_extra_columns,
        fill_gaps=fill_gaps,
        fix_duplicates=fix_duplicates,
        verbose=verbose,
    )


def read_ip_real_prices(
    *,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    freq: str | None = None,
    keep_extra_columns: bool = True,
    fill_gaps: bool = True,
    fix_duplicates: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    return read_real_prices(
        market="IP",
        data_dir=data_dir,
        freq=freq,
        keep_extra_columns=keep_extra_columns,
        fill_gaps=fill_gaps,
        fix_duplicates=fix_duplicates,
        verbose=verbose,
    )


# ---------------- Join helper ----------------

def join_forecast_with_real(
    market: Market,
    model: Model,
    kind: OutputKind = "deterministic",
    *,
    quantiles: Iterable[float] = DEFAULT_QUANTILES,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    freq: str | None = None,
    how: str = "inner",
    fill_gaps: bool = True,
    fix_duplicates: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Join real + forecast on DateTime index (naive local time).

    DA:
      - real: ['Price']
      - forecast deterministic: ['LEAR'/'XGB'/'QR']
      - forecast probabilistic: ['q0.1'..'q0.9']

    IP:
      - real: ['Price']
      - forecast deterministic: ['qh1'..'qh8']
      - forecast probabilistic: ['qh1_q0.1'..'qh8_q0.9'] (standardized output names)
    """
    y = read_real_prices(
        market,
        data_dir=data_dir,
        freq=freq,
        keep_extra_columns=False,
        fill_gaps=fill_gaps,
        fix_duplicates=fix_duplicates,
        verbose=verbose,
    )
    f = read_forecast(
        market=market,
        model=model,
        kind=kind,
        quantiles=quantiles,
        data_dir=data_dir,
        freq=freq,
        fill_gaps=fill_gaps,
        fix_duplicates=fix_duplicates,
        verbose=verbose,
    )
    return y.join(f, how=how)
