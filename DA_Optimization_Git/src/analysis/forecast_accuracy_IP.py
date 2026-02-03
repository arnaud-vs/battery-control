from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ----------------------------
# Utilities: parsing + alignment
# ----------------------------

def _infer_dt_from_index(idx: pd.DatetimeIndex) -> pd.Timedelta:
    freq = pd.infer_freq(idx)
    if freq is not None:
        return pd.to_timedelta(freq)
    diffs = idx.to_series().diff().dropna()
    if diffs.empty:
        raise ValueError("Cannot infer dt from an index with <2 timestamps.")
    return diffs.value_counts().idxmax()


def _align_horizon_series(
    y: pd.Series,
    origin_index: pd.DatetimeIndex,
    *,
    h: int,
    dt: pd.Timedelta,
    # Convention: qh1 predicts y at origin time t (offset 0).
    # If instead qh1 predicts y at t+dt, set base_offset_steps=1.
    base_offset_steps: int = 0,
) -> pd.Series:
    """
    Return y_target indexed by origin_index, where y_target.loc[t] == y.loc[t + (base_offset_steps + (h-1))*dt].
    """
    offset_steps = base_offset_steps + (h - 1)
    target_times = origin_index + offset_steps * dt
    y_aligned = y.reindex(target_times)
    y_aligned.index = origin_index
    return y_aligned


def _parse_prob_cols_to_multiindex(prob: pd.DataFrame) -> Tuple[pd.DataFrame, List[int], List[float]]:
    """
    Expect columns like 'qh{h}_q{q}'.
    Returns:
      - df with MultiIndex columns: (horizon:int, quantile:float)
      - horizons list
      - quantiles list
    """
    horizons = []
    quantiles = []
    tuples = []

    for c in prob.columns:
        # strict parse: qh{h}_q{q}
        if not c.startswith("qh") or "_q" not in c:
            raise ValueError(
                f"Probabilistic column '{c}' not in expected format 'qh{{h}}_q{{q}}'."
            )
        left, right = c.split("_q", 1)
        h_str = left.replace("qh", "")
        q_str = right
        h = int(h_str)
        q = float(q_str)
        horizons.append(h)
        quantiles.append(q)
        tuples.append((h, q))

    mi = pd.MultiIndex.from_tuples(tuples, names=["horizon", "quantile"])
    out = prob.copy()
    out.columns = mi

    H = sorted(set(horizons))
    Q = sorted(set(quantiles))
    return out, H, Q


# ----------------------------
# Deterministic metrics
# ----------------------------

def evaluate_ip_deterministic(
    det: pd.DataFrame,
    real: pd.Series | pd.DataFrame,
    *,
    dt: Optional[pd.Timedelta] = None,
    horizons: Sequence[int] = (1, 2, 3, 4, 5, 6, 7, 8),
    base_offset_steps: int = 0,
) -> pd.DataFrame:
    """
    det: DataFrame with columns qh1..qh8, indexed by origin DateTime.
    real: Series (or DataFrame with 'Price') indexed by DateTime of realized intervals.

    Returns a DataFrame indexed by horizon with MAE and RMSE.
    """
    if isinstance(real, pd.DataFrame):
        if "Price" not in real.columns:
            raise ValueError("real DataFrame must contain a 'Price' column.")
        y = real["Price"].copy()
    else:
        y = real.copy()

    if dt is None:
        dt = _infer_dt_from_index(det.index)

    rows = []
    for h in horizons:
        col = f"qh{h}"
        if col not in det.columns:
            raise ValueError(f"Missing deterministic column '{col}' in det forecast.")
        y_target = _align_horizon_series(y, det.index, h=h, dt=dt, base_offset_steps=base_offset_steps)
        f = det[col]

        mask = y_target.notna() & f.notna()
        if mask.sum() == 0:
            mae = np.nan
            rmse = np.nan
        else:
            e = (f[mask] - y_target[mask]).to_numpy()
            mae = float(np.mean(np.abs(e)))
            rmse = float(np.sqrt(np.mean(e**2)))

        rows.append({"horizon": h, "MAE": mae, "RMSE": rmse, "N": int(mask.sum())})

    out = pd.DataFrame(rows).set_index("horizon")
    return out


# ----------------------------
# Probabilistic metrics
# ----------------------------

def _pinball_loss(y: np.ndarray, qhat: np.ndarray, tau: float) -> np.ndarray:
    """
    Pinball loss for quantile tau.
    """
    u = y - qhat
    return np.maximum(tau * u, (tau - 1.0) * u)


def _crps_from_quantiles(y: np.ndarray, qhats: np.ndarray, taus: np.ndarray) -> np.ndarray:
    """
    Approximate CRPS via quantile representation:
      CRPS(F, y) = 2 * ∫_0^1 ρ_tau(y - q_tau) d tau
    Discrete approximation using trapezoidal rule over tau.
    Inputs:
      y: (n,)
      qhats: (n, m) for m quantiles corresponding to taus
      taus: (m,) sorted
    Returns:
      crps: (n,)
    """
    # compute pinball losses per tau: (n, m)
    losses = np.column_stack([_pinball_loss(y, qhats[:, j], float(taus[j])) for j in range(len(taus))])

    # trapezoidal integration over taus
    # integral approx: sum_j w_j * loss_j, with w from trapezoid
    w = np.zeros_like(taus, dtype=float)
    if len(taus) == 1:
        w[0] = 1.0
    else:
        w[0] = (taus[1] - taus[0]) / 2.0
        w[-1] = (taus[-1] - taus[-2]) / 2.0
        for j in range(1, len(taus) - 1):
            w[j] = (taus[j + 1] - taus[j - 1]) / 2.0

    integral = losses @ w  # (n,)
    return 2.0 * integral


def evaluate_ip_probabilistic(
    prob: pd.DataFrame,
    real: pd.Series | pd.DataFrame,
    *,
    dt: Optional[pd.Timedelta] = None,
    horizons: Optional[Sequence[int]] = None,
    quantiles: Optional[Sequence[float]] = None,
    # intervals as (low_q, high_q); coverage and sharpness computed for each.
    intervals: Sequence[Tuple[float, float]] = ((0.1, 0.9), (0.05, 0.95), (0.25, 0.75)),
    base_offset_steps: int = 0,
) -> Dict[str, pd.DataFrame]:
    """
    prob: DataFrame with columns like 'qh1_q0.1'...'qh8_q0.9', indexed by origin DateTime.
    real: realized price series indexed by DateTime.

    Returns dict of DataFrames:
      - 'coverage': index horizon, columns interval labels
      - 'calibration': MultiIndex (horizon, quantile) with empirical CDF value P(y <= qhat_tau)
      - 'sharpness': index horizon, columns interval labels (avg width)
      - 'crps': index horizon with mean CRPS
    """
    if isinstance(real, pd.DataFrame):
        if "Price" not in real.columns:
            raise ValueError("real DataFrame must contain a 'Price' column.")
        y = real["Price"].copy()
    else:
        y = real.copy()

    if dt is None:
        dt = _infer_dt_from_index(prob.index)

    prob_mi, H_found, Q_found = _parse_prob_cols_to_multiindex(prob)

    H = list(horizons) if horizons is not None else H_found
    Q = list(quantiles) if quantiles is not None else Q_found

    # ensure availability
    missing_H = [h for h in H if h not in H_found]
    missing_Q = [q for q in Q if q not in Q_found]
    if missing_H:
        raise ValueError(f"Requested horizons not found in prob columns: {missing_H}. Found: {H_found}")
    if missing_Q:
        raise ValueError(f"Requested quantiles not found in prob columns: {missing_Q}. Found: {Q_found}")

    taus = np.array(sorted(Q), dtype=float)

    # Prepare outputs
    coverage_rows = []
    sharp_rows = []
    crps_rows = []
    calib_rows = []

    interval_labels = [f"[q{lo:g}, q{hi:g}]" for lo, hi in intervals]

    for h in H:
        # align real for this horizon
        y_target = _align_horizon_series(y, prob.index, h=h, dt=dt, base_offset_steps=base_offset_steps)

        # get quantile forecasts at horizon h: DataFrame indexed by origin, columns taus
        qdf = prob_mi[h].copy()  # columns are quantiles
        qdf = qdf.reindex(columns=taus)  # enforce order

        # common valid mask across y and at least needed columns
        mask = y_target.notna()
        # For CRPS/calibration we need all selected quantiles; for coverage/sharpness only lo/hi.
        mask_all_q = mask & qdf.notna().all(axis=1)

        # ----- Coverage + sharpness for intervals -----
        cov = {}
        shp = {}
        for (lo, hi), label in zip(intervals, interval_labels):
            if lo not in qdf.columns or hi not in qdf.columns:
                cov[label] = np.nan
                shp[label] = np.nan
                continue

            qlo = qdf[lo]
            qhi = qdf[hi]
            mask_int = mask & qlo.notna() & qhi.notna()

            if mask_int.sum() == 0:
                cov[label] = np.nan
                shp[label] = np.nan
            else:
                yt = y_target[mask_int]
                inside = (yt >= qlo[mask_int]) & (yt <= qhi[mask_int])
                cov[label] = float(inside.mean())
                shp[label] = float((qhi[mask_int] - qlo[mask_int]).mean())

        coverage_rows.append({"horizon": h, **cov, "N": int(mask.sum())})
        sharp_rows.append({"horizon": h, **shp, "N": int(mask.sum())})

        # ----- Calibration per quantile: P(y <= qhat_tau) should be ~ tau -----
        if mask_all_q.sum() == 0:
            for tau in taus:
                calib_rows.append({"horizon": h, "quantile": float(tau), "empirical": np.nan, "N": 0})
        else:
            yt = y_target[mask_all_q].to_numpy()
            qhat = qdf.loc[mask_all_q].to_numpy()  # (n, m)
            # empirical CDF at qhat_tau: mean( y <= qhat_tau )
            emp = (yt[:, None] <= qhat).mean(axis=0)
            n_cal = int(mask_all_q.sum())
            for j, tau in enumerate(taus):
                calib_rows.append({"horizon": h, "quantile": float(tau), "empirical": float(emp[j]), "N": n_cal})

        # ----- CRPS (mean) -----
        if mask_all_q.sum() == 0:
            crps_mean = np.nan
            n_crps = 0
        else:
            yt = y_target[mask_all_q].to_numpy()
            qhat = qdf.loc[mask_all_q].to_numpy()
            crps_vals = _crps_from_quantiles(yt, qhat, taus)
            crps_mean = float(np.mean(crps_vals))
            n_crps = int(mask_all_q.sum())

        crps_rows.append({"horizon": h, "CRPS": crps_mean, "N": n_crps})

    coverage_df = pd.DataFrame(coverage_rows).set_index("horizon")
    sharpness_df = pd.DataFrame(sharp_rows).set_index("horizon")
    crps_df = pd.DataFrame(crps_rows).set_index("horizon")
    calib_df = pd.DataFrame(calib_rows).set_index(["horizon", "quantile"]).sort_index()

    return {
        "coverage": coverage_df,
        "sharpness": sharpness_df,
        "calibration": calib_df,
        "crps": crps_df,
    }


# ----------------------------
# Plot helpers (optional)
# ----------------------------

def plot_det_horizon_metrics(metrics: pd.DataFrame, *, title: str = "Deterministic forecast accuracy"):
    """
    metrics: output of evaluate_ip_deterministic (index=horizon, columns MAE/RMSE)
    """
    import matplotlib.pyplot as plt

    horizons = metrics.index.to_numpy()
    plt.figure()
    if "MAE" in metrics.columns:
        plt.plot(horizons, metrics["MAE"].to_numpy(), marker="o", label="MAE")
    if "RMSE" in metrics.columns:
        plt.plot(horizons, metrics["RMSE"].to_numpy(), marker="o", label="RMSE")
    plt.xlabel("Horizon (qh)")
    plt.ylabel("Error (q0.5)")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_prob_coverage(coverage: pd.DataFrame, *, title: str = "Coverage by horizon"):
    import matplotlib.pyplot as plt

    horizons = coverage.index.to_numpy()
    cols = [c for c in coverage.columns if c.startswith("[q")]
    plt.figure()
    for c in cols:
        plt.plot(horizons, coverage[c].to_numpy(), marker="o", label=c)
    plt.xlabel("Horizon (qh)")
    plt.ylabel("Empirical coverage")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_calibration_curve(calibration: pd.DataFrame, *, horizon: int = 1, title: Optional[str] = None):
    """
    calibration: output dict['calibration'] from evaluate_ip_probabilistic
    plots empirical vs nominal for one horizon.
    """
    import matplotlib.pyplot as plt

    cal_h = calibration.xs(horizon, level="horizon").reset_index()
    cal_h = cal_h.sort_values("quantile")

    plt.figure()
    plt.plot(cal_h["quantile"].to_numpy(), cal_h["empirical"].to_numpy(), marker="o", label="Empirical")
    plt.plot(cal_h["quantile"].to_numpy(), cal_h["quantile"].to_numpy(), linestyle="--", label="Ideal")
    plt.xlabel("Nominal quantile")
    plt.ylabel("Empirical P(y ≤ q̂)")
    plt.title(title or f"Calibration curve (horizon qh{horizon})")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_prob_sharpness(
    sharpness: pd.DataFrame,
    *,
    title: str = "Sharpness (average interval width) by horizon",
    normalize: bool = False,
):
    """
    sharpness: prob_out["sharpness"] (index=horizon, columns interval labels + N)

    normalize=True: divide widths by |median| to reduce scale effects (optional).
    If you want a more principled normalization, use IQR or median absolute deviation.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    horizons = sharpness.index.to_numpy()
    cols = [c for c in sharpness.columns if c.startswith("[q")]

    plt.figure()
    for c in cols:
        y = sharpness[c].to_numpy(dtype=float)
        if normalize:
            # simple normalization: by median width over horizons (keeps dimensionless)
            denom = np.nanmedian(y)
            if np.isfinite(denom) and denom > 0:
                y = y / denom
        plt.plot(horizons, y, marker="o", label=c)

    plt.xlabel("Horizon (qh)")
    plt.ylabel("Avg interval width" + (" (normalized)" if normalize else " (€/MWh)"))
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_prob_crps(crps: pd.DataFrame, *, title: str = "CRPS by horizon"):
    import matplotlib.pyplot as plt

    horizons = crps.index.to_numpy()
    plt.figure()
    plt.plot(horizons, crps["CRPS"].to_numpy(dtype=float), marker="o", label="CRPS")
    plt.xlabel("Horizon (qh)")
    plt.ylabel("CRPS (€/MWh)")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
