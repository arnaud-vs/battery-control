# adapters: deterministic vs probabilistic -> scenarios/expectations

# src/ip/forecast.py
from __future__ import annotations

from typing import Iterable, Tuple
import numpy as np
import pandas as pd


def deterministic_vector_from_row(f_row: pd.Series, horizon_steps: int = 8) -> np.ndarray:
    """
    Extract p[t] from columns qh1..qh8 from a single row.
    Returns shape (T,).
    """
    cols = [f"qh{i}" for i in range(1, horizon_steps + 1)]
    missing = [c for c in cols if c not in f_row.index]
    if missing:
        raise ValueError(f"Missing deterministic columns: {missing}. Available: {list(f_row.index)[:50]}")
    return f_row[cols].astype(float).to_numpy()


def scenario_matrix_from_quantiles_row(
    q_row: pd.Series,
    quantiles: Iterable[float],
    horizon_steps: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build scenarios from quantiles treating each quantile as one scenario.

    Returns
    -------
    P : ndarray shape (T, S) where S=len(quantiles)
    w : ndarray shape (S,) equal weights (1/S)
    """
    qs = [float(q) for q in quantiles]
    S = len(qs)
    T = horizon_steps
    P = np.zeros((T, S), dtype=float)

    for s, q in enumerate(qs):
        q_str = f"{q:g}"
        cols = [f"qh{t}_q{q_str}" for t in range(1, T + 1)]
        missing = [c for c in cols if c not in q_row.index]
        if missing:
            raise ValueError(
                f"Missing probabilistic columns for q={q_str}: {missing}. "
                f"Available: {list(q_row.index)[:60]}"
            )
        P[:, s] = q_row[cols].astype(float).to_numpy()

    w = np.ones(S, dtype=float) / S
    return P, w
