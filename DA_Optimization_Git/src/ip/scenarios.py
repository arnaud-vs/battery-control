import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Optional, Sequence

DT_DEFAULT = pd.Timedelta(minutes=15)

def _get_qcols(k: int, quantiles: Sequence[float]) -> list[str]:
    return [f"qh{k}_q{q:.1f}" for q in quantiles]

def inverse_cdf_from_quantiles(u: float, quantiles: np.ndarray, values: np.ndarray) -> float:
    q_ext = np.concatenate(([0.0], quantiles, [1.0]))
    v_ext = np.concatenate(([values[0]], values, [values[-1]]))
    return float(np.interp(u, q_ext, v_ext))

def compute_gaussian_errors_with_pit(
    ip_prob: pd.DataFrame,
    ip_real: pd.Series,
    *,
    K: int,
    quantiles: np.ndarray,
    as_of: pd.Timestamp,
    dt: pd.Timedelta = DT_DEFAULT,
):
    """
    STRICT no-leak:
      only use forecast origins s such that s + (K-1)dt < as_of  <=>  s <= as_of - K*dt
    """
    ip_prob = ip_prob.sort_index()
    ip_real = ip_real.sort_index()

    cutoff = pd.to_datetime(as_of) - K * dt
    ip_prob = ip_prob.loc[ip_prob.index <= cutoff]

    X_rows, idx = [], []
    for s in ip_prob.index:
        x_s = []
        ok = True
        for k in range(1, K + 1):
            t_real = s + (k - 1) * dt
            if t_real not in ip_real.index:
                ok = False
                break

            p_real = float(ip_real.loc[t_real])
            q_vals = ip_prob.loc[s, _get_qcols(k, quantiles)].values.astype(float)
            q_vals = np.maximum.accumulate(q_vals)

            u = np.interp(p_real, q_vals, quantiles, left=0.0, right=1.0)
            u = float(np.clip(u, 1e-6, 1 - 1e-6))
            x_s.append(float(norm.ppf(u)))

        if ok and len(x_s) == K:
            idx.append(s)
            X_rows.append(x_s)

    X_df = pd.DataFrame(X_rows, index=pd.DatetimeIndex(idx), columns=[f"X_qh{k}" for k in range(1, K + 1)])
    return X_df

def estimate_covariance_ewma(X_df: pd.DataFrame, lam: float) -> np.ndarray:
    X = X_df.values
    K = X.shape[1]
    Sigma = np.eye(K)
    for xt in X:
        xt = xt.reshape(-1, 1)
        Sigma = lam * Sigma + (1 - lam) * (xt @ xt.T)
        std = np.sqrt(np.diag(Sigma))
        Sigma = Sigma / np.outer(std, std)
    return Sigma

def sigma_as_of(
    ip_prob_hist: pd.DataFrame,
    ip_real: pd.Series,
    *,
    t_now: pd.Timestamp,
    K: int,
    quantiles: np.ndarray,
    lam: float = 0.995,
    dt: pd.Timedelta = DT_DEFAULT,
    lookback: Optional[pd.Timedelta] = pd.Timedelta("30D"),
) -> np.ndarray:
    X_df = compute_gaussian_errors_with_pit(
        ip_prob_hist, ip_real, K=K, quantiles=quantiles, as_of=t_now, dt=dt
    )
    if lookback is not None and not X_df.empty:
        X_df = X_df.loc[X_df.index >= (pd.to_datetime(t_now) - K * dt - lookback)]
    if X_df.empty:
        return np.eye(K)
    return estimate_covariance_ewma(X_df, lam=lam)

def generate_copula_scenarios_at_t(
    ip_prob_row: pd.Series,
    *,
    Sigma: np.ndarray,
    K: int,
    quantiles: np.ndarray,
    n_scenarios: int,
    seed: int,
) -> np.ndarray:
    """
    Returns scen_prices: shape (n_scenarios, K) for horizons 1..K.
    """
    rng = np.random.default_rng(seed)
    Xs = rng.multivariate_normal(mean=np.zeros(K), cov=Sigma, size=n_scenarios)
    Ys = norm.cdf(Xs)  # correlated uniforms in (0,1)

    scen = np.zeros((n_scenarios, K), dtype=float)
    for k in range(1, K + 1):
        q_vals = ip_prob_row[_get_qcols(k, quantiles)].values.astype(float)
        q_vals = np.maximum.accumulate(q_vals)
        for s in range(n_scenarios):
            scen[s, k - 1] = inverse_cdf_from_quantiles(float(Ys[s, k - 1]), quantiles, q_vals)
    return scen
