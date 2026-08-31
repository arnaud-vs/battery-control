# src/ip/objectives.py
from __future__ import annotations

from typing import Literal, Optional
import pyomo.environ as pyo
import numpy as np

TerminalPenaltyMode = Literal["none", "L1", "L2"]


def ensure_price_param(m: pyo.ConcreteModel) -> None:
    if not hasattr(m, "price"):
        m.price = pyo.Param(m.T, initialize=0.0, mutable=True)


def ensure_scenario_price_param(
    m: pyo.ConcreteModel,
    *,
    num_scenarios: int,
) -> None:
    """
    Ensure the model has:
      - m.S: RangeSet(0..S-1)
      - m.scen_price[t,s]: mutable Param for scenario prices (EUR/MWh)

    Call once when building the model.
    """
    if num_scenarios <= 0:
        raise ValueError("num_scenarios must be > 0")

    if not hasattr(m, "S"):
        m.S = pyo.RangeSet(0, num_scenarios - 1)
    else:
        # validate existing size
        try:
            existing_S = len(list(m.S))
            if existing_S != num_scenarios:
                raise ValueError(
                    f"Model already defines m.S with {existing_S} scenarios, "
                    f"but requested {num_scenarios}."
                )
        except Exception:
            pass

    if not hasattr(m, "scen_price"):
        m.scen_price = pyo.Param(m.T, m.S, initialize=0.0, mutable=True)


def _clear_existing_objective_and_penalty(m: pyo.ConcreteModel) -> None:
    # Replace existing objective
    if hasattr(m, "obj"):
        m.del_component(m.obj)

    # Clean up old penalty components if switching modes
    for comp in ["E_target", "lambda_terminal", "dev_pos", "dev_neg", "terminal_dev"]:
        if hasattr(m, comp):
            m.del_component(comp)


def _build_terminal_penalty_expr(
    m: pyo.ConcreteModel,
    *,
    terminal_target_kwh: Optional[float],
    terminal_penalty: float,
    terminal_penalty_mode: TerminalPenaltyMode,
):
    """
    Returns a Pyomo expression for the terminal penalty and, if needed,
    creates the required params/vars/constraints on the model.

    - L1: uses dev_pos/dev_neg and constraint terminal_dev
    - L2: adds L2 term (MIQP if binaries exist)
    """
    penalty_expr = 0.0

    if terminal_penalty_mode != "none":
        if terminal_target_kwh is None:
            raise ValueError("terminal_target_kwh must be set when terminal_penalty_mode != 'none'")
        if terminal_penalty <= 0.0:
            raise ValueError("terminal_penalty must be > 0 when terminal_penalty_mode != 'none'")

    use_penalty = (
        terminal_penalty_mode != "none"
        and terminal_target_kwh is not None
        and terminal_penalty > 0.0
    )

    if not use_penalty:
        return penalty_expr

    # Use plain floats as coefficients so Pyomo can determine polynomial degree (needed for appsi_highs).
    # Mutable Param * Var² gives degree=None; float * Var² gives degree=2.
    # m.E_target = pyo.Param(initialize=float(terminal_target_kwh), mutable=True)
    # m.lambda_terminal = pyo.Param(initialize=float(terminal_penalty), mutable=True)
    lam = float(terminal_penalty)
    target = float(terminal_target_kwh)

    T_end = max(m.Tp1)  # horizon_steps

    if terminal_penalty_mode == "L1":
        m.dev_pos = pyo.Var(domain=pyo.NonNegativeReals)
        m.dev_neg = pyo.Var(domain=pyo.NonNegativeReals)
        # m.terminal_dev = pyo.Constraint(expr=m.E[T_end] - m.E_target == m.dev_pos - m.dev_neg)
        # penalty_expr = m.lambda_terminal * (m.dev_pos + m.dev_neg)
        m.terminal_dev = pyo.Constraint(expr=m.E[T_end] - target == m.dev_pos - m.dev_neg)
        penalty_expr = lam * (m.dev_pos + m.dev_neg)

    elif terminal_penalty_mode == "L2":
        # penalty_expr = m.lambda_terminal * (m.E[T_end] - m.E_target) ** 2
        penalty_expr = lam * (m.E[T_end] - target) ** 2

    else:
        raise ValueError(f"Unknown terminal_penalty_mode: {terminal_penalty_mode}")

    return penalty_expr

def _build_cycling_penalty_expr(
    m: pyo.ConcreteModel,
    *,
    cycle_penalty_eur_per_mwh: float,
) -> float:
    """
    Linear penalty on energy throughput to discourage cycling/chatter.

    cycle_penalty_eur_per_mwh: €/MWh of throughput (charge + discharge).
    Model vars e_ch/e_dis are in kWh, so convert €/MWh -> €/kWh by /1000.
    """
    if cycle_penalty_eur_per_mwh <= 0.0:
        return 0.0

    lam = float(cycle_penalty_eur_per_mwh) / 1000.0  # €/kWh
    return lam * sum(m.e_ch[t] + m.e_dis[t] for t in m.T)


def add_objective_certainty_equivalent(
    m: pyo.ConcreteModel,
    *,
    grid_fee_eur_per_mwh: float = 0.0,
    terminal_target_kwh: Optional[float] = None,
    terminal_penalty: float = 0.0,                 # λ
    terminal_penalty_mode: TerminalPenaltyMode = "none",
    cycle_penalty_eur_per_mwh: float = 0.0,  
) -> None:
    """
    CE objective with optional terminal SOC soft penalty.

    Profit (EUR):
      sum_t ((price[t] - fee)/1000) * (e_dis[t] - e_ch[t])

    Terminal penalty at horizon end (EUR):
      - L1:        λ * |E[T] - target|
      - L2: λ * (E[T] - target)^2
    """
    ensure_price_param(m)
    _clear_existing_objective_and_penalty(m)

    penalty_expr = _build_terminal_penalty_expr(
        m,
        terminal_target_kwh=terminal_target_kwh,
        terminal_penalty=terminal_penalty,
        terminal_penalty_mode=terminal_penalty_mode,
    )

    cycling_expr = _build_cycling_penalty_expr(m, cycle_penalty_eur_per_mwh=cycle_penalty_eur_per_mwh)


    def _obj(mm):
        profit = sum(
            ((mm.price[t] - grid_fee_eur_per_mwh) / 1000.0) * (mm.e_dis[t] - mm.e_ch[t])
            for t in mm.T
        )
        return profit - penalty_expr - cycling_expr

    m.obj = pyo.Objective(rule=_obj, sense=pyo.maximize)


def add_objective_mean_cvar(
    m: pyo.ConcreteModel,
    scenario_weights: np.ndarray,             # shape (S,)
    *,
    grid_fee_eur_per_mwh: float = 0.0,
    alpha: float = 0.9,                       # CVaR confidence level
    lambda_cvar: float = 0.0,                 # risk aversion (0 => risk-neutral expected profit)
    terminal_target_kwh: Optional[float] = None,
    terminal_penalty: float = 0.0,
    terminal_penalty_mode: TerminalPenaltyMode = "none",
    cycle_penalty_eur_per_mwh: float = 0.0,   # NEW
) -> None:
    """
    Mean–CVaR objective with mutable scenario prices m.scen_price[t,s].

    Maximize: E[profit] - lambda_cvar * CVaR_alpha(loss) - terminal_penalty_expr
    where loss_s = - profit_s.

    Requires:
      - ensure_scenario_price_param(m, num_scenarios=S) called beforehand
      - you update m.scen_price[t,s] each rolling step
    """
    # ---- cleanup objective + penalty + any previous CVaR components
    if hasattr(m, "obj"):
        m.del_component(m.obj)

    for comp in [
        "E_target", "lambda_terminal", "dev_pos", "dev_neg", "terminal_dev",   # terminal penalty parts
        "cvar_eta", "cvar_z", "cvar_constr",                                   # CVaR parts
    ]:
        if hasattr(m, comp):
            m.del_component(getattr(m, comp))

    if not hasattr(m, "S") or not hasattr(m, "scen_price"):
        raise ValueError("Model missing m.S and/or m.scen_price. Call ensure_scenario_price_param first.")

    # ---- validate alpha / lambda
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1)")
    if lambda_cvar < 0.0:
        raise ValueError("lambda_cvar must be >= 0")

    # ---- weights
    w = np.asarray(scenario_weights, dtype=float)
    if w.ndim != 1:
        raise ValueError("scenario_weights must be 1D")
    if not np.isfinite(w).all():
        raise ValueError("scenario_weights contains non-finite values")
    if w.sum() <= 0:
        raise ValueError("scenario_weights must sum to > 0")
    w = w / w.sum()

    S_model = len(list(m.S))
    if len(w) != S_model:
        raise ValueError(f"scenario_weights length {len(w)} does not match model scenarios {S_model}")

    # ---- terminal penalty (deterministic; same across scenarios)
    penalty_expr = _build_terminal_penalty_expr(
        m,
        terminal_target_kwh=terminal_target_kwh,
        terminal_penalty=terminal_penalty,
        terminal_penalty_mode=terminal_penalty_mode,
    )

    cycling_expr = _build_cycling_penalty_expr(m, cycle_penalty_eur_per_mwh=cycle_penalty_eur_per_mwh)

    # ---- scenario profit expressions (as Pyomo expressions)
    def _profit_s(mm, s):
        return sum(
            ((mm.scen_price[t, s] - grid_fee_eur_per_mwh) / 1000.0) * (mm.e_dis[t] - mm.e_ch[t])
            for t in mm.T
        )

    # ---- CVaR auxiliary variables/constraints (only if lambda_cvar > 0)
    # CVaR(loss) with loss_s = -profit_s
    if lambda_cvar > 0.0:
        m.cvar_eta = pyo.Var(domain=pyo.Reals)
        m.cvar_z = pyo.Var(m.S, domain=pyo.NonNegativeReals)

        def _cvar_constr_rule(mm, s):
            loss_s = -_profit_s(mm, s)
            return mm.cvar_z[s] >= loss_s - mm.cvar_eta

        m.cvar_constr = pyo.Constraint(m.S, rule=_cvar_constr_rule)

        cvar_expr = m.cvar_eta + (1.0 / (1.0 - alpha)) * sum(
            float(w[int(s)]) * m.cvar_z[s] for s in m.S
        )
    else:
        cvar_expr = 0.0

    # ---- expected profit
    exp_profit = sum(
        float(w[int(s)]) * _profit_s(m, s) for s in m.S
    )

    # ---- objective
    m.obj = pyo.Objective(
        expr=exp_profit - penalty_expr - cycling_expr - float(lambda_cvar) * cvar_expr,
        sense=pyo.maximize,
    )
