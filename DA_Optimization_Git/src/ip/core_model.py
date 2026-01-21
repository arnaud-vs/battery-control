# src/ip/core_model.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pyomo.environ as pyo

from src.battery import BatteryDiscrete
from src.markets import MarketConfig


@dataclass(frozen=True)
class IPCore:
    model: pyo.ConcreteModel
    T: range


def build_ip_core(
    bd: BatteryDiscrete,
    market: MarketConfig,
    horizon_steps: int = 8,
    *,
    enforce_no_simultaneous: bool = True,
) -> IPCore:
    """
    Build the IP battery dispatch model core (shared constraints).
    Decision variables are energy per step (kWh): e_ch[t], e_dis[t].
    State variable: E[t] (kWh), for t=0..T.
    """
    if market.dt_minutes <= 0:
        raise ValueError("market.dt_minutes must be > 0")

    m = pyo.ConcreteModel("IP_Battery_Core")
    T = range(horizon_steps)           # 0..T-1
    Tp1 = range(horizon_steps + 1)     # 0..T

    m.T = pyo.RangeSet(0, horizon_steps - 1)
    m.Tp1 = pyo.RangeSet(0, horizon_steps)

    # Variables
    m.e_ch = pyo.Var(m.T, domain=pyo.NonNegativeReals)   # kWh charged in step t
    m.e_dis = pyo.Var(m.T, domain=pyo.NonNegativeReals)  # kWh discharged in step t
    m.E = pyo.Var(m.Tp1, domain=pyo.NonNegativeReals)

    # Optional: enforce no simultaneous charge/discharge with a binary
    # If you prefer LP relaxation, set enforce_no_simultaneous=False
    if enforce_no_simultaneous:
        m.u = pyo.Var(m.T, domain=pyo.Binary)

    # Parameters (constants)
    m.E_min = pyo.Param(initialize=bd.e_min_kwh)
    m.E_max = pyo.Param(initialize=bd.e_max_kwh)
    m.E_init = pyo.Param(initialize=bd.e_init_kwh, mutable=True)
    m.e_ch_max = pyo.Param(initialize=bd.e_charge_step_max_kwh)
    m.e_dis_max = pyo.Param(initialize=bd.e_discharge_step_max_kwh)
    m.eta_c = pyo.Param(initialize=bd.spec.eta_charge)
    m.eta_d = pyo.Param(initialize=bd.spec.eta_discharge)

    # Initial SOC
    def _init_soc_rule(mm):
        return mm.E[0] == mm.E_init
    m.init_soc = pyo.Constraint(rule=_init_soc_rule)

    # Bounds on E[t]
    def _soc_bounds_rule(mm, t):
        return pyo.inequality(mm.E_min, mm.E[t], mm.E_max)
    m.soc_bounds = pyo.Constraint(m.Tp1, rule=_soc_bounds_rule)

    # Power/energy per-step bounds
    def _ch_max_rule(mm, t):
        return mm.e_ch[t] <= mm.e_ch_max
    m.ch_max = pyo.Constraint(m.T, rule=_ch_max_rule)

    def _dis_max_rule(mm, t):
        return mm.e_dis[t] <= mm.e_dis_max
    m.dis_max = pyo.Constraint(m.T, rule=_dis_max_rule)

    # No simultaneous charge/discharge (MILP)
    if enforce_no_simultaneous:
        def _no_sim_ch_rule(mm, t):
            return mm.e_ch[t] <= mm.e_ch_max * mm.u[t]
        def _no_sim_dis_rule(mm, t):
            return mm.e_dis[t] <= mm.e_dis_max * (1 - mm.u[t])
        m.no_sim_ch = pyo.Constraint(m.T, rule=_no_sim_ch_rule)
        m.no_sim_dis = pyo.Constraint(m.T, rule=_no_sim_dis_rule)

    # Dynamics: E[t+1] = E[t] + eta_c * e_ch[t] - (1/eta_d) * e_dis[t]
    def _dyn_rule(mm, t):
        return mm.E[t + 1] == mm.E[t] + mm.eta_c * mm.e_ch[t] - (1 / mm.eta_d) * mm.e_dis[t]
    m.dynamics = pyo.Constraint(m.T, rule=_dyn_rule)

    return IPCore(model=m, T=T)
