# src/battery.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class BatterySpec:
    name: str
    energy_kwh: float
    p_charge_kw_max: float
    p_discharge_kw_max: float
    eta_charge: float
    eta_discharge: float
    soc_min: float
    soc_max: float
    soc_init: float
    soc_target: Optional[float] = None

    def validate(self) -> None:
        if self.energy_kwh <= 0:
            raise ValueError("energy_kwh must be > 0")
        if self.p_charge_kw_max < 0 or self.p_discharge_kw_max < 0:
            raise ValueError("Power limits must be >= 0")
        if not (0 < self.eta_charge <= 1) or not (0 < self.eta_discharge <= 1):
            raise ValueError("Efficiencies must be in (0,1]")
        if not (0 <= self.soc_min < self.soc_max <= 1):
            raise ValueError("SOC bounds must satisfy 0<=soc_min<soc_max<=1")
        if not (self.soc_min <= self.soc_init <= self.soc_max):
            raise ValueError("soc_init must be within [soc_min, soc_max]")
        if self.soc_target is not None and not (self.soc_min <= self.soc_target <= self.soc_max):
            raise ValueError("soc_target must be within [soc_min, soc_max]")

    @property
    def e_min_kwh(self) -> float:
        return self.soc_min * self.energy_kwh

    @property
    def e_max_kwh(self) -> float:
        return self.soc_max * self.energy_kwh

    @property
    def e_init_kwh(self) -> float:
        return self.soc_init * self.energy_kwh

    def discretize(self, dt_minutes: int) -> "BatteryDiscrete":
        """
        Convert power limits (kW) into per-timestep energy limits (kWh) for a time step dt.
        """
        self.validate()
        dt_h = dt_minutes / 60.0
        return BatteryDiscrete(
            spec=self,
            dt_minutes=dt_minutes,
            dt_hours=dt_h,
            e_min_kwh=self.e_min_kwh,
            e_max_kwh=self.e_max_kwh,
            e_init_kwh=self.e_init_kwh,
            e_charge_step_max_kwh=self.p_charge_kw_max * dt_h,
            e_discharge_step_max_kwh=self.p_discharge_kw_max * dt_h,
        )

@dataclass(frozen=True)
class BatteryDiscrete:
    spec: BatterySpec
    dt_minutes: int
    dt_hours: float
    e_min_kwh: float
    e_max_kwh: float
    e_init_kwh: float
    e_charge_step_max_kwh: float
    e_discharge_step_max_kwh: float
