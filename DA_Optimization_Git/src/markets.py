from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MarketName = Literal["da", "ip"]

@dataclass(frozen=True)
class MarketConfig:
    name: MarketName
    dt_minutes: int
    horizon_steps: int
    grid_fee_eur_per_mwh: float = 0.0
    allow_simultaneous_charge_discharge: bool = False

    def validate(self) -> None:
        if self.dt_minutes <= 0:
            raise ValueError("dt_minutes must be > 0")
        if self.horizon_steps <= 0:
            raise ValueError("horizon_steps must be > 0")
