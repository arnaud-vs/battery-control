from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import yaml

from .battery import BatterySpec
from .markets import MarketConfig, MarketName


def load_config(config_path: str | Path) -> Tuple[BatterySpec, Dict[MarketName, MarketConfig]]:
    """
    Load YAML config into typed objects.

    Expected YAML structure
    -----------------------
    battery: { ... BatterySpec fields ... }
    markets:
      da: { ... MarketConfig fields except name ... }
      ip: { ... MarketConfig fields except name ... }

    Returns
    -------
    battery : BatterySpec
    markets : dict with keys "da" and "ip" mapping to MarketConfig
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping/dict. Got: {type(raw)}")

    if "battery" not in raw:
        raise ValueError("Missing top-level key: 'battery'")
    if "markets" not in raw:
        raise ValueError("Missing top-level key: 'markets'")

    # ---- Battery ----
    battery_cfg = raw["battery"]
    if not isinstance(battery_cfg, dict):
        raise ValueError(f"'battery' must be a mapping/dict. Got: {type(battery_cfg)}")

    try:
        battery = BatterySpec(**battery_cfg)
    except TypeError as e:
        raise ValueError(f"Battery config does not match BatterySpec fields: {e}") from e

    battery.validate()

    # ---- Markets ----
    markets_cfg = raw["markets"]
    if not isinstance(markets_cfg, dict):
        raise ValueError(f"'markets' must be a mapping/dict. Got: {type(markets_cfg)}")

    markets: Dict[MarketName, MarketConfig] = {}

    for name in ("da", "ip"):
        if name not in markets_cfg:
            raise ValueError(f"Missing market config for '{name}' under 'markets:'")

        cfg = markets_cfg[name]
        if not isinstance(cfg, dict):
            raise ValueError(f"markets.{name} must be a mapping/dict. Got: {type(cfg)}")

        try:
            m = MarketConfig(name=name, **cfg)
        except TypeError as e:
            raise ValueError(f"Market '{name}' config does not match MarketConfig fields: {e}") from e

        m.validate()
        markets[name] = m

    return battery, markets
