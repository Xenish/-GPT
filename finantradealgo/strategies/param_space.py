from __future__ import annotations

import math
import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Literal

ScalarType = Literal["int", "float", "bool", "categorical"]


@dataclass
class ParamSpec:
    """
    Definition of a single strategy parameter search space.

    name must match the config key used throughout the system.
    """

    name: str
    type: ScalarType
    low: Optional[float] = None
    high: Optional[float] = None
    log: bool = False
    choices: Optional[Sequence[Any]] = None


ParamSpace = Dict[str, ParamSpec]


class ParamSpaceError(ValueError):
    """Raised when a parameter space definition is invalid."""


def validate_param_space(space: ParamSpace) -> None:
    for name, spec in space.items():
        if spec.name != name:
            raise ParamSpaceError(
                f"ParamSpec.name mismatch: key='{name}', spec.name='{spec.name}'"
            )

        if spec.type in ("int", "float"):
            if spec.low is None or spec.high is None:
                raise ParamSpaceError(
                    f"Numeric param '{name}' must have low/high set."
                )
            if spec.low >= spec.high:
                raise ParamSpaceError(
                    f"Param '{name}' has invalid range: low={spec.low}, high={spec.high}"
                )

        if spec.type == "categorical":
            if not spec.choices:
                raise ParamSpaceError(
                    f"Categorical param '{name}' must define choices."
                )


def sample_params(
    space: ParamSpace,
    rng: Optional[random.Random] = None,
) -> Dict[str, Any]:
    """
    Sample a single parameter set from the provided ParamSpace.
    """
    rng = rng or random
    out: Dict[str, Any] = {}

    for name, spec in space.items():
        if spec.type == "int":
            low = int(spec.low)  # validated previously
            high = int(spec.high)
            out[name] = rng.randint(low, high)
        elif spec.type == "float":
            low = float(spec.low)
            high = float(spec.high)
            if spec.log:
                u = rng.uniform(math.log10(low), math.log10(high))
                out[name] = 10 ** u
            else:
                out[name] = rng.uniform(low, high)
        elif spec.type == "bool":
            out[name] = bool(rng.getrandbits(1))
        elif spec.type == "categorical":
            if not spec.choices:
                raise ParamSpaceError(
                    f"Param '{name}' is categorical but choices is empty."
                )
            out[name] = rng.choice(list(spec.choices))
        else:
            raise ParamSpaceError(
                f"Unknown param type '{spec.type}' for param '{name}'"
            )

    return out


def apply_strategy_params_to_cfg(
    cfg: Dict[str, Any],
    strategy_name: str,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Merge strategy-specific params into a config copy without mutating the original.
    """
    cfg_local = deepcopy(cfg)
    strategy_section = cfg_local.setdefault("strategy", {})
    strategy_block = strategy_section.setdefault(strategy_name, {})
    strategy_block.update(params or {})
    return cfg_local
