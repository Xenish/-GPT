from __future__ import annotations

from itertools import product
from typing import Any, Callable, Dict, List, Sequence


def run_generic_grid_search(
    param_grid: Dict[str, Sequence[Any]],
    evaluate: Callable[[Dict[str, Any]], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Enumerate all combinations in param_grid and evaluate each via the provided callback.
    """
    if not param_grid:
        metrics = evaluate({})
        entry = {"params": {}}
        if metrics:
            entry.update(metrics)
        return [entry]

    keys = list(param_grid.keys())
    spaces: List[Sequence[Any]] = []
    for key in keys:
        values = param_grid.get(key)
        if values is None or len(values) == 0:
            raise ValueError(f"Param grid for '{key}' must provide at least one value.")
        spaces.append(values)

    results: List[Dict[str, Any]] = []
    for combo in product(*spaces):
        params = dict(zip(keys, combo))
        metrics = evaluate(dict(params)) or {}
        row = {"params": dict(params)}
        row.update(metrics)
        results.append(row)
    return results


__all__ = ["run_generic_grid_search"]
