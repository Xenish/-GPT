from __future__ import annotations

from research.grid_search import run_generic_grid_search


def test_run_generic_grid_search_cartesian():
    param_grid = {"a": [1, 2], "b": ["x"]}

    def evaluate(params):
        return {"score": params["a"], "label": f"{params['a']}{params['b']}"}

    results = run_generic_grid_search(param_grid, evaluate)
    assert len(results) == 2
    assert results[0]["params"]["a"] in {1, 2}
    assert "score" in results[0]


def test_run_generic_grid_search_empty_grid():
    results = run_generic_grid_search({}, lambda params: {"score": 1.0})
    assert len(results) == 1
    assert results[0]["params"] == {}
    assert results[0]["score"] == 1.0
