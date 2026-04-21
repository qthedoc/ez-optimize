from __future__ import annotations

import numpy as np
from numpy.testing import assert_equal
import pytest
from unittest.mock import Mock

from ez_optimize.basinhopping import basinhopping
# from ez_optimize.brute import brute
from ez_optimize.differential_evolution import differential_evolution
from ez_optimize.direct import direct
from ez_optimize.dual_annealing import dual_annealing
from ez_optimize.shgo import shgo
from ez_optimize.utilities import EzOptimizeResult

# https://en.wikipedia.org/wiki/Rastrigin_function
def rastrigin(x, A=10):
    """Rastrigin function, a common test function for global optimization algorithms."""
    n = len(x)
    return A * n + sum(xi**2 - A * np.cos(2 * np.pi * xi) for xi in x)

rastrigin_max_lookup = {
    1 : 40.35329019,
    2 : 80.70658039,
    3 : 121.0598706,
    4 : 161.4131608,
    5 : 201.7664509,
    6 : 242.1197412,
    7 : 282.4730314,
    8 : 322.8263216,
    9 : 363.1796117,
}

def rastrigin_kw_2d(x1, x2, A=10):
    """2D Rastrigin function in keyword mode."""
    x = [x1, x2]
    return rastrigin(x, A)

MODE_OPTIONS = {
    "dict": {
        "func": rastrigin_kw_2d,
        "x0": {'x1': 4.0, 'x2': 4.0},
        "bounds": {"x1": (-5.12, 5.12), "x2": (-5.12, 5.12)},
    },
    "array": {
        "func": rastrigin,
        "x0": np.array([4.0, 4.0]),
        "bounds": [(-5.12, 5.12), (-5.12, 5.12)],
    },
    "dict-array-value": {
        "func": rastrigin,
        "x0": {'x': np.array([4.0, 4.0])},
        "bounds": {'x': [(-5.12, 5.12), (-5.12, 5.12)]},
    },
}

def _assert_result_structure_2d(res, mode: str):
    assert isinstance(res, EzOptimizeResult)
    assert np.asarray(res.x_flat).shape == (2,)
    assert np.isfinite(res.func)
    assert np.all(np.isfinite(res.x_flat))

    if mode == "dict":
        assert isinstance(res.x, dict)
        assert set(res.x.keys()) == {"x1", "x2"}
    if mode == "dict-array-value":
        assert isinstance(res.x, dict)
        assert set(res.x.keys()) == {"x"}
        assert isinstance(res.x['x'], np.ndarray)
        assert_equal(res.x['x'], res.x_flat)
    else:
        assert isinstance(res.x_flat, np.ndarray)


def _run_differential_evolution(mode: str, **extra):
    kw = {"maxiter": 1000, "popsize": 25, "rng": 42}
    kw.update(bounds=MODE_OPTIONS[mode]["bounds"])
    kw.update(extra)
    return differential_evolution(
        MODE_OPTIONS[mode]["func"],
        **kw
    )

def _run_dual_annealing(mode: str, **extra):
    kw = {"maxiter": 120, "rng": 42}
    kw.update(bounds=MODE_OPTIONS[mode]["bounds"])
    kw.update(extra)
    return dual_annealing(
        MODE_OPTIONS[mode]["func"],
        **kw
    )

def _run_shgo(mode: str, **extra):
    callback = extra.get("callback")
    iters = 3 if callback is not None else 1

    kw = {"n": 256, "iters": iters}
    kw.update(bounds=MODE_OPTIONS[mode]["bounds"])
    kw.update(extra)

    return shgo(
        MODE_OPTIONS[mode]["func"],
        **kw,
    )


def _run_direct(mode: str, **extra):
    kw = {"maxiter": 300}
    kw.update(bounds=MODE_OPTIONS[mode]["bounds"])
    kw.update(extra)
    return direct(
        MODE_OPTIONS[mode]["func"],
        **kw,
    )


# def _run_brute(mode: str, **extra):
#     kw = {"Ns": 11, "finish": None}
#     kw.update(ranges=MODE_OPTIONS[mode]["bounds"])
#     kw.update(extra)

#     return brute(
#         MODE_OPTIONS[mode]["func"],
#         **kw,
#     )


def _run_basinhopping(mode: str, **extra):

    kw = {"niter": 60, "rng": 42}
    kw.update(x0=MODE_OPTIONS[mode]["x0"])
    kw.update(extra)
    return basinhopping(
        MODE_OPTIONS[mode]["func"], 
        **kw
    )


GLOBAL_RUNNERS = {
    "differential_evolution": _run_differential_evolution,
    "dual_annealing": _run_dual_annealing,
    "shgo": _run_shgo,
    "direct": _run_direct,
    # "brute": _run_brute,
    "basinhopping": _run_basinhopping,
}

CALLBACK_METHODS = {
    "differential_evolution",
    "dual_annealing",
    "shgo",
    "direct",
    "basinhopping",
}


@pytest.mark.parametrize("method_name", list(GLOBAL_RUNNERS.keys()))
@pytest.mark.parametrize("mode", ["array", "dict"])
@pytest.mark.parametrize("direction", ["min", "max"])
def test_global_method_smoke_result_shape(method_name: str, mode: str, direction: str):
    runner = GLOBAL_RUNNERS[method_name]
    res = runner(mode=mode, direction=direction)
    _assert_result_structure_2d(res, mode)


def _make_callback(method_name: str, callback_mock: callable):
    if method_name == "differential_evolution":
        def cb(x, convergence=None):
            callback_mock(x, convergence)
        return cb

    if method_name == "dual_annealing":
        def cb(x, f, context):
            callback_mock(x, f, context)
        return cb

    if method_name in {"shgo", "direct"}:
        def cb(xk):
            callback_mock(xk)
        return cb

    if method_name == "basinhopping":
        def cb(x, f, accept):
            callback_mock(x, f, accept)
        return cb

    raise ValueError(f"Unsupported callback method: {method_name}")


def _assert_callback_args(method_name: str, mode: str, call):
    args = call[0]

    # Helper for the first positional argument (x/xk).
    def _assert_x_shape_and_type(x_arg):
        if mode == "dict":
            assert isinstance(x_arg, dict)
            assert set(x_arg.keys()) == {"x1", "x2"}
        else:
            assert isinstance(x_arg, np.ndarray)
            assert x_arg.shape == (2,)

    if method_name == "differential_evolution":
        assert len(args) == 2
        _assert_x_shape_and_type(args[0])
        assert args[1] is None or np.isscalar(args[1])
        return

    if method_name == "dual_annealing":
        assert len(args) == 3
        _assert_x_shape_and_type(args[0])
        assert np.isscalar(args[1])
        assert np.isfinite(args[1])
        assert args[2] in {0, 1, 2}
        return

    if method_name in {"shgo", "direct"}:
        assert len(args) == 1
        _assert_x_shape_and_type(args[0])
        return

    if method_name == "basinhopping":
        assert len(args) == 3
        _assert_x_shape_and_type(args[0])
        assert np.isscalar(args[1])
        assert np.isfinite(args[1])
        assert isinstance(args[2], (bool, np.bool_))
        return

    raise AssertionError(f"Unsupported callback method: {method_name}")


@pytest.mark.parametrize("method_name", sorted(CALLBACK_METHODS))
@pytest.mark.parametrize("mode", ["array", "dict"])
def test_global_method_callback_adaptation(method_name: str, mode: str):
    runner = GLOBAL_RUNNERS[method_name]
    callback_mock = Mock()
    callback = _make_callback(method_name, callback_mock)

    res = runner(mode=mode, direction="min", callback=callback)
    _assert_result_structure_2d(res, mode)

    assert callback_mock.call_count > 0
    for call in callback_mock.call_args_list:
        _assert_callback_args(method_name, mode, call)


METHOD_OPTION_CASES = {
    "differential_evolution": [
        {"strategy": "rand1bin"},
        {'popsize': 30, 'mutation': 0.8, 'recombination': 0.9},
        {'init': 'random'},
        {'tol': 1e-6, 'atol': 1e-6},
        {"polish": False},
        {'updating': 'deferred'},
    ],
    "dual_annealing": [
        {"no_local_search": True},
        {"visit": 2.8},
        {'initial_temp': 10000, 'visit': 3.0},
        {'restart_temp_ratio': 1e-4, 'accept': -10.0},
        {'minimizer_kwargs': {'method': 'Nelder-Mead'}}
    ],
    "shgo": [
        {"sampling_method": "sobol"},
        {"options": {"maxiter": 5}},
    ],
    "direct": [
        {"locally_biased": False},
        {"eps": 1e-5},
    ],
    # "brute": [
    #     {"Ns": 9},
    #     {"disp": False},
    # ],
    "basinhopping": [
        {"T": 2.0, "stepsize": 0.2},
        {"niter_success": 8},
    ],
}

_OPTION_PARAMS = [
    (method_name, opt_kwargs)
    for method_name, cases in METHOD_OPTION_CASES.items()
    for opt_kwargs in cases
]


@pytest.mark.parametrize("method_name,opt_kwargs", _OPTION_PARAMS)
@pytest.mark.parametrize("mode", ["array", "dict"])
def test_global_method_accepts_method_options(method_name: str, opt_kwargs: dict, mode: str):
    runner = GLOBAL_RUNNERS[method_name]
    res = runner(mode=mode, direction="min", **opt_kwargs)
    _assert_result_structure_2d(res, mode)



