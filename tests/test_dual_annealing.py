from numpy.testing import assert_allclose, assert_equal
import pytest

import numpy as np

from ez_optimize.dual_annealing import dual_annealing
from ez_optimize.utilities import EzOptimizeResult

# https://en.wikipedia.org/wiki/Rastrigin_function
def rastrigin(x, A=10):
    """Rastrigin function, a common test function for optimization algorithms."""
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
    """Rastrigin function in keyword mode."""
    x = [x1, x2]
    return rastrigin(x, A)

def rastrigin_kw_nd(A=10, **kwargs):
    """Rastrigin function in keyword mode for 9D."""
    x = [kwargs.get(f'x{i+1}', 0.0) for i in range(9)]
    return rastrigin(x, A)

@pytest.mark.parametrize("direction", ["min", "max"])
@pytest.mark.parametrize("opt_kwargs", [
    {},
    {'x0': {'x1': 4.0, 'x2': 4.0}},
    {'no_local_search': True},
    {'initial_temp': 10000, 'visit': 3.0},
    {'restart_temp_ratio': 1e-4, 'accept': -10.0},
    {'minimizer_kwargs': {'method': 'Nelder-Mead'}},
])
def test_dual_annealing_rastrigin_dict(direction, opt_kwargs: dict):
    bounds = {
        'x1': (-5.12, 5.12),
        'x2': (-5.12, 5.12)
    }
    res = dual_annealing(rastrigin_kw_2d, bounds=bounds, direction=direction, maxiter=1000, **opt_kwargs)

    # relax xtol if no_local_search is True
    xtol = 1e-2 if opt_kwargs.get('no_local_search') else 1e-4
    ftol = 1e-4 if opt_kwargs.get('no_local_search') else 1e-6

    assert isinstance(res, EzOptimizeResult)
    if direction == "min":
        assert_allclose(res.x_flat, np.zeros(2), atol=xtol)
        assert_allclose(res.func, 0.0, atol=ftol)
    else:
        # For maximization, the optimal value is 80.70658039 at (±4.52299366, ±4.52299366)
        assert_allclose(np.abs(res.x_flat), np.array([4.52299366, 4.52299366]), atol=xtol)
        assert_equal(res.x, {'x1': res.x_flat[0], 'x2': res.x_flat[1]})
        assert_allclose(res.func, rastrigin_max_lookup[2], atol=ftol)

@pytest.mark.parametrize("direction", ["min", "max"])
@pytest.mark.parametrize("bounds", [
    [(-5.12, 5.12), (-5.12, 5.12)],
    {'x': [(-5.12, 5.12), (-5.12, 5.12)]}
])
def test_dual_annealing_rastrigin_array(direction, bounds):
    res = dual_annealing(rastrigin, bounds=bounds, direction=direction, maxiter=1000)

    assert isinstance(res, EzOptimizeResult)
    if direction == "min":
        assert_allclose(res.x_flat, np.zeros(2), atol=1e-4)
        assert_allclose(res.func, 0.0, atol=1e-8)
    else:
        # For maximization, the optimal value is 80.70658039 at (±4.52299366, ±4.52299366)
        assert_allclose(np.abs(res.x_flat), np.array([4.52299366, 4.52299366]), atol=1e-4)
        assert_allclose(res.func, rastrigin_max_lookup[2], atol=1e-6)

    if isinstance(bounds, dict):
        assert_equal(res.x, {'x': res.x_flat})

@pytest.mark.parametrize("direction", ["min", "max"])
@pytest.mark.parametrize("dims", [1, 4, 7])
def test_dual_annealing_rastrigin_nd(direction, dims):
    bounds = {f"x{i+1}": (-5.12, 5.12) for i in range(dims)}
    res = dual_annealing(rastrigin_kw_nd, bounds=bounds, direction=direction, maxiter=2000)

    assert isinstance(res, EzOptimizeResult)
    if direction == "min":
        assert_allclose(res.x_flat, np.zeros(dims), atol=1e-4)
        assert_allclose(res.func, 0.0, atol=1e-6)
    else:
        # For maximization, the optimal value is [] at (±4.52299366, ..., ±4.52299366)
        assert_allclose(np.abs(res.x_flat), np.array([4.52299366] * dims), atol=1e-4)
        assert_allclose(res.func, rastrigin_max_lookup[dims], atol=1e-6)

    if isinstance(bounds, dict):
        assert_equal(res.x, {f"x{i+1}": res.x_flat[i] for i in range(dims)})
