from numpy.testing import assert_allclose, assert_equal
import pytest

import numpy as np

from ez_optimize.differential_evolution import differential_evolution
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
    {'polish': False},
    {'strategy': 'rand1bin'},
    {'popsize': 30, 'mutation': 0.8, 'recombination': 0.9},
    {'init': 'random'},
    {'tol': 1e-6, 'atol': 1e-6},
    {'updating': 'deferred'},
])
def test_differential_evolution_rastrigin_dict(direction, opt_kwargs: dict):
    bounds = {
        'x1': (-5.12, 5.12),
        'x2': (-5.12, 5.12)
    }
    kw = {'popsize': 25}
    kw.update(opt_kwargs)
    res = differential_evolution(rastrigin_kw_2d, bounds=bounds, direction=direction, maxiter=1000, rng=42, **kw)

    # relax tolerance if polish is False
    xtol = 5e-2 if not opt_kwargs.get('polish', True) else 1e-4
    ftol = 5e-3 if not opt_kwargs.get('polish', True) else 1e-6

    assert isinstance(res, EzOptimizeResult)
    if direction == "min":
        assert_allclose(res.x_flat, np.zeros(2), atol=xtol)
        assert_allclose(res.func, 0.0, atol=ftol)
    else:
        assert_allclose(np.abs(res.x_flat), np.array([4.52299366, 4.52299366]), atol=xtol)
        assert_equal(res.x, {'x1': res.x_flat[0], 'x2': res.x_flat[1]})
        assert_allclose(res.func, rastrigin_max_lookup[2], atol=ftol)


@pytest.mark.parametrize("direction", ["min", "max"])
@pytest.mark.parametrize("bounds", [
    [(-5.12, 5.12), (-5.12, 5.12)],
    {'x': [(-5.12, 5.12), (-5.12, 5.12)]}
])
def test_differential_evolution_rastrigin_array(direction, bounds):
    res = differential_evolution(rastrigin, bounds=bounds, direction=direction, maxiter=1000, rng=42, popsize=25)

    assert isinstance(res, EzOptimizeResult)
    if direction == "min":
        assert_allclose(res.x_flat, np.zeros(2), atol=1e-4)
        assert_allclose(res.func, 0.0, atol=1e-8)
    else:
        assert_allclose(np.abs(res.x_flat), np.array([4.52299366, 4.52299366]), atol=1e-4)
        assert_allclose(res.func, rastrigin_max_lookup[2], atol=1e-6)

    if isinstance(bounds, dict):
        assert_equal(res.x, {'x': res.x_flat})


@pytest.mark.parametrize("direction", ["min", "max"])
@pytest.mark.parametrize("dims", [1, 4, 7])
def test_differential_evolution_rastrigin_nd(direction, dims):
    bounds = {f"x{i+1}": (-5.12, 5.12) for i in range(dims)}
    res = differential_evolution(rastrigin_kw_nd, bounds=bounds, direction=direction, maxiter=2000, rng=42, popsize=25)

    assert isinstance(res, EzOptimizeResult)
    if direction == "min":
        assert_allclose(res.x_flat, np.zeros(dims), atol=1e-4)
        assert_allclose(res.func, 0.0, atol=1e-6)
    else:
        assert_allclose(np.abs(res.x_flat), np.array([4.52299366] * dims), atol=1e-4)
        assert_allclose(res.func, rastrigin_max_lookup[dims], atol=1e-6)

    if isinstance(bounds, dict):
        assert_equal(res.x, {f"x{i+1}": res.x_flat[i] for i in range(dims)})


def test_differential_evolution_callback_dict():
    """Test that callback receives dict-mode variables."""
    callback_log = []

    def my_callback(x, convergence=None):
        callback_log.append(x)

    bounds = {'x1': (-5.12, 5.12), 'x2': (-5.12, 5.12)}
    res = differential_evolution(rastrigin_kw_2d, bounds=bounds, callback=my_callback, maxiter=10)

    assert len(callback_log) > 0
    assert isinstance(callback_log[0], dict)
    assert set(callback_log[0].keys()) == {'x1', 'x2'}


def test_differential_evolution_callback_array():
    """Test that callback receives array-mode variables."""
    callback_log = []

    def my_callback(x, convergence=None):
        callback_log.append(x)

    bounds = [(-5.12, 5.12), (-5.12, 5.12)]
    res = differential_evolution(rastrigin, bounds=bounds, callback=my_callback, maxiter=10)

    assert len(callback_log) > 0
    assert isinstance(callback_log[0], np.ndarray)


def test_differential_evolution_kwargs():
    """Test passing extra kwargs to the objective function."""
    def func_with_kwargs(x1, x2, A=10):
        x = [x1, x2]
        return rastrigin(x, A)

    bounds = {'x1': (-5.12, 5.12), 'x2': (-5.12, 5.12)}
    # A=0 makes the function purely quadratic (easier to minimize)
    res = differential_evolution(func_with_kwargs, bounds=bounds, kwargs={'A': 0}, maxiter=500)

    assert isinstance(res, EzOptimizeResult)
    assert_allclose(res.x_flat, np.zeros(2), atol=1e-4)


def test_differential_evolution_rng():
    """Test that rng parameter produces repeatable results."""
    bounds = [(-5.12, 5.12), (-5.12, 5.12)]

    res1 = differential_evolution(rastrigin, bounds=bounds, rng=42, maxiter=100)
    res2 = differential_evolution(rastrigin, bounds=bounds, rng=42, maxiter=100)

    assert_allclose(res1.x_flat, res2.x_flat)
    assert_allclose(res1.func, res2.func)


def test_differential_evolution_disp(capsys):
    """Test that disp=True produces output."""
    bounds = [(-5.12, 5.12), (-5.12, 5.12)]
    differential_evolution(rastrigin, bounds=bounds, disp=True, maxiter=5)

    captured = capsys.readouterr()
    assert "differential_evolution" in captured.out.lower() or len(captured.out) > 0


def test_differential_evolution_workers():
    """Test that workers parameter is accepted (basic smoke test)."""
    bounds = [(-5.12, 5.12), (-5.12, 5.12)]
    res = differential_evolution(
        rastrigin, bounds=bounds, workers=1, updating='deferred', maxiter=100
    )
    assert isinstance(res, EzOptimizeResult)


def test_differential_evolution_integrality():
    """Test integrality constraint."""
    def simple_func(x):
        return (x[0] - 1.5)**2 + (x[1] - 2.7)**2

    bounds = [(0, 5), (0, 5)]
    res = differential_evolution(simple_func, bounds=bounds, integrality=[True, False], rng=42)

    # x[0] should be integer (closest to 1.5 is 2)
    assert_allclose(res.x_flat[0], 2.0, atol=1e-6)
    assert_allclose(res.x_flat[1], 2.7, atol=1e-4)
