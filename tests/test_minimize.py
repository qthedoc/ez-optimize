import pytest
from functools import partial

import numpy as np

from numpy.testing import assert_allclose

from ez_optimize.minimize import minimize
from ez_optimize.utilities import EzOptimizeResult

no_grad_methods = ["nelder-mead", "powell", "CG", "BFGS", "L-BFGS-B"]


def rosen(x, a, b) -> float:
    """The Rosenbrock function"""
    return sum(a * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0) + b


def rosen_2d_kw(x, y, a, b) -> float:
    """The 2D Rosenbrock function in keyword mode"""
    return rosen(np.array([x, y]), a, b)


@pytest.mark.parametrize("method", no_grad_methods, ids=no_grad_methods)
def test_array_5d(method: str):
    x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])

    res = minimize(rosen, x0, method=method, args=(100, 0), tol=1e-8)

    assert isinstance(res, EzOptimizeResult)
    assert_allclose(res.x, np.ones(5), atol=1e-4, rtol=1e-4)
    assert_allclose(res.fun, 0.0, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("method", no_grad_methods, ids=no_grad_methods)
def test_kw_2d(method: str):
    x0 = {'x': 1.3, 'y': 0.7}

    res = minimize(partial(rosen_2d_kw, a=100, b=0), x0, method=method, tol=1e-8)

    assert isinstance(res, EzOptimizeResult)
    assert_allclose(res.x, np.ones(2), atol=1e-4, rtol=1e-4)
    assert_allclose(res.fun, 0.0, atol=1e-8, rtol=1e-8)
    assert isinstance(res.x_original, dict)
    assert_allclose(list(res.x_original.values()), np.ones(2), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("method", no_grad_methods, ids=no_grad_methods)
def test_kw_array(method: str):
    x0 = {'x':np.array([1.3, 0.7, 0.8, 1.9, 1.2])}

    res = minimize(partial(rosen, a=100, b=0), x0, method=method, tol=1e-8)

    assert isinstance(res, EzOptimizeResult)
    assert_allclose(res.x, np.ones(5), atol=1e-4, rtol=1e-4)
    assert_allclose(res.fun, 0.0, atol=1e-8, rtol=1e-8)
    assert isinstance(res.x_original, dict)
    assert_allclose(res.x_original['x'], np.ones(5), atol=1e-4, rtol=1e-4)


def test_array_direction_max():
    def f(x):
        return - (x - 1)**2

    res = minimize(f, np.array([0.]), method='SLSQP', direction='max', bounds=[(0, 2)], tol=1e-8)

    assert isinstance(res, EzOptimizeResult)
    assert_allclose(res.x, 1.0, atol=1e-4)
    assert_allclose(res.fun, 0.0, atol=1e-8)


def test_kw_direction_max():
    def f_kw(x):
        return - (x - 1)**2

    res = minimize(f_kw, {'x': 0.}, method='SLSQP', direction='max', tol=1e-8)

    assert isinstance(res, EzOptimizeResult)
    assert_allclose(res.x, np.ones(1), atol=1e-4, rtol=1e-4)
    assert_allclose(res.fun, 0.0, atol=1e-8, rtol=1e-8)
    assert isinstance(res.x_original, dict)
    assert_allclose(res.x_original['x'], np.ones(1), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("method", ["SLSQP", "L-BFGS-B"], ids=["SLSQP", "L-BFGS-B"])
def test_array_with_bounds(method: str):
    x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
    bounds = [(0, 2), (0, 2), (0, 2), (0, 2), (0, 2)]

    res = minimize(rosen, x0, method=method, args=(100, 0), bounds=bounds, tol=1e-8)

    assert isinstance(res, EzOptimizeResult)
    assert_allclose(res.x, np.ones(5), atol=1e-4, rtol=1e-4)
    assert_allclose(res.fun, 0.0, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("method", ["SLSQP", "L-BFGS-B"], ids=["SLSQP", "L-BFGS-B"])
def test_kw_with_bounds(method: str):
    x0 = {'x': 1.3, 'y': 0.7}
    bounds = {'x': (0, 2), 'y': (0, 2)}

    res = minimize(partial(rosen_2d_kw, a=100, b=0), x0, method=method, bounds=bounds, tol=1e-8)

    assert isinstance(res, EzOptimizeResult)
    assert_allclose(res.x, np.ones(2), atol=1e-2, rtol=1e-2)
    assert_allclose(res.fun, 0.0, atol=1e-5, rtol=1e-5)
    assert isinstance(res.x_original, dict)
    assert_allclose(list(res.x_original.values()), np.ones(2), atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("method", ["SLSQP", "L-BFGS-B"], ids=["SLSQP", "L-BFGS-B"])
def test_kw_array_with_bounds(method: str):
    x0 = {'x':np.array([1.3, 0.7, 0.8, 1.9, 1.2])}
    bounds = {'x': [(0, 2), (0, 2), (0, 2), (0, 2), (0, 2)]}

    res = minimize(partial(rosen, a=100, b=0), x0, method=method, bounds=bounds, tol=1e-8)

    assert isinstance(res, EzOptimizeResult)
    assert_allclose(res.x, np.ones(5), atol=1e-4, rtol=1e-4)
    assert_allclose(res.fun, 0.0, atol=1e-8, rtol=1e-8)
    assert isinstance(res.x_original, dict)
    assert_allclose(res.x_original['x'], np.ones(5), atol=1e-4, rtol=1e-4)
