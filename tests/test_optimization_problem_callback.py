import pytest
import numpy as np
from unittest.mock import Mock

from ez_optimize.constants import MINIMIZE_METHODS_NO_GRAD, MINIMIZE_METHODS_OPTIONAL_GRAD
from ez_optimize.optimization_problem import OptimizationProblem

# Test variations:
# - use both array and dict modes
# - use all callback formats
# - use all the methods minimize methods
# - test direction='max'

def quadratic(x_array):
    """n-dimensional quadratic function: sum((x_i - 1)^2)"""
    return np.sum((x_array - 1) ** 2)


def quadratic_2d(x, y):
    """Quadratic in dict mode"""
    return (x - 1) ** 2 + (y - 1) ** 2

# Supported methods for: callback(xk)
xk_supported_methods = MINIMIZE_METHODS_NO_GRAD + MINIMIZE_METHODS_OPTIONAL_GRAD

# Supported methods for: callback(intermediate_result)
intermediate_supported_methods = [m for m in MINIMIZE_METHODS_NO_GRAD + MINIMIZE_METHODS_OPTIONAL_GRAD if m != 'TNC']

## Supported methods for: callback(xk, intermediate_result)
xk_intermediate_supported_methods = ['trust-constr']


@pytest.mark.parametrize("method", xk_supported_methods)
@pytest.mark.parametrize("mode", ['array', 'dict'])
def test_callback_xk_supported_methods(method, mode):
    """Test xk callback for methods that support it"""
    callback_mock = Mock()
    fun = quadratic if mode == 'array' else quadratic_2d
    x0 = np.array([0.0, 0.0]) if mode == 'array' else {'x': 0.0, 'y': 0.0}

    prob = OptimizationProblem(fun, x0, method=method, callback=callback_mock, tol=1e-8)
    res = prob.optimize()

    assert callback_mock.call_count > 0
    for call in callback_mock.call_args_list:
        x_arg = call[0][0]
        if mode == 'array':
            assert isinstance(x_arg, np.ndarray)
            assert x_arg.shape == (2,)
        else:
            assert isinstance(x_arg, dict)
            assert 'x' in x_arg
            assert 'y' in x_arg

@pytest.mark.parametrize("method", intermediate_supported_methods)
@pytest.mark.parametrize("mode", ['array', 'dict'])
def test_callback_intermediate_result_supported_methods(method, mode):
    """Test intermediate_result callback for methods that support it"""
    callback_mock = Mock()

    def callback(intermediate_result):
        callback_mock(intermediate_result)

    fun = quadratic if mode == 'array' else quadratic_2d
    x0 = np.array([0.0, 0.0]) if mode == 'array' else {'x': 0.0, 'y': 0.0}
    prob = OptimizationProblem(fun, x0, method=method, callback=callback, tol=1e-8)
    res = prob.optimize()

    assert callback_mock.call_count > 0
    for call in callback_mock.call_args_list:
        res_arg = call[0][0]
        assert hasattr(res_arg, 'x')
        assert hasattr(res_arg, 'fun')
        if mode == 'array':
            assert isinstance(res_arg.x, np.ndarray)
            assert res_arg.x.shape == (2,)
        else:
            assert isinstance(res_arg.x, dict)
            assert 'x' in res_arg.x
            assert 'y' in res_arg.x
        assert isinstance(res_arg.fun, float)

@pytest.mark.parametrize("method", xk_intermediate_supported_methods)
@pytest.mark.parametrize("mode", ['array', 'dict'])
def test_callback_xk_intermediate_result_supported_methods(method, mode):
    """Test xk and intermediate_result callback for methods that support it"""
    callback_mock = Mock()

    def callback(xk, intermediate_result=None):
        callback_mock(xk, intermediate_result)

    fun = quadratic if mode == 'array' else quadratic_2d
    x0 = np.array([0.0, 0.0]) if mode == 'array' else {'x': 0.0, 'y': 0.0}
    prob = OptimizationProblem(fun, x0, method=method, callback=callback, tol=1e-8)
    res = prob.optimize()

    assert callback_mock.call_count > 0
    for call in callback_mock.call_args_list:
        x_arg = call[0][0]
        if mode == 'array':
            assert isinstance(x_arg, np.ndarray)
            assert x_arg.shape == (2,)
        else:
            assert isinstance(x_arg, dict)
            assert 'x' in x_arg
            assert 'y' in x_arg
            
        res_arg = call[0][1]
        assert hasattr(res_arg, 'x')
        assert hasattr(res_arg, 'fun')
        if mode == 'array':
            assert isinstance(res_arg.x, np.ndarray)
            assert res_arg.x.shape == (2,)
        else:
            assert isinstance(res_arg.x, dict)
            assert 'x' in res_arg.x
            assert 'y' in res_arg.x
        assert isinstance(res_arg.fun, float)

# def test_callback_xk_trust_constr_not_supported():
#     """Test that xk callback is not called for trust-constr (does not support xk)"""
#     callback_mock = Mock()
#     x0 = np.array([0.0])

#     prob = OptimizationProblem(quadratic, x0, method='trust-constr', callback=callback_mock, tol=1e-8)
#     res = prob.optimize()

#     # trust-constr does not support xk callback, so it should not be called
#     assert callback_mock.call_count == 0


# def test_callback_intermediate_result_tnc_not_supported():
#     """Test that intermediate_result callback is not called for TNC (does not support intermediate_result)"""
#     callback_mock = Mock()

#     def callback(intermediate_result):
#         callback_mock(intermediate_result)

#     x0 = np.array([0.0])
#     prob = OptimizationProblem(quadratic, x0, method='TNC', callback=callback, tol=1e-8)
#     res = prob.optimize()

#     # TNC does not support intermediate_result callback, so it should not be called
#     assert callback_mock.call_count == 0


# def test_callback_intermediate_result_max_direction():
#     """Test that fun in intermediate_result is un-negated for max direction"""
#     callback_mock = Mock()

#     def callback(intermediate_result):
#         callback_mock(intermediate_result.fun)

#     x0 = np.array([0.0])
#     prob = OptimizationProblem(quadratic, x0, method='BFGS', direction='max', callback=callback, tol=1e-8)
#     res = prob.optimize()

#     # For max direction, fun should be un-negated in callback
#     assert callback_mock.call_count > 0
#     for call in callback_mock.call_args_list:
#         fun_val = call[0][0]
#         assert fun_val >= 0  # quadratic is always >= 0