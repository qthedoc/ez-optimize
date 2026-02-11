"""Test args and kwargs functionality"""
import pytest
import warnings
import numpy as np
from ez_optimize import minimize


def test_array_mode_with_args():
    """Test array mode with positional args"""
    def objective_with_args(x, factor):
        """Objective: (x[0] - 2)^2 + (x[1] - 3)^2, scaled by factor"""
        return factor * ((x[0] - 2)**2 + (x[1] - 3)**2)

    result = minimize(
        fun=objective_with_args,
        x0=np.array([0.0, 0.0]),
        method='BFGS',
        args=(2.0,)  # factor = 2.0
    )
    
    assert result.success
    np.testing.assert_allclose(result.x, [2.0, 3.0], atol=1e-5)
    assert result.fun < 1e-8


def test_array_mode_with_kwargs():
    """Test array mode with keyword arguments"""
    def objective_with_kwargs(x, scale=1.0, offset=0.0):
        """Objective with keyword arguments"""
        return scale * ((x[0] - 2)**2 + (x[1] - 3)**2) + offset

    result = minimize(
        fun=objective_with_kwargs,
        x0=np.array([0.0, 0.0]),
        method='BFGS',
        kwargs={'scale': 1.5, 'offset': 10.0}
    )
    
    assert result.success
    np.testing.assert_allclose(result.x, [2.0, 3.0], atol=1e-5)
    np.testing.assert_allclose(result.fun, 10.0, atol=1e-8)


def test_dict_mode_with_kwargs_no_conflict():
    """Test dict mode with kwargs that don't conflict with x0"""
    def objective_dict_no_conflict(a, b, multiplier=1.0):
        """Objective with dict params and extra kwarg"""
        return multiplier * ((a - 2)**2 + (b - 3)**2)

    result = minimize(
        fun=objective_dict_no_conflict,
        x0={'a': 0.0, 'b': 0.0},
        method='BFGS',
        x_mode='dict',
        kwargs={'multiplier': 2.5}
    )
    
    assert result.success
    assert isinstance(result.x, dict)
    np.testing.assert_allclose(result.x['a'], 2.0, atol=1e-5)
    np.testing.assert_allclose(result.x['b'], 3.0, atol=1e-5)
    assert result.fun < 1e-8


def test_dict_mode_with_conflicting_kwargs_warns():
    """Test dict mode with conflicting kwargs issues warning and x0 takes precedence"""
    def objective_dict_conflict(a, b):
        """Objective with dict params"""
        return (a - 5)**2 + (b - 7)**2

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = minimize(
            fun=objective_dict_conflict,
            x0={'a': 0.0, 'b': 0.0},
            method='BFGS',
            x_mode='dict',
            kwargs={'a': 100.0}  # 'a' conflicts with x0
        )
        
        # Check that a warning was issued
        assert len(w) >= 1
        assert "Conflicting parameter names" in str(w[0].message)
    
    assert result.success
    assert isinstance(result.x, dict)
    # x0 values should take precedence, so we should converge to a=5, b=7
    np.testing.assert_allclose(result.x['a'], 5.0, atol=1e-5)
    np.testing.assert_allclose(result.x['b'], 7.0, atol=1e-5)
    assert result.fun < 1e-8


def test_dict_mode_with_args_raises_error():
    """Test that dict mode with args raises ValueError"""
    def objective_dict(a, b, multiplier=1.0):
        """Objective with dict params"""
        return multiplier * ((a - 2)**2 + (b - 3)**2)

    with pytest.raises(ValueError, match="Positional args are not allowed in 'dict' mode"):
        minimize(
            fun=objective_dict,
            x0={'a': 0.0, 'b': 0.0},
            method='BFGS',
            x_mode='dict',
            args=(2.0,)  # This should fail
        )


def test_array_mode_with_both_args_and_kwargs():
    """Test array mode with both positional and keyword arguments"""
    def objective_both(x, factor, offset=0.0):
        """Objective with both positional and keyword args"""
        return factor * ((x[0] - 2)**2 + (x[1] - 3)**2) + offset

    result = minimize(
        fun=objective_both,
        x0=np.array([0.0, 0.0]),
        method='BFGS',
        args=(3.0,),
        kwargs={'offset': 5.0}
    )
    
    assert result.success
    np.testing.assert_allclose(result.x, [2.0, 3.0], atol=1e-5)
    np.testing.assert_allclose(result.fun, 5.0, atol=1e-8)
