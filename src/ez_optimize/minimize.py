from __future__ import annotations

from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np

from ez_optimize.optimization_problem import OptimizationProblem
from ez_optimize.utilities import EzOptimizeResult
from ez_optimize.constants import MinimizeMethod

def minimize(
    fun: Callable,
    x0: Union[np.ndarray, Dict[str, Any]],
    method: Optional[MinimizeMethod] = None,
    direction: Literal["min", "max"] = "min",
    bounds: Optional[Union[List[Tuple[float, float]], Dict[str, Tuple[float, float]], Dict[str, List[Tuple[float, float]]]]] = None,
    x_mode: Optional[Literal["array", "dict"]] = None,
    args: Optional[Tuple] = None,
    kwargs: Optional[Dict[str, Any]] = None,
    **optimizer_kwargs,  # bounds, constraints, tol, options, etc. stored for later use
) -> EzOptimizeResult:
    """
    Minimize (or maximize) a scalar function of one or more variables.

    This function provides a high-level interface to optimization algorithms,
    supporting both array and dict-based parameter modes, with automatic
    flattening and restoration of structures.

    Parameters
    ----------
    fun : callable
        The objective function to be minimized (or maximized).
        For array mode: fun(x, *args, **kwargs) where x is a numpy array.
        For dict mode: fun(**params, **kwargs) where params is a dict of parameters.
    x0 : array_like or dict
        Initial guess. Array of real elements of size (n,),
        or dict with parameter names as keys.
    method : str, optional
        Type of solver. Should be one of the methods supported by SciPy.
    direction : {'min', 'max'}, optional
        Direction of optimization. Default is 'min'.
    x_mode : {'array', 'dict'}, optional
        Mode for parameter handling. If None, inferred from x0.
    bounds : sequence or dict, optional
        Bounds on variables. For array mode: list of (min, max) pairs.
        For dict mode: dict with same keys as x0, values as (min, max) or list of pairs.
    args : tuple, optional
        Additional positional arguments to pass to the objective function.
        Not allowed in 'dict' mode (will raise ValueError).
        In 'array' mode, these are passed after the x array: fun(x, *args).
    kwargs : dict, optional
        Additional keyword arguments to pass to the objective function.
        In 'dict' mode, if keys conflict with x0 keys, x0 values take precedence
        and a warning is issued.
    **optimizer_kwargs
        Additional keyword arguments passed to the optimizer.

    Returns
    -------
    EzOptimizeResult
        The optimization result represented as an EzOptimizeResult object.
        Important attributes are: ``x`` the solution array/dict, ``success`` a
        Boolean flag indicating if the optimizer exited successfully and
        ``message`` which describes the cause of the termination. See
        EzOptimizeResult for a description of other attributes.

    Notes
    -----
    This function uses the OptimizationProblem class internally to handle
    parameter flattening, bounds, and direction, then delegates to SciPy's
    minimize function.
    """
    problem = OptimizationProblem(
        fun=fun,
        x0=x0,
        method=method,
        direction=direction,
        x_mode=x_mode,
        bounds=bounds,
        args=args,
        kwargs=kwargs,
        **optimizer_kwargs
    )
    return problem.optimize()
