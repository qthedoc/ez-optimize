from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from scipy.optimize import direct as scipy_direct

from ez_optimize.optimization_problem import OptimizationProblem
from ez_optimize.utilities import EzOptimizeResult

def direct(
    func: Callable,
    bounds: Union[List[Tuple[float, float]], Dict[str, Tuple[float, float]], Dict[str, List[Tuple[float, float]]]],
    direction: Literal["min", "max"] = "min",
    kwargs: Optional[Dict[str, Any]] = None,
    *,
    args: Optional[Tuple] = None,
    eps: float = 1e-4,
    maxfun: Optional[int] = None,
    maxiter: int = 1000,
    locally_biased: bool = True,
    f_min: float = -np.inf,
    f_min_rtol: float = 1e-4,
    vol_tol: float = 1e-16,
    len_tol: float = 1e-6,
    callback: Optional[Callable] = None,
    x0: Optional[Union[np.ndarray, Dict[str, Any]]] = None,
    var_mode: Optional[Literal["array", "dict"]] = None,
):
    """Find the global minimum of a function using the DIRECT algorithm.

    This is a drop-in scipy.optimize.direct() wrapper with enhanced features:
    - keyword variables (e.g. bounds={'x': (-5, 5), 'y': (-5, 5)} instead of bounds=[(-5, 5), (-5, 5)])
    - direction (min or max)

    Parameters
    ----------
    func : callable
        The objective function to be minimized.

        For array mode::
            func(x, *args, **kwargs) -> float
        where ``x`` is a 1-D numpy array.

        For dict mode::
            func(**vars, **kwargs) -> float
        where ``vars`` is a dict of variable names to values.

        Additional positional and keyword arguments can be passed via `args`
        and `kwargs` parameters of this function.
    bounds : sequence or dict
        Bounds for variables. Three ways to specify the bounds:

        1. Sequence of ``(min, max)`` pairs for each element in `x` (array mode).
        2. Dict mapping variable names to ``(min, max)`` tuples for scalar
           variables (dict mode).
        3. Dict mapping variable names to lists of ``(min, max)`` tuples for
           array-valued variables (dict mode).

        In dict mode, bounds also define the variable names and shapes when
        ``x0`` is not provided.
    direction : {'min', 'max'}, optional
        Direction of optimization. Default is 'min'. If 'max', the function
        will be maximized instead of minimized by internally negating the
        objective function.
    kwargs : dict, optional
        Additional keyword arguments to pass to the objective function.
        In 'dict' mode, if keys conflict with x0 keys, x0 values take precedence
        and a warning is issued.
    args : tuple, optional
        Extra positional arguments passed to the objective function.
        Not allowed in 'dict' mode (will raise ValueError).
    eps : float, optional
        Minimal required difference of the objective function values between
        the current best hyperrectangle and the next potentially optimal
        hyperrectangle to be divided. Serves as a tradeoff between local and
        global search. Default is 1e-4.
    maxfun : int or None, optional
        Approximate upper bound on objective function evaluations. If None,
        will be automatically set to ``1000 * N`` where ``N`` represents the
        number of dimensions. Default is None.
    maxiter : int, optional
        Maximum number of iterations. Default is 1000.
    locally_biased : bool, optional
        If True (default), use the locally biased variant of the algorithm
        known as DIRECT_L. If False, use the original unbiased DIRECT
        algorithm.
    f_min : float, optional
        Function value of the global optimum. Set this value only if the
        global optimum is known. Default is ``-np.inf``.
    f_min_rtol : float, optional
        Terminate the optimization once the relative error between the current
        best minimum and the supplied global minimum ``f_min`` is smaller than
        ``f_min_rtol``. Default is 1e-4.
    vol_tol : float, optional
        Terminate the optimization once the volume of the hyperrectangle
        containing the lowest function value is smaller than ``vol_tol`` of
        the complete search space. Default is 1e-16.
    len_tol : float, optional
        Terminate the optimization once half of the normalized maximal side
        length (or diagonal if ``locally_biased=False``) of the
        hyperrectangle containing the lowest function value is smaller than
        ``len_tol``. Default is 1e-6.
    callback : callable, optional
        A callback function with signature ``callback(xk)`` where ``xk``
        represents the best function value found so far.

        In array mode, ``xk`` is a numpy array. In dict mode, ``xk`` is a dict
        mapping variable names to values.
    x0 : array_like or dict, optional
        Not used by DIRECT (variable structure is inferred from ``bounds``),
        but accepted for API consistency.
    var_mode : {'array', 'dict'}, optional
        Expected format of variables for ``x0``, ``func``, and ``bounds``.
        If None, inferred from the type of ``x0`` or ``bounds``. Use 'dict'
        to work with named variables as dicts instead of numpy arrays.

    Returns
    -------
    res : OptimizeResult
        The optimization result represented as a ``OptimizeResult`` object.
        Important attributes are: ``x`` the solution array, ``success`` a
        Boolean flag indicating if the optimizer exited successfully, and
        ``message`` which describes the cause of the termination.
        See ``OptimizeResult`` for a description of other attributes.

    Notes
    -----
    DIviding RECTangles (DIRECT) is a deterministic global optimization
    algorithm capable of minimizing a black box function with its variables
    subject to lower and upper bound constraints by sampling potential
    solutions in the search space.
    """

    # Wrapper handles relevant arguments to be transformed
    problem = OptimizationProblem(
        func=func,
        x0=x0,
        direction=direction,
        var_mode=var_mode,
        bounds=bounds,
        args=args,
        kwargs=kwargs,
        callback=callback,
    )

    # Run SciPy
    scipy_result = scipy_direct(
        # Pass wrapped args
        func=problem.scipy.direct.func(),
        bounds=problem.scipy.direct.bounds(),
        callback=problem.scipy.direct.callback(),

        # Pass args that aren't wrapped
        eps=eps,
        maxfun=maxfun,
        maxiter=maxiter,
        locally_biased=locally_biased,
        f_min=f_min,
        f_min_rtol=f_min_rtol,
        vol_tol=vol_tol,
        len_tol=len_tol,
    )

    return problem.scipy.direct.interpret_result(scipy_result)
