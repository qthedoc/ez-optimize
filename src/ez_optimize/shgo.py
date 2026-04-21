from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from scipy.optimize import shgo as scipy_shgo

from ez_optimize.optimization_problem import OptimizationProblem
from ez_optimize.utilities import EzOptimizeResult

def shgo(
    func: Callable,
    bounds: Union[List[Tuple[float, float]], Dict[str, Tuple[float, float]], Dict[str, List[Tuple[float, float]]]],
    direction: Literal["min", "max"] = "min",
    args: Optional[Tuple] = None,
    kwargs: Optional[Dict[str, Any]] = None,
    constraints=None,
    n: int = 100,
    iters: int = 1,
    callback: Optional[Callable] = None,
    minimizer_kwargs=None,
    options: Optional[Dict[str, Any]] = None,
    sampling_method: str = "simplicial",
    *,
    workers: int = 1,
    x0: Optional[Union[np.ndarray, Dict[str, Any]]] = None,
    var_mode: Optional[Literal["array", "dict"]] = None,
):
    """Find the global minimum of a function using SHG optimization.

    This is a drop-in scipy.optimize.shgo() wrapper with enhanced features:
    - keyword variables (e.g. bounds={'x': (-5, 5), 'y': (-5, 5)} instead of bounds=[(-5, 5), (-5, 5)])
    - direction (min or max)

    SHGO stands for "simplicial homology global optimization".

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
    args : tuple, optional
        Extra positional arguments passed to the objective function.
        Not allowed in 'dict' mode (will raise ValueError).
    kwargs : dict, optional
        Additional keyword arguments to pass to the objective function.
        In 'dict' mode, if keys conflict with x0 keys, x0 values take precedence
        and a warning is issued.
    constraints : {Constraint, dict} or List of {Constraint, dict}, optional
        Constraints definition. Only for COBYLA, COBYQA, SLSQP and trust-constr
        local minimize methods.
    n : int, optional
        Number of sampling points used in the construction of the simplicial
        complex. Default is 100.
    iters : int, optional
        Number of iterations used in the construction of the simplicial
        complex. Default is 1.
    callback : callable, optional
        Called after each iteration, as ``callback(xk)``, where ``xk`` is the
        current parameter vector.

        In array mode, ``xk`` is a numpy array. In dict mode, ``xk`` is a dict
        mapping variable names to values.
    minimizer_kwargs : dict, optional
        Extra keyword arguments to be passed to the local minimizer
        ``scipy.optimize.minimize``. Some important options could be:

        method : str
            The minimization method. If not given, chosen to be one of BFGS,
            L-BFGS-B, SLSQP, depending on whether or not the problem has
            constraints or bounds.
    options : dict, optional
        A dictionary of solver options. Many of the options specified for the
        global routine are also passed to the ``scipy.optimize.minimize``
        routine.
    sampling_method : str or function, optional
        Current built in sampling method options are ``halton``, ``sobol`` and
        ``simplicial``. The default ``simplicial`` provides the theoretical
        guarantee of convergence to the global minimum in finite time.
    workers : int or map-like callable, optional
        Sample and run the local serial minimizations in parallel. Supply -1
        to use all available CPU cores. Default is 1.
    x0 : array_like or dict, optional
        Initial guess. If not provided, the variable structure is inferred
        from ``bounds``.
    var_mode : {'array', 'dict'}, optional
        Expected format of variables for ``x0``, ``func``, and ``bounds``.
        If None, inferred from the type of ``x0`` or ``bounds``. Use 'dict'
        to work with named variables as dicts instead of numpy arrays.

    Returns
    -------
    res : OptimizeResult
        The optimization result represented as a ``OptimizeResult`` object.
        Important attributes are: ``x`` the solution array corresponding to
        the global minimum, ``fun`` the function output at the global solution,
        ``xl`` an ordered list of local minima solutions, ``funl`` the function
        output at the corresponding local solutions, ``success`` a Boolean flag
        indicating if the optimizer exited successfully, and ``message`` which
        describes the cause of the termination.
        See ``OptimizeResult`` for a description of other attributes.

    Notes
    -----
    Global optimization using simplicial homology global optimization.
    Appropriate for solving general purpose NLP and blackbox optimization
    problems to global optimality (low-dimensional problems).
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

    # Warnings
    if problem.var_mode == "dict" and constraints is not None:
        warnings.warn("Passing 'constraints' in dict mode is not yet fully supported by ez_optimize.shgo. constraints will be passed directly to SciPy's shgo.", UserWarning)

    if problem.var_mode == "dict" and minimizer_kwargs is not None:
        warnings.warn("Passing 'minimizer_kwargs' in dict mode is not yet fully supported by ez_optimize.shgo. minimizer_kwargs will be passed directly to SciPy's minimize.", UserWarning)

    # Run SciPy
    scipy_result = scipy_shgo(
        # Pass wrapped args
        func=problem.scipy.shgo.func(),
        bounds=problem.scipy.shgo.bounds(),
        callback=problem.scipy.shgo.callback(),

        # Pass args that aren't wrapped
        constraints=constraints,
        n=n,
        iters=iters,
        minimizer_kwargs=minimizer_kwargs,
        options=options,
        sampling_method=sampling_method,
        workers=workers,
    )

    return problem.scipy.shgo.interpret_result(scipy_result)
