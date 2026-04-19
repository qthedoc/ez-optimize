from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from scipy.optimize import differential_evolution as scipy_differential_evolution

from ez_optimize.optimization_problem import OptimizationProblem
from ez_optimize.utilities import EzOptimizeResult

def differential_evolution(
    func: Callable,
    bounds: Union[List[Tuple[float, float]], Dict[str, Tuple[float, float]], Dict[str, List[Tuple[float, float]]]],
    direction: Literal["min", "max"] = "min",
    args: Optional[Tuple] = None,
    kwargs: Optional[Dict[str, Any]] = None,
    strategy: str = "best1bin",
    maxiter: int = 1000,
    popsize: int = 15,
    tol: float = 0.01,
    mutation: Union[float, Tuple[float, float]] = (0.5, 1),
    recombination: float = 0.7,
    rng=None,
    callback: Optional[Callable] = None,
    disp: bool = False,
    polish: bool = True,
    init: str = "latinhypercube",
    atol: float = 0,
    updating: str = "immediate",
    workers: int = 1,
    constraints=(),
    x0: Optional[Union[np.ndarray, Dict[str, Any]]] = None,
    integrality=None,
    vectorized: bool = False,
    var_mode: Optional[Literal["array", "dict"]] = None,
):
    """Find the global minimum of a multivariate function using Differential Evolution.

    This is a drop-in scipy.optimize.differential_evolution() wrapper with enhanced features:
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

        The total number of bounds is used to determine the number of
        parameters, N.
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
    strategy : str, optional
        The differential evolution strategy to use. Should be one of:

        - 'best1bin'
        - 'best1exp'
        - 'rand1bin'
        - 'rand1exp'
        - 'rand2bin'
        - 'rand2exp'
        - 'randtobest1bin'
        - 'randtobest1exp'
        - 'currenttobest1bin'
        - 'currenttobest1exp'
        - 'best2exp'
        - 'best2bin'

        The default is 'best1bin'.
    maxiter : int, optional
        The maximum number of generations over which the entire population
        is evolved. Default value is 1000.
    popsize : int, optional
        A multiplier for setting the total population size. The population
        has ``popsize * N`` individuals. Default is 15.
    tol : float, optional
        Relative tolerance for convergence. Default is 0.01.
    mutation : float or tuple(float, float), optional
        The mutation constant. If specified as a float it should be in the
        range [0, 2). If specified as a tuple ``(min, max)`` dithering is
        employed. Default is (0.5, 1).
    recombination : float, optional
        The recombination constant, should be in the range [0, 1]. Default
        is 0.7.
    rng : `numpy.random.Generator`, optional
        Pseudorandom number generator state. When `rng` is None, a new
        `numpy.random.Generator` is created using entropy from the
        operating system. Types other than `numpy.random.Generator` are
        passed to `numpy.random.default_rng` to instantiate a `Generator`.

        Specify `rng` for repeatable minimizations.
    callback : callable, optional
        A callable called after each iteration. Has the signature:

            ``callback(intermediate_result: OptimizeResult)``

        where ``intermediate_result`` is a keyword parameter containing an
        ``OptimizeResult`` with attributes ``x`` and ``fun``, the best
        solution found so far and the objective function.

        In dict mode, ``x`` in the intermediate result is a dict mapping
        variable names to values.

        The callback also supports a signature like:

            ``callback(x, convergence: float=val)``

        where ``val`` represents the fractional value of the population
        convergence.

        In array mode, ``x`` is a numpy array. In dict mode, ``x`` is a dict
        mapping variable names to values.

        If the callback raises ``StopIteration`` or returns ``True``,
        global minimization will halt.
    disp : bool, optional
        Prints the evaluated func at every iteration. Default is False.
    polish : bool or callable, optional
        If True (default), then ``scipy.optimize.minimize`` with the
        L-BFGS-B method is used to polish the best population member at
        the end. If a constrained problem is being studied then the
        trust-constr method is used instead.
    init : str or array-like, optional
        Specify which type of population initialization is performed.
        Should be one of:

        - 'latinhypercube'
        - 'sobol'
        - 'halton'
        - 'random'
        - array specifying the initial population with shape ``(S, N)``.

        Default is 'latinhypercube'.
    atol : float, optional
        Absolute tolerance for convergence. Default is 0.
    updating : {'immediate', 'deferred'}, optional
        If ``'immediate'``, the best solution vector is continuously updated
        within a single generation. With ``'deferred'``, the best solution
        vector is updated once per generation. Only ``'deferred'`` is
        compatible with parallelization or vectorization. Default is
        ``'immediate'``.
    workers : int or map-like callable, optional
        If workers is an int the population is subdivided into workers
        sections and evaluated in parallel. Supply -1 to use all available
        CPU cores. Default is 1.
    constraints : {NonLinearConstraint, LinearConstraint, Bounds}, optional
        Constraints on the solver, over and above those applied by the
        bounds keyword.
    x0 : array_like or dict, optional
        Provides an initial guess to the minimization. Once the population
        has been initialized this vector replaces the first (best) member.
        If not provided, the variable structure is inferred from ``bounds``.
    integrality : 1-D array, optional
        For each decision variable, a boolean value indicating whether the
        decision variable is constrained to integer values.
    vectorized : bool, optional
        If True, ``func`` is sent an x array with ``x.shape == (N, S)``,
        and is expected to return an array of shape ``(S,)``. Default is
        False.
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
    Differential evolution is a stochastic population based method that is
    useful for global optimization problems. At each pass through the
    population the algorithm mutates each candidate solution by mixing with
    other candidate solutions to create a trial candidate. The algorithm
    does not use gradient methods to find the minimum, and can search large
    areas of candidate space, but often requires larger numbers of function
    evaluations than conventional gradient-based techniques.
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
    if problem.var_mode == "dict" and constraints != ():
        warnings.warn("Passing 'constraints' in dict mode is not yet fully supported by ez_optimize.differential_evolution. constraints will be passed directly to SciPy's differential_evolution.", UserWarning)

    if problem.var_mode == "dict" and vectorized:
        warnings.warn("Passing 'vectorized=True' in dict mode is not yet fully supported by ez_optimize.differential_evolution. func will receive flat arrays with shape (N, S) instead of dicts.", UserWarning)

    if problem.var_mode == "dict" and callable(strategy):
        warnings.warn("Passing a callable 'strategy' in dict mode is not yet fully supported by ez_optimize.differential_evolution. strategy will receive the raw population array instead of dicts.", UserWarning)

    if problem.var_mode == "dict" and not isinstance(init, str):
        warnings.warn("Passing a custom 'init' (initial population) in dict mode is not yet fully supported by ez_optimize.differential_evolution. init must use the internal flat variable ordering.", UserWarning)

    if problem.var_mode == "dict" and integrality is not None:
        warnings.warn("Passing 'integrality' in dict mode is not yet fully supported by ez_optimize.differential_evolution. integrality must use the internal flat variable ordering.", UserWarning)

    # Run SciPy
    scipy_result = scipy_differential_evolution(
        # Pass wrapped args
        func=problem.scipy.get_func(),
        bounds=problem.scipy.get_bounds(),
        callback=problem.scipy.get_callback(),
        x0=problem.scipy.get_x0(),

        # Pass args that aren't wrapped
        strategy=strategy,
        maxiter=maxiter,
        popsize=popsize,
        tol=tol,
        mutation=mutation,
        recombination=recombination,
        rng=rng,
        disp=disp,
        polish=polish,
        init=init,
        atol=atol,
        updating=updating,
        workers=workers,
        constraints=constraints,
        integrality=integrality,
        vectorized=vectorized,
    )

    return problem.scipy.interpret_result(scipy_result)
