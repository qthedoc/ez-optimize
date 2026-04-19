from __future__ import annotations

from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from scipy.optimize import dual_annealing as scipy_dual_annealing

from ez_optimize.optimization_problem import OptimizationProblem
from ez_optimize.utilities import EzOptimizeResult

def dual_annealing(
    func: Callable, 
    bounds: Optional[Union[List[Tuple[float, float]], Dict[str, Tuple[float, float]], Dict[str, List[Tuple[float, float]]]]] = None,
    direction: Literal["min", "max"] = "min",
    args: Optional[Tuple] = None, 
    kwargs: Optional[Dict[str, Any]] = None,
    maxiter: int = 1000,         
    minimizer_kwargs=None, 
    initial_temp: float = 5230.,
    restart_temp_ratio: float = 2.e-5, 
    visit: float = 2.62, 
    accept: float = -5.0, 
    maxfun: float = 1e7, 
    rng=None, 
    no_local_search: bool = False,
    callback: Optional[Callable] = None, 
    x0: Optional[Union[np.ndarray, Dict[str, Any]]] = None,
    var_mode=None,
):
    """Find the global minimum of a function using Dual Annealing.
    
    This is a drop-in scipy.optimize.dual_annealing() wrapper with enhanced features:
    - keyword variables (e.g. x0={'x': 1, 'y': 2} instead of x0=[1, 2])
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
        Direction of optimization. Default is 'min'.

    args : tuple, optional
        Extra positional arguments passed to the objective function.
        Not allowed in 'dict' mode (will raise ValueError).
    kwargs : dict, optional
        Additional keyword arguments to pass to the objective function.
        In 'dict' mode, if keys conflict with x0 keys, x0 values take precedence
        and a warning is issued.
    maxiter : int, optional
        The maximum number of global search iterations. Default value is 1000.
    minimizer_kwargs : dict, optional
        Keyword arguments to be passed to the local minimizer
        (`minimize`). An important option could be ``method`` for the minimizer
        method to use.
        If no keyword arguments are provided, the local minimizer defaults to
        'L-BFGS-B' and uses the already supplied bounds. If `minimizer_kwargs`
        is specified, then the dict must contain all parameters required to
        control the local minimization. `args` is ignored in this dict, as it is
        passed automatically. `bounds` is not automatically passed on to the
        local minimizer as the method may not support them.
    initial_temp : float, optional
        The initial temperature, use higher values to facilitates a wider
        search of the energy landscape, allowing dual_annealing to escape
        local minima that it is trapped in. Default value is 5230. Range is
        (0.01, 5.e4].
    restart_temp_ratio : float, optional
        During the annealing process, temperature is decreasing, when it
        reaches ``initial_temp * restart_temp_ratio``, the reannealing process
        is triggered. Default value of the ratio is 2e-5. Range is (0, 1).
    visit : float, optional
        Parameter for visiting distribution. Default value is 2.62. Higher
        values give the visiting distribution a heavier tail, this makes
        the algorithm jump to a more distant region. The value range is (1, 3].
    accept : float, optional
        Parameter for acceptance distribution. It is used to control the
        probability of acceptance. The lower the acceptance parameter, the
        smaller the probability of acceptance. Default value is -5.0 with
        a range (-1e4, -5].
    maxfun : int, optional
        Soft limit for the number of objective function calls. If the
        algorithm is in the middle of a local search, this number will be
        exceeded, the algorithm will stop just after the local search is
        done. Default value is 1e7.
    rng : `numpy.random.Generator`, optional
        Pseudorandom number generator state. When `rng` is None, a new
        `numpy.random.Generator` is created using entropy from the
        operating system. Types other than `numpy.random.Generator` are
        passed to `numpy.random.default_rng` to instantiate a `Generator`.

        Specify `rng` for repeatable minimizations. The random numbers
        generated only affect the visiting distribution function
        and new coordinates generation.
    no_local_search : bool, optional
        If `no_local_search` is set to True, a traditional Generalized
        Simulated Annealing will be performed with no local search
        strategy applied.
    callback : callable, optional
        A callback function with signature ``callback(x, f, context)``,
        which will be called for all minima found.
        ``x`` and ``f`` are the coordinates and function value of the
        latest minimum found, and ``context`` has one of the following
        values:

        - ``0``: minimum detected in the annealing process.
        - ``1``: detection occurred in the local search process.
        - ``2``: detection done in the dual annealing process.

        In array mode, ``x`` is a numpy array. In dict mode, ``x`` is a dict
        mapping variable names to values.

        If the callback implementation returns True, the algorithm will stop.
    x0 : array_like or dict, optional
        Initial guess. Array of real elements of size (n,), or dict with
        variable names as keys. If not provided, the variable structure is
        inferred from ``bounds``.
    direction : {'min', 'max'}, optional
        Direction of optimization. Default is 'min'. If 'max', the function
        will be maximized instead of minimized by internally negating the
        objective function.
    var_mode : {'array', 'dict'}, optional
        Expected format of variables for ``x0``, ``func``, and ``bounds``.
        If None, inferred from the type of ``x0`` or ``bounds``. Use 'dict'
        to work with named variables as dicts instead of numpy arrays.

    Returns
    -------
    res : OptimizeResult
        The optimization result represented as a `OptimizeResult` object.
        Important attributes are: ``x`` the solution array, ``fun`` the value
        of the function at the solution, and ``message`` which describes the
        cause of the termination.
        See `OptimizeResult` for a description of other attributes.

    Notes
    -----
    This function implements the Dual Annealing optimization. This stochastic
    approach derived from [3]_ combines the generalization of CSA (Classical
    Simulated Annealing) and FSA (Fast Simulated Annealing) [1]_ [2]_ coupled
    to a strategy for applying a local search on accepted locations [4]_.
    An alternative implementation of this same algorithm is described in [5]_
    and benchmarks are presented in [6]_. This approach introduces an advanced
    method to refine the solution found by the generalized annealing
    process. This algorithm uses a distorted Cauchy-Lorentz visiting
    distribution, with its shape controlled by the parameter :math:`q_{v}`

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
    scipy_result = scipy_dual_annealing(
        # Pass wrapped args
        func=problem.scipy.get_func(),
        bounds=problem.scipy.get_bounds(),
        callback=problem.scipy.get_callback(),
        x0=problem.scipy.get_x0(),

        # Pass args that aren't wrapping
        maxiter=maxiter,
        minimizer_kwargs=minimizer_kwargs,
        initial_temp=initial_temp,
        restart_temp_ratio=restart_temp_ratio,
        visit=visit,
        accept=accept,
        maxfun=maxfun,
        rng=rng,
        no_local_search=no_local_search,
    )

    return problem.scipy.interpret_result(scipy_result)

    