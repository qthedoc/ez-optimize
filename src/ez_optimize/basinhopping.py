from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from scipy.optimize import basinhopping as scipy_basinhopping

from ez_optimize.optimization_problem import OptimizationProblem
from ez_optimize.utilities import EzOptimizeResult

def basinhopping(
    func: Callable,
    x0: Union[np.ndarray, Dict[str, Any]],
    direction: Literal["min", "max"] = "min",
    args: Optional[Tuple] = None,
    kwargs: Optional[Dict[str, Any]] = None,
    niter: int = 100,
    T: float = 1.0,
    stepsize: float = 0.5,
    minimizer_kwargs=None,
    take_step: Optional[Callable] = None,
    accept_test: Optional[Callable] = None,
    callback: Optional[Callable] = None,
    interval: int = 50,
    disp: bool = False,
    niter_success: Optional[int] = None,
    rng=None,
    *,
    target_accept_rate: float = 0.5,
    stepwise_factor: float = 0.9,
    var_mode: Optional[Literal["array", "dict"]] = None,
):
    """Find the global minimum of a function using the basin-hopping algorithm.

    This is a drop-in scipy.optimize.basinhopping() wrapper with enhanced features:
    - keyword variables (e.g. x0={'x': 1, 'y': 2} instead of x0=[1, 2])
    - direction (min or max)

    Basin-hopping is a two-phase method that combines a global stepping
    algorithm with local minimization at each step. Designed to mimic
    the natural process of energy minimization of clusters of atoms, it works
    well for similar problems with "funnel-like, but rugged" energy landscapes.

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
    x0 : array_like or dict
        Initial guess. Array of real elements of size (n,), or dict with
        variable names as keys.
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
    niter : int, optional
        The number of basin-hopping iterations. There will be a total of
        ``niter + 1`` runs of the local minimizer. Default is 100.
    T : float, optional
        The "temperature" parameter for the acceptance or rejection criterion.
        Higher "temperatures" mean that larger jumps in function value will be
        accepted. For best results `T` should be comparable to the
        separation (in function value) between local minima. Default is 1.0.
    stepsize : float, optional
        Maximum step size for use in the random displacement. Default is 0.5.
    minimizer_kwargs : dict, optional
        Extra keyword arguments to be passed to the local minimizer
        ``scipy.optimize.minimize``. Some important options could be:

        method : str
            The minimization method (e.g. ``"L-BFGS-B"``)
        args : tuple
            Extra arguments passed to the objective function (``func``) and
            its derivatives (Jacobian, Hessian).
    take_step : callable, optional
        Replace the default step-taking routine with this routine. The default
        step-taking routine is a random displacement of the coordinates, but
        other step-taking algorithms may be better for some systems.
        ``take_step`` can optionally have the attribute ``take_step.stepsize``.
        If this attribute exists, then ``basinhopping`` will adjust
        ``take_step.stepsize`` in order to try to optimize the global minimum
        search.
    accept_test : callable, optional
        Define a test which will be used to judge whether to accept the
        step. This will be used in addition to the Metropolis test based on
        "temperature" ``T``. The acceptable return values are True,
        False, or ``"force accept"``.
    callback : callable, optional
        A callback function which will be called for all minima found.
        ``callback(x, f, accept)`` where ``x`` and ``f`` are the coordinates
        and function value of the trial minimum, and ``accept`` is whether
        that minimum was accepted.

        In array mode, ``x`` is a numpy array. In dict mode, ``x`` is a dict
        mapping variable names to values.

        If the callback returns True, the ``basinhopping`` routine will stop.
    interval : int, optional
        Interval for how often to update the ``stepsize``. Default is 50.
    disp : bool, optional
        Set to True to print status messages. Default is False.
    niter_success : int, optional
        Stop the run if the global minimum candidate remains the same for this
        number of iterations.
    rng : `numpy.random.Generator`, optional
        Pseudorandom number generator state. When `rng` is None, a new
        `numpy.random.Generator` is created using entropy from the
        operating system. Types other than `numpy.random.Generator` are
        passed to `numpy.random.default_rng` to instantiate a ``Generator``.

        Specify `rng` for repeatable minimizations.
    target_accept_rate : float, optional
        The target acceptance rate that is used to adjust the ``stepsize``.
        Range is (0, 1). Default is 0.5.
    stepwise_factor : float, optional
        The ``stepsize`` is multiplied or divided by this stepwise factor upon
        each update. Range is (0, 1). Default is 0.9.
    var_mode : {'array', 'dict'}, optional
        Expected format of variables for ``x0``, ``func``, and ``bounds``.
        If None, inferred from the type of ``x0``. Use 'dict'
        to work with named variables as dicts instead of numpy arrays.

    Returns
    -------
    res : OptimizeResult
        The optimization result represented as a ``OptimizeResult`` object.
        Important attributes are: ``x`` the solution array, ``fun`` the value
        of the function at the solution, and ``message`` which describes the
        cause of the termination.
        See ``OptimizeResult`` for a description of other attributes.
    """

    # Wrapper handles relevant arguments to be transformed
    problem = OptimizationProblem(
        func=func,
        x0=x0,
        direction=direction,
        var_mode=var_mode,
        args=args,
        kwargs=kwargs,
        callback=callback,
    )

    # Warnings
    if problem.var_mode == "dict" and take_step is not None:
        warnings.warn("Passing 'take_step' in dict mode is not yet fully supported by ez_optimize.basinhopping. take_step will receive flat arrays instead of dicts.", UserWarning)

    if problem.var_mode == "dict" and accept_test is not None:
        warnings.warn("Passing 'accept_test' in dict mode is not yet fully supported by ez_optimize.basinhopping. accept_test will receive flat arrays instead of dicts.", UserWarning)

    if problem.var_mode == "dict" and minimizer_kwargs is not None:
        warnings.warn("Passing 'minimizer_kwargs' in dict mode is not yet fully supported by ez_optimize.basinhopping. minimizer_kwargs will be passed directly to SciPy's minimize.", UserWarning)

    # Run SciPy
    scipy_result = scipy_basinhopping(
        # Pass wrapped args
        func=problem.scipy.basinhopping.func(),
        x0=problem.scipy.basinhopping.x0(),
        callback=problem.scipy.basinhopping.callback(),

        # Pass args that aren't wrapped
        niter=niter,
        T=T,
        stepsize=stepsize,
        minimizer_kwargs=minimizer_kwargs,
        take_step=take_step,
        accept_test=accept_test,
        interval=interval,
        disp=disp,
        niter_success=niter_success,
        rng=rng,
        target_accept_rate=target_accept_rate,
        stepwise_factor=stepwise_factor,
    )

    return problem.scipy.basinhopping.interpret_result(scipy_result)
