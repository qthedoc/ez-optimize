from __future__ import annotations
import warnings

from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from scipy.optimize import minimize as scipy_minimize

from ez_optimize.optimization_problem import OptimizationProblem
from ez_optimize.utilities import EzOptimizeResult
from ez_optimize.constants import MinimizeMethod

def minimize(
    fun: Callable,
    x0: Union[np.ndarray, Dict[str, Any]],
    method: Optional[MinimizeMethod] = None,
    args: Optional[Tuple] = None,
    kwargs: Optional[Dict[str, Any]] = None,
    jac = None,
    hess = None,
    hessp = None,
    bounds: Optional[Union[List[Tuple[float, float]], Dict[str, Tuple[float, float]], Dict[str, List[Tuple[float, float]]]]] = None,
    constraints = None,
    tol: Optional[float] = None,
    options: Optional[Dict[str, Any]] = None,
    callback: Optional[Callable] = None,
    direction: Literal["min", "max"] = "min",
    var_mode: Optional[Literal["array", "dict"]] = None,
) -> EzOptimizeResult:
    """Minimize (or maximize) a scalar function of one or more variables.
    
    This is a drop-in scipy.optimize.minimize() wrapper with enhanced features:
    - keyword vars (e.g. x0={'x': 1, 'y': 2} instead of x0=[1, 2])
    - direction (min or max)

    Parameters
    ----------
    fun : callable
        The objective function to be minimized:

        For array mode::
            fun(x, *args, **kwargs) -> float
        where ``x`` is a numpy array.

        For dict mode::
            fun(**vars, **kwargs) -> float
        where ``vars`` is a dict of independent variables.

        additional positional and keyword arguments can be passed via `args` and 
        `kwargs` parameters of this function. 

        However it is generally recommended to wrap the function to only accept 
        the optimization variables. e.g., pass ``fun=lambda x: f0(x, *my_args, **my_kwargs)`` as the
        callable, where ``my_args`` (tuple) and ``my_kwargs`` (dict) have been
        gathered before invoking this function.
    x0 : array_like or dict
        Initial guess. Array of real elements of size (n,),
        or dict with variable names as keys.
    method : str or callable, optional
        Type of solver.  Should be one of

        - 'Nelder-Mead' :ref:`(see here) <optimize.minimize-neldermead>`
        - 'Powell'      :ref:`(see here) <optimize.minimize-powell>`
        - 'CG'          :ref:`(see here) <optimize.minimize-cg>`
        - 'BFGS'        :ref:`(see here) <optimize.minimize-bfgs>`
        - 'Newton-CG'   :ref:`(see here) <optimize.minimize-newtoncg>`
        - 'L-BFGS-B'    :ref:`(see here) <optimize.minimize-lbfgsb>`
        - 'TNC'         :ref:`(see here) <optimize.minimize-tnc>`
        - 'COBYLA'      :ref:`(see here) <optimize.minimize-cobyla>`
        - 'COBYQA'      :ref:`(see here) <optimize.minimize-cobyqa>`
        - 'SLSQP'       :ref:`(see here) <optimize.minimize-slsqp>`
        - 'trust-constr':ref:`(see here) <optimize.minimize-trustconstr>`
        - 'dogleg'      :ref:`(see here) <optimize.minimize-dogleg>`
        - 'trust-ncg'   :ref:`(see here) <optimize.minimize-trustncg>`
        - 'trust-exact' :ref:`(see here) <optimize.minimize-trustexact>`
        - 'trust-krylov' :ref:`(see here) <optimize.minimize-trustkrylov>`
        - custom - a callable object, see below for description.

        If not given, chosen to be one of ``BFGS``, ``L-BFGS-B``, ``SLSQP``,
        depending on whether or not the problem has constraints or bounds.
    args : tuple, optional
        Extra positional arguments passed to the objective function and its
        derivatives (`fun`, `jac` and `hess` functions).
        Not allowed in 'dict' mode (will raise ValueError).
    kwargs : dict, optional
        Additional keyword arguments to pass to the objective function.
        In 'dict' mode, if keys conflict with x0 keys, x0 values take precedence
        and a warning is issued.
    jac : {callable,  '2-point', '3-point', 'cs', bool}, optional
        Method for computing the gradient vector. Only for CG, BFGS,
        Newton-CG, L-BFGS-B, TNC, SLSQP, dogleg, trust-ncg, trust-krylov,
        trust-exact and trust-constr.
        If it is a callable, it should be a function that returns the gradient
        vector::

            jac(x, *args) -> array_like, shape (n,)

        where ``x`` is an array with shape (n,) and ``args`` is a tuple with
        the fixed parameters. If `jac` is a Boolean and is True, `fun` is
        assumed to return a tuple ``(f, g)`` containing the objective
        function and the gradient.
        Methods 'Newton-CG', 'trust-ncg', 'dogleg', 'trust-exact', and
        'trust-krylov' require that either a callable be supplied, or that
        `fun` return the objective and gradient.
        If None or False, the gradient will be estimated using 2-point finite
        difference estimation with an absolute step size.
        Alternatively, the keywords  {'2-point', '3-point', 'cs'} can be used
        to select a finite difference scheme for numerical estimation of the
        gradient with a relative step size. These finite difference schemes
        obey any specified `bounds`.
    hess : {callable, '2-point', '3-point', 'cs', HessianUpdateStrategy}, optional
        Method for computing the Hessian matrix. Only for Newton-CG, dogleg,
        trust-ncg, trust-krylov, trust-exact and trust-constr.
        If it is callable, it should return the Hessian matrix::

            hess(x, *args) -> {LinearOperator, spmatrix, array}, (n, n)

        where ``x`` is a (n,) ndarray and ``args`` is a tuple with the fixed
        parameters.
        The keywords {'2-point', '3-point', 'cs'} can also be used to select
        a finite difference scheme for numerical estimation of the hessian.
        Alternatively, objects implementing the `HessianUpdateStrategy`
        interface can be used to approximate the Hessian. Available
        quasi-Newton methods implementing this interface are:

        - `BFGS`
        - `SR1`

        Not all of the options are available for each of the methods; for
        availability refer to the notes.
    hessp : callable, optional
        Hessian of objective function times an arbitrary vector p. Only for
        Newton-CG, trust-ncg, trust-krylov, trust-constr.
        Only one of `hessp` or `hess` needs to be given. If `hess` is
        provided, then `hessp` will be ignored. `hessp` must compute the
        Hessian times an arbitrary vector::

            hessp(x, p, *args) ->  ndarray shape (n,)

        where ``x`` is a (n,) ndarray, ``p`` is an arbitrary vector with
        dimension (n,) and ``args`` is a tuple with the fixed
        parameters.
    bounds : sequence or `Bounds`, optional
        Bounds on variables for Nelder-Mead, L-BFGS-B, TNC, SLSQP, Powell,
        trust-constr, COBYLA, and COBYQA methods. There are two ways to specify
        the bounds:

        1. Instance of `Bounds` class.
        2. Sequence of ``(min, max)`` pairs for each element in `x`. None
           is used to specify no bound.

    constraints : {Constraint, dict} or List of {Constraint, dict}, optional
        Constraints definition. Only for COBYLA, COBYQA, SLSQP and trust-constr.

        Constraints for 'trust-constr', 'cobyqa', and 'cobyla' are defined as a single
        object or a list of objects specifying constraints to the optimization problem.
        Available constraints are:

        - `LinearConstraint`
        - `NonlinearConstraint`

        Constraints for COBYLA, SLSQP are defined as a list of dictionaries.
        Each dictionary with fields:

        type : str
            Constraint type: 'eq' for equality, 'ineq' for inequality.
        fun : callable
            The function defining the constraint.
        jac : callable, optional
            The Jacobian of `fun` (only for SLSQP).
        args : sequence, optional
            Extra arguments to be passed to the function and Jacobian.

        Equality constraint means that the constraint function result is to
        be zero whereas inequality means that it is to be non-negative.

    tol : float, optional
        Tolerance for termination. When `tol` is specified, the selected
        minimization algorithm sets some relevant solver-specific tolerance(s)
        equal to `tol`. For detailed control, use solver-specific
        options.
    options : dict, optional
        A dictionary of solver options. All methods except `TNC` accept the
        following generic options:

        maxiter : int
            Maximum number of iterations to perform. Depending on the
            method each iteration may use several function evaluations.

            For `TNC` use `maxfun` instead of `maxiter`.
        disp : bool
            Set to True to print convergence messages.

        For method-specific options, see :func:`show_options()`.
    callback : callable, optional
        A callable called after each iteration.

        All methods except TNC support a callable with
        the signature::

            callback(intermediate_result: EzOptimizeResult)

        note: 1 arg that is exactly `intermediate_result`
        where ``intermediate_result`` is a keyword parameter containing an
        `EzOptimizeResult` with attributes ``x`` and ``fun``, the present values
        of the parameter vector and objective function. Not all attributes of
        `EzOptimizeResult` may be present. The name of the parameter must be
        ``intermediate_result`` for the callback to be passed an `EzOptimizeResult`.
        These methods will also terminate if the callback raises ``StopIteration``.

        All methods except trust-constr (also) support a signature like::

            callback(xk)

        note: 2 or more args, any names
        where ``xk`` is the current parameter vector.

        Introspection is used to determine which of the signatures above to
        invoke.
    direction : {'min', 'max'}, optional
        Direction of optimization. Default is 'min'. If 'max', the function 
        will be maximized instead of minimized by internally negating the objective function.
    var_mode : {'array', 'dict'}, optional
        Expected format of variables for x0, fun, bounds. If None, inferred from x0. If 'dict', variables

    Returns
    -------
    res : EzOptimizeResult
        The optimization result represented as a ``EzOptimizeResult`` object.
        Important attributes are: ``x`` the solution array, ``success`` a
        Boolean flag indicating if the optimizer exited successfully and
        ``message`` which describes the cause of the termination. See
        `EzOptimizeResult` for a description of other attributes.

    Notes
    -----
    This function uses the OptimizationProblem class internally to handle
    variable flattening, bounds, and direction, then delegates to SciPy's
    minimize function.
    """

    # Wrapper handles relevant arguments to be transformed
    problem = OptimizationProblem(
        func=fun,
        x0=x0,
        method=method,
        direction=direction,
        var_mode=var_mode,
        bounds=bounds,
        args=args,
        kwargs=kwargs,
        callback=callback,
    )

    # Warnings
    if problem.var_mode == "dict" and jac is not None and not (isinstance(jac, bool) or isinstance(jac, str)):
        warnings.warn("Passing 'jac' in dict mode is not yet fully supported by ez_optimize.minimize. jac will be passed directly to SciPy's minimize.", UserWarning)

    if problem.var_mode == "dict" and hess is not None and not (isinstance(hess, str) or callable(hess)):
        warnings.warn("Passing 'hess' in dict mode is not yet fully supported by ez_optimize.minimize. hess will be passed directly to SciPy's minimize.", UserWarning)

    if problem.var_mode == "dict" and hessp is not None:
        warnings.warn("Passing 'hessp' in dict mode is not yet fully supported by ez_optimize.minimize. hessp will be passed directly to SciPy's minimize.", UserWarning)

    if problem.var_mode == "dict" and constraints is not None:
        warnings.warn("Passing 'constraints' in dict mode is not yet fully supported by ez_optimize.minimize. constraints will be passed directly to SciPy's minimize.", UserWarning)

    # Run SciPy
    scipy_result = scipy_minimize(
        # Pass wrapped args
        fun=problem.scipy.minimize.func(),
        x0=problem.scipy.minimize.x0(),
        bounds=problem.scipy.minimize.bounds(),
        callback=problem.scipy.minimize.callback(),

        # Pass args that aren't wrapped
        method=method,
        jac=jac,
        hess=hess,
        hessp=hessp,
        constraints=constraints,
        tol=tol,
        options=options,
    )

    return problem.scipy.minimize.interpret_result(scipy_result)
