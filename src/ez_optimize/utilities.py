import logging

import inspect
from collections.abc import Callable
from typing import Any, Dict, Optional, Union
import warnings
from scipy.optimize import OptimizeResult as ScipyOptimizeResult

import numpy as np

_log = logging.getLogger(__name__)

def call_with_kwargs(func: Callable, kwargs: Dict[str, Any]) -> Any:
    '''
    Calls a function with matching kwargs, passing all provided kwargs if the function has **kwargs,
    while checking for missing required arguments. Functions with *args are not supported and log a warning.

    Args:
        func: The function to call.
        kwargs: Dictionary of keyword arguments to pass.

    Returns:
        The result of the function call.

    Raises:
        ValueError: If required arguments (without defaults) are missing.
    '''
    # Get the function signature
    signature = inspect.signature(func)
    parameters = signature.parameters

    # Check for *args (VAR_POSITIONAL)
    if any(param.kind == inspect.Parameter.VAR_POSITIONAL for param in parameters.values()):
        _log.warning(
            f"Function '{func.__name__}' has a *args parameter, which may not be fully supported by call_with_kwargs. "
            "Consider using keyword arguments or **kwargs instead."
        )

    # Check for missing required arguments
    missing_args = []
    for name, param in parameters.items():
        if (
            param.default == inspect.Parameter.empty
            and param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            and name not in kwargs
        ):
            missing_args.append(name)

    # Raise an error if required arguments are missing
    if missing_args:
        missing_args_str = [f"'{arg}'" for arg in missing_args]
        raise ValueError(
            f"Missing required parameters for function '{func.__name__}': {', '.join(missing_args_str)}"
        )

    # Check if the function has a **kwargs parameter
    has_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values())

    if has_kwargs:
        # Pass all kwargs, including those matching named parameters and extras
        return func(**kwargs)
    else:
        # Pass only kwargs that match the function's parameters, log if extras
        extra_kwargs = set(kwargs.keys()) - set(parameters.keys())
        if extra_kwargs:
            _log.warning(
                f"Extra kwargs provided to '{func.__name__}' without **kwargs support: {extra_kwargs}. They will be ignored."
            )
        matched_kwargs = {name: kwargs[name] for name in kwargs if name in parameters}
        return func(**matched_kwargs)

def wrap_reconstruct_args(
    fun: Optional[Callable],
    x_mode: str,
    x_to_original: Callable[[np.ndarray], Any],
    pos_arg_names: list[str] = None,               # e.g. ['p'] for hessp
) -> Optional[Callable]:
    """
    Returns a wrapped version of a function that:
    - take flat x (and optionally other args like p) for SciPy compatibility
    - reconstruct original parameter structure
    - call user function

    Args:
        fun: The user-provided callable (objective, jac, hess, hessp, ...)
        x_mode: "array" or "dict"
        x_to_original: Function that turns flat array → original x (dict or array)
        pos_arg_names: Names to map positional args to (e.g. ['p'] for hessp)
    """
    if fun is None:
        return None

    pos_arg_names = pos_arg_names or []

    def wrapped(x_flat: np.ndarray, *extra_args) -> Any:
        # Reconstruct original parameters
        x = x_to_original(x_flat)

        # Prepare call arguments
        if x_mode == "dict":
            call_args = x.copy()  # dict
        else:
            call_args = [x]       # positional for array mode

        # Map positional extra args (e.g. p for hessp)
        extra_kwargs = {}
        for name, value in zip(pos_arg_names, extra_args):
            if name in extra_kwargs:
                raise ValueError(f"Duplicate argument: {name}")
            extra_kwargs[name] = value

        # Call the function
        if x_mode == "dict":
            result = call_with_kwargs(fun, {**call_args, **extra_kwargs})
        else:
            result = fun(*call_args, *extra_args)

        return result

    return wrapped

def wrap_negate_if_max(fun: Callable, direction: str) -> Callable:
    """
    Wraps a function to negate its output if the optimization direction is 'max'.

    Args:
        fun: The original function to wrap.
        direction: 'min' or 'max'. If 'max', the output of `fun` will be negated.
    Returns:
        A wrapped function that negates the output of `fun` if direction is 'max'.
    """
    if direction == 'min':
        return fun
    
    def wrapped(*args, **kwargs):
        return -fun(*args, **kwargs)

    return wrapped

class EzOptimizeResult():
    """
    Enhanced result object for optimizations performed via ez-optimize.

    """
    def __init__(
            self,
            scipy_result: ScipyOptimizeResult,
            x_mode: str = 'array', 
            x_map: Optional[list[str]] = None, 
            x_to_original: Optional[Callable[[np.ndarray], Any]] = None,
            direction: str = 'min',
        ):
        # super().__init__(**scipy_result.__dict__)
        self.scipy_result = scipy_result
        self._x_mode = x_mode  # Store mode ('array' or 'dict')
        self._x_map = x_map  # Store sorted keys from x0 dict
        self._x_to_original = x_to_original
        self._direction = direction
        self.x_flat = scipy_result.x  # flat optimized parameters

        # Process: Restore x to original structure
        if self._x_to_original:
            self.x = self._x_to_original(scipy_result.x)
        else:
            self.x = scipy_result.x

        # Process: Correct sign for maximization
        self.fun = scipy_result.fun
        if self._direction == "max" and self.fun is not None:
            self.fun = -self.fun
        # Similarly for jac if present: self.jac = -scipy_result.jac if applicable
    
        # Attach other attributes from scipy_result
        for attr in dir(scipy_result):
            if not attr.startswith('_') and attr not in ('x', 'fun', 'jac'):  # Avoid overwriting processed ones
                setattr(self, attr, getattr(scipy_result, attr))

    