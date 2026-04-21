from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import inspect

import numpy as np

from ez_optimize.utilities import EzOptimizeResult, wrap_reconstruct_args, wrap_negate_if_max
from ez_optimize.constants import MINIMIZE_METHODS, MinimizeMethod
from scipy.optimize import minimize as scipy_minimize, OptimizeResult  # lazy import

class OptimizationProblem:
    """
    Defines an optimization problem in a backend-agnostic way.

    SciPy-specific functionality is isolated in `.scipy` namespace.
    """

    def __init__(
        self,
        func: Callable,
        x0: Optional[Union[np.ndarray, Dict[str, Any]]],
        method: Optional[MinimizeMethod] = None,
        direction: Literal["min", "max"] = "min",
        bounds: Optional[Union[List[Tuple[float, float]], Dict[str, Tuple[float, float]], Dict[str, List[Tuple[float, float]]]]] = None,
        args: Optional[Tuple] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        callback: Optional[Callable] = None,
        var_mode: Optional[Literal["array", "dict"]] = None,
        
        # fused_fun_map: Optional[Union[Literal["dict"], tuple[str, ...], Dict[str, str]]] = None,
        # jac: Optional[Callable] = None,
        # hess: Optional[Callable] = None,
        # hessp: Optional[Callable] = None,
        **optimizer_kwargs,  # bounds, constraints, tol, options, etc. stored for later use
    ):
        # Detect and prepare variable structure: flatten x0/bounds, set x_map, x_keys, x_to_original
        self._prepare_variable_layout(var_mode, x0, bounds)

        self.method = self._prepare_method(method)

        self.user_func = func
        self.direction = direction.lower()
        if self.direction not in ("min", "max"):
            raise ValueError("direction must be 'min' or 'max'")

        # Validate and store args and user_kwargs
        self.user_args = args if args is not None else ()
        self.user_kwargs = kwargs if kwargs is not None else {}
        self.user_callback = callback
        
        # Validate args usage based on var_mode
        if self.var_mode == "dict" and len(self.user_args) > 0:
            raise ValueError("Positional args are not allowed in 'dict' mode. Use user_kwargs instead.")
        
        # Check for conflicting kwargs in dict mode
        if self.var_mode == "dict" and self.user_kwargs:
            conflicting_keys = set(self.x_keys) & set(self.user_kwargs.keys())
            if conflicting_keys:
                warnings.warn(
                    f"Conflicting variable names found in x0 and user_kwargs: {conflicting_keys}. "
                    "Values from x0 will take precedence during optimization.",
                    UserWarning
                )

        # Pass-through optimizer_kwargs (bounds, constraints, tol, options, etc.)
        self.optimizer_kwargs = optimizer_kwargs

    # ────────────────────────────────────────────────────────────────
    # Public API – backend agnostic
    # ────────────────────────────────────────────────────────────────

    @property
    def scipy(self):
        """SciPy optimize sub-namespace: one interface per scipy.optimize function."""
        return OptimizationProblem._Scipy(self)

    def optimize(self) -> EzOptimizeResult:
        """
        Convenience: run optimization using SciPy backend and return interpreted result.
        """
        res = scipy_minimize(
            fun=self.scipy.minimize.func(),
            x0=self.scipy.minimize.x0(),
            method=self.method,
            bounds=self.scipy.minimize.bounds(),
            callback=self.scipy.minimize.callback(),
            **self.optimizer_kwargs,
        )
        return self.scipy.minimize.interpret_result(res)

    # ────────────────────────────────────────────────────────────────
    # Internal helpers – shared across backends
    # ────────────────────────────────────────────────────────────────

    def _prepare_variable_layout(self, var_mode, x0, bounds):
        """
        Single entry point for all variable structure setup.
        Sets x0_flat, bounds_flat, x_map, x_keys, and x_to_original.
        Structure is inferred from x0 if provided, otherwise from bounds.
        """
        self.var_mode = self._detect_mode(var_mode, x0, bounds)

        if self.var_mode == "array":
            self._prepare_variable_layout_array(x0, bounds)
        else:
            self._prepare_variable_layout_dict(x0, bounds)

    def _detect_mode(self, var_mode: Optional[str], x0, bounds) -> str:
        """Detect variable mode (array vs dict) based on var_mode argument and types of x0 and bounds."""
        # Priority 1: var_mode argument
        if var_mode is not None:
            if var_mode.lower() not in ("array", "dict"):
                raise ValueError("var_mode must be 'array' or 'dict'")
            return var_mode.lower()
        
        # Priority 2: Auto-detect mode based on x0 type
        if x0 is not None:
            if isinstance(x0, dict):
                return "dict"
            else:
                return "array"

        # Priority 3: Auto-detect mode based on bounds type
        if bounds is not None:
            if isinstance(bounds, dict):
                return "dict"
            else:
                return "array"

        # Default if nothing provided
        return "array"

    def _prepare_variable_layout_array(self, x0, bounds):
        self.x_keys = None
        self.x_to_original = lambda x: np.asarray(x).reshape(self.x_map)

        # Flatten x0 and derive x_map from it
        if x0 is not None:
            x0_array = np.atleast_1d(np.asarray(x0, dtype=float))
            if x0_array.ndim > 1:
                raise ValueError("x0 must be 1D or scalar for array mode; multi-dimensional arrays are not supported")
            self.x0_flat = x0_array.flatten()
            self.x_map = x0_array.shape
        else:
            self.x0_flat = None
            self.x_map = None  # Filled in from bounds below

        # Flatten and validate bounds
        if bounds is None:
            if self.x_map is None:
                raise ValueError("Either x0 or bounds must be provided in array mode")
            self.bounds_flat = None
            return

        if not isinstance(bounds, list) or any(not isinstance(b, tuple) or len(b) != 2 for b in bounds):
            raise ValueError("Bounds must be a list of (min, max) tuples in array mode.")

        if self.x_map is None:
            self.x_map = (len(bounds),)  # Infer shape from bounds when x0 is absent
        else:
            if len(bounds) != np.prod(self.x_map, dtype=int):
                raise ValueError(f"Bounds length ({len(bounds)}) does not match x0 size ({np.prod(self.x_map, dtype=int)}) in array mode.")

        self.bounds_flat = bounds

    def _prepare_variable_layout_dict(self, x0, bounds):
        self.x_to_original = self._reconstruct_dict

        # Build x_keys and x_map from x0
        if x0 is not None:
            if not isinstance(x0, dict):
                raise ValueError("x0 must be a dict when var_mode='dict'")
            self.x_keys = list(x0.keys())
            self.x_map = {}
            flat_parts = []
            for k in self.x_keys:
                val = np.asarray(x0[k], dtype=float)
                if val.ndim > 1:
                    raise ValueError(f"x0['{k}'] must be 1D or scalar for dict mode; multi-dimensional arrays are not supported")
                self.x_map[k] = val.shape
                flat_parts.append(val.ravel())
            self.x0_flat = np.concatenate(flat_parts) if flat_parts else np.array([])
        else:
            self.x0_flat = None
            self.x_keys = None  # Filled in from bounds below
            self.x_map = {}

        # Flatten and validate bounds
        if bounds is None:
            if self.x_keys is None:
                raise ValueError("Either x0 or bounds must be provided in dict mode")
            self.bounds_flat = None
            return

        if not isinstance(bounds, dict):
            raise ValueError("Bounds must be a dict when var_mode='dict'")

        # Infer x_keys and x_map from bounds when x0 is absent
        if self.x_keys is None:
            self.x_keys = list(bounds.keys())
            self.x_map = {}
            for k in self.x_keys:
                b = bounds[k]
                if isinstance(b, tuple):
                    self.x_map[k] = ()
                elif isinstance(b, list) and all(isinstance(t, tuple) and len(t) == 2 for t in b):
                    self.x_map[k] = (len(b),)
                else:
                    raise ValueError(f"Bounds for key '{k}' must be a (min, max) tuple for scalars or a list of tuples for arrays.")

        flat_bounds = []
        for k in self.x_keys:
            if k not in bounds:
                flat_bounds.extend([(None, None)] * np.prod(self.x_map[k], dtype=int))
                continue
            b = bounds[k]
            shape = self.x_map[k]
            if isinstance(b, tuple) and len(shape) == 0:
                flat_bounds.append(b)
            elif isinstance(b, list) and len(b) == np.prod(shape) and all(isinstance(t, tuple) and len(t) == 2 for t in b):
                flat_bounds.extend(b)
            else:
                raise ValueError(f"Bounds for key '{k}' must be a tuple for scalar variables or a list of tuples matching the variable size.")

        total_size = sum(np.prod(self.x_map[k], dtype=int) for k in self.x_keys)
        if len(flat_bounds) != total_size:
            raise ValueError(f"Internal error: bounds count ({len(flat_bounds)}) does not match variable count ({total_size})")
        self.bounds_flat = flat_bounds

    def _reconstruct_dict(self, flat: np.ndarray) -> Dict[str, Any]:
        result = {}
        idx = 0
        for k in self.x_keys:
            shape = self.x_map[k]
            size = np.prod(shape, dtype=int)
            if shape == ():  # Scalar case: convert to float
                result[k] = float(flat[idx])
            else:  # Array case: reshape to array
                result[k] = flat[idx : idx + size].reshape(shape)
            idx += size
        return result
    
    @staticmethod
    def _prepare_method(method: Optional[str]) -> Optional[str]:
        if method is None:
            return None
        if not isinstance(method, str):
            raise ValueError("method must be a str")
        if method not in MINIMIZE_METHODS:
            raise ValueError(f"Unsupported method '{method}'. Supported: {', '.join(MINIMIZE_METHODS)}")
        return method

    class _Scipy:
        """One-to-one namespace with scipy.optimize, providing a per-function interface."""

        def __init__(self, parent: "OptimizationProblem"):
            self._parent = parent

        @property
        def minimize(self):
            return OptimizationProblem._MinimizeInterface(self._parent)

        @property
        def differential_evolution(self):
            return OptimizationProblem._DifferentialEvolutionInterface(self._parent)

        @property
        def dual_annealing(self):
            return OptimizationProblem._DualAnnealingInterface(self._parent)

        @property
        def shgo(self):
            return OptimizationProblem._ShgoInterface(self._parent)

        @property
        def direct(self):
            return OptimizationProblem._DirectInterface(self._parent)

        @property
        def basinhopping(self):
            return OptimizationProblem._BasinhoppingInterface(self._parent)


    class _ScipyMethodBase:
        """
        Shared implementation for all scipy.optimize method interfaces.

        Subclasses must implement _wrap_callback() to provide the correct
        scipy callback signature for their specific method.
        """

        def __init__(self, parent: "OptimizationProblem"):
            self._parent = parent

        def func(self) -> Callable:
            """Return a scipy-ready objective function with argument reconstruction and direction handling."""
            fn = wrap_reconstruct_args(
                func=self._parent.user_func,
                var_mode=self._parent.var_mode,
                x_to_original=self._parent.x_to_original,
                user_args=self._parent.user_args,
                user_kwargs=self._parent.user_kwargs,
            )
            return wrap_negate_if_max(fn, self._parent.direction)

        def x0(self) -> Optional[np.ndarray]:
            """Return the flattened initial guess."""
            return self._parent.x0_flat

        def bounds(self) -> Optional[List[Tuple[float, float]]]:
            """Return the flattened bounds."""
            return self._parent.bounds_flat

        def callback(self) -> Optional[Callable]:
            """Return a scipy-ready callback or None."""
            if self._parent.user_callback is None:
                return None
            return self._wrap_callback()

        def interpret_result(self, scipy_result: OptimizeResult) -> EzOptimizeResult:
            """Convert a SciPy OptimizeResult into an EzOptimizeResult."""
            # TODO: i think we need better handling of cases where optimization fails and intermediate results
            # Check success and warn if not successful 
            if scipy_result.get('success', None) is False:
                warnings.warn(
                    f"Optimization did not converge: {scipy_result.get('message', '')}",
                    RuntimeWarning,
                )
            return EzOptimizeResult(
                scipy_result=scipy_result,
                var_mode=self._parent.var_mode,
                x_map=self._parent.x_map,
                x_to_original=self._parent.x_to_original,
                direction=self._parent.direction,
            )

        def _wrap_callback(self) -> Callable:
            raise NotImplementedError(f"{type(self).__name__} must implement _wrap_callback()")


    class _MinimizeInterface(_ScipyMethodBase):
        """
        Interface for scipy.optimize.minimize.

        Scipy minimize callback behavior:
        - callback(intermediate_result): if the single param is named exactly
          'intermediate_result', scipy passes an OptimizeResult object.
        - callback(xk): scipy passes the current parameter vector.
        - callback(xk, intermediate_result=None): trust-constr passes both.

        Introspection is used to select the correct path.
        """
        def fun(self) -> Callable:
            """Alias for func() to match scipy.optimize.minimize's expected argument name."""
            return self.func()

        def _wrap_callback(self) -> Callable:
            sig = inspect.signature(self._parent.user_callback)
            params = list(sig.parameters.keys())

            if len(params) == 1 and params[0] == 'intermediate_result':
                def wrapped(intermediate_result):
                    ez = self.interpret_result(intermediate_result) if intermediate_result is not None else None
                    self._parent.user_callback(intermediate_result=ez)
                return wrapped
            else:
                def wrapped(xk, intermediate_result=None):
                    x_reconstructed = self._parent.x_to_original(xk)
                    if isinstance(intermediate_result, OptimizeResult):
                        ez_intermediate = self.interpret_result(intermediate_result)
                    else:
                        ez_intermediate = intermediate_result
                    self._parent.user_callback(x_reconstructed, ez_intermediate)
                return wrapped


    class _DifferentialEvolutionInterface(_ScipyMethodBase):
        """
        Interface for scipy.optimize.differential_evolution.

        Scipy DE callback signatures:
        - callback(intermediate_result: OptimizeResult): if param is named 'intermediate_result'.
        - callback(x, convergence=val): x is the current best vector,
          convergence is the fractional population convergence value.
        """

        def _wrap_callback(self) -> Callable:
            sig = inspect.signature(self._parent.user_callback)
            params = list(sig.parameters.keys())

            if len(params) == 1 and params[0] == 'intermediate_result':
                def wrapped(intermediate_result):
                    ez = self.interpret_result(intermediate_result) if intermediate_result is not None else None
                    self._parent.user_callback(intermediate_result=ez)
                return wrapped
            else:
                def wrapped(x, convergence=None):
                    x_reconstructed = self._parent.x_to_original(x)
                    self._parent.user_callback(x_reconstructed, convergence)
                return wrapped


    class _DualAnnealingInterface(_ScipyMethodBase):
        """
        Interface for scipy.optimize.dual_annealing.

        Scipy dual_annealing callback signature:
        - callback(x, f, context): x is the parameter vector at the latest minimum,
          f is the function value, context is 0 (annealing), 1 (local search),
          or 2 (dual annealing).

        f is un-negated before being passed to the user callback when direction='max'.
        """

        def _wrap_callback(self) -> Callable:
            def wrapped(x, f, context):
                x_reconstructed = self._parent.x_to_original(x)
                if self._parent.direction == "max":
                    f = -f
                return self._parent.user_callback(x_reconstructed, f, context)
            return wrapped


    class _ShgoInterface(_ScipyMethodBase):
        """
        Interface for scipy.optimize.shgo.

        Scipy shgo callback signature:
        - callback(xk): xk is the current parameter vector.
        """

        def _wrap_callback(self) -> Callable:
            def wrapped(xk):
                x_reconstructed = self._parent.x_to_original(np.asarray(xk))
                return self._parent.user_callback(x_reconstructed)
            return wrapped


    class _DirectInterface(_ScipyMethodBase):
        """
        Interface for scipy.optimize.direct.

        Scipy direct callback signature:
        - callback(xk): xk is the current parameter vector.
        """

        def _wrap_callback(self) -> Callable:
            def wrapped(xk):
                x_reconstructed = self._parent.x_to_original(xk)
                return self._parent.user_callback(x_reconstructed)
            return wrapped


    class _BasinhoppingInterface(_ScipyMethodBase):
        """
        Interface for scipy.optimize.basinhopping.

        Scipy basinhopping callback signature:
        - callback(x, f, accept): x is the current parameter vector,
          f is the function value, accept is True/False for whether the step was accepted.

        f is un-negated before being passed to the user callback when direction='max'.
        """

        def _wrap_callback(self) -> Callable:
            def wrapped(x, f, accept):
                x_reconstructed = self._parent.x_to_original(x)
                if self._parent.direction == "max":
                    f = -f
                return self._parent.user_callback(x_reconstructed, f, accept)
            return wrapped
        

