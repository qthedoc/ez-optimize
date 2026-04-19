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
        self._prepare_variable_structure(var_mode, x0, bounds)

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
        """SciPy-specific interface."""
        return OptimizationProblem.SciPyInterface(self)

    def optimize(self) -> EzOptimizeResult:
        """
        Convenience: run optimization using SciPy backend and return interpreted result.
        """

        args = self.scipy.get_minimize_args()
        res = scipy_minimize(**args)
        return self.scipy.interpret_result(res)

    # ────────────────────────────────────────────────────────────────
    # Internal helpers – shared across backends
    # ────────────────────────────────────────────────────────────────

    def _prepare_variable_structure(self, var_mode, x0, bounds):
        """
        Single entry point for all variable structure setup.
        Sets x0_flat, bounds_flat, x_map, x_keys, and x_to_original.
        Structure is inferred from x0 if provided, otherwise from bounds.
        """
        self.var_mode = self._detect_mode(var_mode, x0, bounds)

        if self.var_mode == "array":
            self._prepare_variable_structure_array(x0, bounds)
        else:
            self._prepare_variable_structure_dict(x0, bounds)

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
    

    def _prepare_variable_structure_array(self, x0, bounds):
        self.x_keys = None
        self.x_to_original = lambda x: x.reshape(self.x_map)

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

    def _prepare_variable_structure_dict(self, x0, bounds):
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

    class SciPyInterface:
        """
        Everything related to scipy.optimize.minimize is isolated here.
        """

        def __init__(self, parent: "OptimizationProblem"):
            self.parent = parent

        def get_func(self) -> Callable:
            """Return a scipy-ready objective function, with argument reconstruction and direction handling."""
            return self._wrap_func()
        
        def get_x0(self) -> np.ndarray:
            """Return the flattened initial guess."""
            return self.parent.x0_flat
        
        def get_bounds(self) -> Optional[List[Tuple[float, float]]]:
            """Return the flattened bounds."""
            return self.parent.bounds_flat

        def get_callback(self) -> Optional[Callable]:
            """Return a scipy-ready callback function, with argument reconstruction and direction handling."""
            if self.parent.user_callback is not None:
                return self._wrap_callback()
            return None
        

        def get_minimize_args(
            self,
        ) -> dict:
            """Build arguments for scipy.optimize.minimize."""

            # Validate method
            method = self.parent.method
            if method is None or method not in MINIMIZE_METHODS:
                raise ValueError(f"Unsupported method '{method}'. Supported: {', '.join(MINIMIZE_METHODS)}")

            # Core arguments
            args = {
                "fun": self.get_func(),
                "x0": self.get_x0(),
                "method": method,
                "bounds": self.get_bounds(),
                "callback": self.get_callback(),
            }

            # Merge any extra kwargs
            args.update(self.parent.optimizer_kwargs)

            return args

        def interpret_result(self, scipy_result: OptimizeResult) -> EzOptimizeResult:
            """Convert SciPy result into EasyOptimizeResult with restored structure."""

            # TODO: i think we need better handling of cases where optimization fails and intermediate results
            # Check success and warn if not successful 
            if scipy_result.get('success', None) is False:
                warnings.warn(f"Optimization did not converge: {scipy_result.get('message', '')}", RuntimeWarning)

            return EzOptimizeResult(
                scipy_result=scipy_result,
                var_mode=self.parent.var_mode,
                x_map=self.parent.x_map,
                x_to_original=self.parent.x_to_original,
                direction=self.parent.direction,
            )
        
        # ─── SciPy-specific wrappers ────────────────────────────────────────

        def _wrap_func(self) -> Callable:
            """
            Wrap the user's objective function, return a scipy-ready function
            """
            fun = self.parent.user_func

            # Argument transformation and wrapping for SciPy
            fun = wrap_reconstruct_args(
                fun=fun,
                var_mode=self.parent.var_mode,
                x_to_original=self.parent.x_to_original,
                user_args=self.parent.user_args,
                user_kwargs=self.parent.user_kwargs,
            )

            fun = wrap_negate_if_max(fun, self.parent.direction)

            return fun
        
        def _wrap_callback(self) -> Callable:
            """
            Wrap the user's callback function to handle variable reconstruction and direction.
            Preserve SciPy callback signature behavior (callback(intermediate_result) or callback(xk)).

            SciPy callback behavior:
            - if: there is is one arg and its called `intermediate_result`
                scipy will pass an OptimizeResult object containing intermediate optimization results

            - else: (if there are two or more args, or if there is one arg but it's not called `intermediate_result`)
                scipy will pass the current parameter vector as the first argument, and optionally the OptimizeResult as a second argument (depending on the method and callback signature)
            """
            sig = inspect.signature(self.parent.user_callback)
            params = list(sig.parameters.keys())
            
            if len(params) == 1 and params[0] == 'intermediate_result':
                # Callback expects intermediate_result: OptimizeResult
                def wrapped_callback(intermediate_result):
                    ez_intermediate_result = self.interpret_result(intermediate_result) if intermediate_result is not None else None
                    self.parent.user_callback(intermediate_result=ez_intermediate_result)
                return wrapped_callback
            else:
                def wrapped_callback(xk, intermediate_result=None):
                    x_reconstructed = self.parent.x_to_original(xk)

                    # Reconstruct intermediate_result if user callback expects it
                    ez_intermediate_result = self.interpret_result(intermediate_result) if intermediate_result is not None else None
                    
                    # Call user callback with reconstructed x and optionally intermediate result
                    self.parent.user_callback(x_reconstructed, ez_intermediate_result)

                return wrapped_callback

