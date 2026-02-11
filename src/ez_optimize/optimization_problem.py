from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np

from ez_optimize.utilities import EzOptimizeResult, wrap_reconstruct_args, wrap_negate_if_max
from ez_optimize.constants import MINIMIZE_METHODS, MinimizeMethod
from scipy.optimize import minimize as scipy_minimize, OptimizeResult  # lazy import

class OptimizationProblem:
    """
    Defines an optimization problem in a backend-agnostic way.

    Handles:
    - named / dict parameters
    - flattening of nested structures (scalars, arrays, matrices)
    - min / max direction
    - fused objective return formats

    SciPy-specific functionality is isolated in `.scipy` namespace.
    """

    def __init__(
        self,
        fun: Callable,
        x0: Union[np.ndarray, Dict[str, Any]],
        method: Optional[MinimizeMethod] = None,
        direction: Literal["min", "max"] = "min",
        bounds: Optional[Union[List[Tuple[float, float]], Dict[str, Tuple[float, float]], Dict[str, List[Tuple[float, float]]]]] = None,
        x_mode: Optional[Literal["array", "dict"]] = None,
        args: Optional[Tuple] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        
        # fused_fun_map: Optional[Union[Literal["dict"], tuple[str, ...], Dict[str, str]]] = None,
        # jac: Optional[Callable] = None,
        # hess: Optional[Callable] = None,
        # hessp: Optional[Callable] = None,
        **optimizer_kwargs,  # bounds, constraints, tol, options, etc. stored for later use
    ):
        # Detect mode & flattening of x0, store mapping for later reconstruction
        self._prepare_parameters(x0, x_mode)

        self._prepare_bounds(bounds)

        self.method = self._prepare_method(method)

        self.user_fun = fun
        self.direction = direction.lower()
        if self.direction not in ("min", "max"):
            raise ValueError("direction must be 'min' or 'max'")

        # Validate and store args and user_kwargs
        self.user_args = args if args is not None else ()
        self.user_kwargs = kwargs if kwargs is not None else {}
        
        # Validate args usage based on x_mode
        if self.x_mode == "dict" and len(self.user_args) > 0:
            raise ValueError("Positional args are not allowed in 'dict' mode. Use user_kwargs instead.")
        
        # Check for conflicting kwargs in dict mode
        if self.x_mode == "dict" and self.user_kwargs:
            conflicting_keys = set(self.x_keys) & set(self.user_kwargs.keys())
            if conflicting_keys:
                warnings.warn(
                    f"Conflicting parameter names found in x0 and user_kwargs: {conflicting_keys}. "
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

    def _prepare_parameters(self, x0, x_mode: Optional[str] = None):
        """Determine mode, flatten x0 and store mapping back to original structure."""

        # Determine mode: array vs dict
        self.x_mode = x_mode.lower() if x_mode is not None else ("dict" if isinstance(x0, dict) else "array")
        if self.x_mode not in ("array", "dict"):
            raise ValueError("x_mode must be 'array' or 'dict'")
        
        # For array mode, flatten x0 to 1D and store shape for restoration
        if self.x_mode == "array":
            x0_array = np.atleast_1d(np.asarray(x0, dtype=float))
            if x0_array.ndim > 1:
                raise ValueError("x0 must be 1D or scalar for array mode; multi-dimensional arrays are not supported")
            self.x0_flat = x0_array.flatten()
            self.x_map = x0_array.shape
            self.x_keys = None
            self.x_to_original = lambda x: x.reshape(self.x_map)
            return
        
        # dict mode
        if not isinstance(x0, dict):
            raise ValueError("x0 must be a dict when x_mode='dict'")
        
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
        self.x_to_original = self._reconstruct_dict

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
    
    def _prepare_bounds(self, bounds: Optional[Union[List[Tuple[float, float]], Dict[str, Tuple[float, float]], Dict[str, List[Tuple[float, float]]]]]):
        """
        Validates bounds against x_mode and x_map, and flattens bounds if in dict mode.
        """
        if bounds is None:
            self.bounds_flat = None
            return
        
        if self.x_mode == "array":
            # Expect bounds as list of (min, max) tuples matching x0 length
            if not isinstance(bounds, list) or len(bounds) != self.x0_flat.size or any(not isinstance(b, tuple) or len(b) != 2 for b in bounds):
                raise ValueError(f"Bounds must be a list of (min, max) tuples matching the size of x0 ({self.x0_flat.size}) in array mode.")
            self.bounds_flat = bounds
            return
        
        # Dict mode: expect bounds as dict with same keys as x0, values are (min, max) tuples or lists of tuples
        if not isinstance(bounds, dict):
            raise ValueError("Bounds must be a dict when x_mode='dict'")
        
        flat_bounds = []
        for k in self.x_keys:
            if k not in bounds:
                # fill in with (None, None) if key is missing
                flat_bounds.extend([(None, None)] * np.prod(self.x_map[k], dtype=int))
                continue

            b = bounds[k]
            shape = self.x_map[k]
            if isinstance(b, tuple) and len(shape) == 0:
                # Scalar parameter with (min, max)
                flat_bounds.append(b)
            elif isinstance(b, list) and len(b) == np.prod(shape) and all(isinstance(t, tuple) and len(t) == 2 for t in b):
                # Parameter with shape > 0, bounds provided as list of tuples
                flat_bounds.extend(b)
            else:
                raise ValueError(f"Bounds for key '{k}' must be a tuple for scalar parameters or a list of tuples matching the size of the parameter in dict mode.")
        
        if len(flat_bounds) != self.x0_flat.size:
            raise ValueError(f"Internal error preparing bounds: Number of bounds ({len(flat_bounds)}) does not match number of parameters ({self.x0_flat.size})")

        self.bounds_flat = flat_bounds
    
    @staticmethod
    def _prepare_method(method: Optional[str]) -> Optional[str]:
        if method is None or not isinstance(method, str):
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
                "fun": self._wrap_fun(),
                "x0": self.parent.x0_flat,
                "method": method,
                "bounds": self.parent.bounds_flat,
            }

            # Merge any extra kwargs
            args.update(self.parent.optimizer_kwargs)

            return args

        def interpret_result(self, scipy_result: OptimizeResult) -> EzOptimizeResult:
            """Convert SciPy result into EasyOptimizeResult with restored structure."""

            # result_dict = dict(scipy_result)

            # Check success and warn if not successful
            if not scipy_result.success:
                warnings.warn(f"Optimization did not converge: {scipy_result.message}", RuntimeWarning)

            return EzOptimizeResult(
                scipy_result=scipy_result,
                x_mode=self.parent.x_mode,
                x_map=self.parent.x_map,
                x_to_original=self.parent.x_to_original,
                direction=self.parent.direction,
            )
        
        # ─── SciPy-specific wrappers ────────────────────────────────────────

        def _wrap_fun(self) -> Callable:
            """
            Wrap the user's objective function, return a scipy-ready function
            """
            fun = self.parent.user_fun

            # Argument transformation and wrapping for SciPy
            fun = wrap_reconstruct_args(
                fun=fun,
                x_mode=self.parent.x_mode,
                x_to_original=self.parent.x_to_original,
                user_args=self.parent.user_args,
                user_kwargs=self.parent.user_kwargs,
            )

            fun = wrap_negate_if_max(fun, self.parent.direction)

            return fun
