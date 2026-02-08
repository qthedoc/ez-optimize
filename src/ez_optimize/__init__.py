import logging

from ez_optimize.minimize import minimize
from ez_optimize.optimization_problem import OptimizationProblem

_log = logging.getLogger(__name__)

if not logging.root.handlers:
    _log.setLevel(logging.INFO)
    if len(_log.handlers) == 0:
        handler = logging.StreamHandler()
        _log.addHandler(handler)

# __version__ = get_versions()["version"]
__all__ = ["minimize", "OptimizationProblem"]