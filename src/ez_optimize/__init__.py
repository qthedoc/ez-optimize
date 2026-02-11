import logging

from .minimize import minimize
from .optimization_problem import OptimizationProblem
from ._version import __version__

_log = logging.getLogger(__name__)

if not logging.root.handlers:
    _log.setLevel(logging.INFO)
    if len(_log.handlers) == 0:
        handler = logging.StreamHandler()
        _log.addHandler(handler)

# __version__ = get_versions()["version"]
__all__ = ["minimize", "OptimizationProblem"]