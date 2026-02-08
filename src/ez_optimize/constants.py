from typing import Literal, get_args

MinimizeMethod = Literal[
    "nelder-mead",
    "powell",
    "CG",
    "BFGS",
    "Newton-CG",
    "L-BFGS-B",
    "TNC",
    "COBYLA",
    "SLSQP",
    "trust-constr",
    "dogleg",
    "trust-ncg",
    "trust-exact",
    "trust-krylov",
]

MINIMIZE_METHODS = tuple(get_args(MinimizeMethod))

RootMethod = Literal[
    "hybr",
    "lm",
    "broyden1",
    "broyden2",
    "anderson",
    "linearmixing",
    "diagbroyden",
    "excitingmixing",
    "krylov",
    "df-sane",
]

ROOT_METHODS = tuple(get_args(RootMethod))
