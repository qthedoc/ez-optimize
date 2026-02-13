from typing import Literal, get_args

## TODO: create more robust way to generate these for tests and validation
# im thinking giant dict of all method props, then generate these from that dict

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

MinimizeMethod_NoGrad = Literal[
    "nelder-mead",
    "powell",
    "CG",
    "BFGS",
    "L-BFGS-B",
]

MINIMIZE_METHODS_NO_GRAD = tuple(get_args(MinimizeMethod_NoGrad))

MinimizeMethod_OptionalGrad = Literal[
    "TNC",
    "COBYLA",
    "SLSQP",
    "trust-constr",
]

MINIMIZE_METHODS_OPTIONAL_GRAD = tuple(get_args(MinimizeMethod_OptionalGrad))

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
