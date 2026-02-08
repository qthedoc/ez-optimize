
# ez-optimize

**Author**: Quinn Marsh\
**GitHub**: https://github.com/qthedoc/ez-optimize/ \
**PyPI**: https://pypi.org/project/ez-optimize/ 


`ez-optimize` is a more intuitive front-end for `scipy.optimize` that simplifies optimization with features like:
- keyword-based parameter definitions (e.g., `x0={'x': 1, 'y': 2}`)
- easy switching between minimization and maximization (`direction='max'`)

`ez-optimize` is the Ironman suit for optimization.

## Why ez-optimize?

### 1. Keyword-Based Optimization (e.g.: `x0={'x': 1, 'y': 2}`)
By default, optimization uses arrays `x0=[1, 2]`. However sometimes it's more intuitive to use named parameters `x0={'x': 1, 'y': 2}`. `ez-optimize` allows you to define parameters as dictionaries. Then under the hood, `ez-optimize` automatically flattens parameters (and wraps your function) for SciPy while restoring the original structure in results. Keyword-based optimization is especially useful in complex systems like aerospace or energy models where parameters have meaningful names representing physical quantities.

### 2. Switch to Maximize with `direction='max'`
By default, optimization minimizes the objective function. To maximize, you typically need to write a negated version of your function. With `ez-optimize`, simply set `direction='max'` and the library will automatically negates your function under the hood.

## Quick Start

**Install:** 
```
pip install ez-optimize
```

**Full set of examples**: [examples.ipynb](./docs/examples.ipynb)*\
*This is currently the main form of documentation.

### Example 1: Minimizing with Keyword-Based Parameters

```python
import numpy as np
from ez_optimize import minimize

def rosenbrock_2d(x, y, a, b):
    return (a - x)**2 + b * (y - x**2)**2

x0 = {'x': 1.3, 'y': 0.7}

result = minimize(rosenbrock_2d, x0, method='BFGS')

print(f"Optimal x: {result.x}")
print(f"Optimal value: {result.fun}")
```
```
Optimal x: {'x': 1.0, 'y': 1.0}
Optimal value: 0.0
```

### Example 2: Using OptimizationProblem for Advanced Manual Control

For more control, use the `OptimizationProblem` class directly. This also serves as a look under the hood for how `minimize` works.:

```python
from ez_optimize import OptimizationProblem
from scipy.optimize import minimize as scipy_minimize

def objective(a, b, c):
    return a**2 + b**2 + c**2

x0 = {'a': 1.0, 'b': 2.0, 'c': 3.0}
bounds = {'a': (0, 5), 'b': (0, 5), 'c': (0, 5)}

# Define the optimization problem
problem = OptimizationProblem(objective, x0, method='SLSQP', bounds=bounds)

# Run SciPy method directly, passing in the arguments prepared by the OptimizationProblem
scipy_result = scipy_minimize(**problem.scipy.get_minimize_args())

# Use the OptimizationProblem to interpret the result back into our structured format
result = problem.scipy.interpret_result(scipy_result)

print(f"Optimal parameters: {result.x}")
print(f"Optimal value: {result.fun}")
```
```
Optimal parameters: {'a': 0.0, 'b': 0.0, 'c': 0.0}
Optimal value: 0.0
```

## Fundumentally Why?
Lets be honest, there is good reason optimization typically uses arrays and always minimizes... it makes the math simple and efficient. For example, optimizing in a vector space allows the hessian to be represented as a matrix. However, this level of optimization isn't always necessary like with black-box functions that have no gradient or hessian. In those cases, the convenience of defining keyword-based parameters and easy switching between min/max can outweigh the mathematical perfection of array-based optimization.

## Acknowledgments

Inspired by [better_optimize](https://github.com/jessegrabowski/better_optimize) by Jesse Grabowski, licensed under MIT.

## Contributing

Contributions Welcome! Report bugs, request features, or improve documentation via GitHub issues or pull requests.



