
# ez-optimize

**Author**: Quinn Marsh\
**GitHub**: https://github.com/qthedoc/ez-optimize/ \
**PyPI**: https://pypi.org/project/ez-optimize/ 

`ez-optimize` is a more intuitive front-end for `scipy.optimize` that simplifies optimization with features like:
- keyword-based parameter definitions (e.g., `x0={'x': 1, 'y': 2}`)
- easy switching between minimization and maximization (`direction='max'`)

`ez-optimize` is your Ironman suit for optimization.

## Why ez-optimize?

### Keyword-Based Optimization (e.g.: `x0={'x': 1, 'y': 2}`)
By default, optimization uses arrays `x0=[1, 2]`. However sometimes it's more intuitive to use named parameters `x0={'x': 1, 'y': 2}`. `ez-optimize` allows you to define parameters as dictionaries. Then under the hood, `ez-optimize` automatically flattens parameters (and wraps your function) for SciPy while restoring the original structure in results. Keyword-based optimization is especially useful in physical simulations where parameters have meaningful names representing physical quantities.

### Switch to Maximize with `direction='max'`
By default, optimization minimizes the objective function. To maximize, you typically need to write a negated wrapper around your function. With `ez-optimize`, simply set `direction='max'` and the library will automatically handle negation under the hood.

## Quick Start

**Install:** 
```
pip install ez-optimize
```

**Full set of examples**: [examples.ipynb](./docs/examples.ipynb)*\
*This is currently the main form of documentation.

### Example 1: Minimizing with Keyword-Based Parameters

```python
from ez_optimize import minimize

def rosenbrock_2d(x, y, a=1, b=100):
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

### Example 2: Keyword-Based Bounds

```python

x0 = {'x': 1.3, 'y': 0.7}
bounds = {'x': (0, 2), 'y': (0, 2)}

result = minimize(rosenbrock_2d, x0, method='SLSQP', bounds=bounds)
```

### Example 3: Maximization

```python
def quadratic(x):
    return - (x - 1)**2

result = minimize(quadratic, {'x': 0.}, method='SLSQP', direction='max')

print(f"Optimal x: {result.x}")
print(f"Optimal value: {result.fun}")
```
```
Optimal x: {'x': 1.0}
Optimal value: 0.0
```

## The Array in the Room
Lets be honest, there is good reason optimization typically uses arrays and always minimizes... it makes the math simple and efficient. For example, optimizing in a vector space allows the curvature to be represented in a Hessian matrix. However, this isn't always necessary like with black-box functions that have no defined gradient or hessian. In those cases, the convenience of defining keyword-based parameters can outweigh the mathematical perfection of array-based optimization.

## Acknowledgments

Inspired by [better_optimize](https://github.com/jessegrabowski/better_optimize) by Jesse Grabowski, licensed under MIT.

## Contributing

Would love any feedback and contributions! Report bugs, request features, or improve documentation via GitHub issues or pull requests.

### Development Setup
1. Clone the repo: `git clone https://github.com/qthedoc/ez-optimize.git`
2. Navigate to the project directory: `cd ez-optimize`
3. Create a virtual environment: `python -m venv .venv`
4. Activate the virtual environment:
   - On Windows: `.\.venv\Scripts\activate`
   - On macOS/Linux: `source .venv/bin/activate`
5. Install the package in editable mode with test dependencies: `pip install -e .[test]`
