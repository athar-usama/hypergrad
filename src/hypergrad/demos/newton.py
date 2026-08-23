"""Newton's method with an exact, from-scratch Hessian vs. gradient descent.

Minimizes the Rosenbrock ("banana") function

    f(x, y) = (1 - x)^2 + 100 * (y - x^2)^2

whose curved, narrow valley is the textbook example of why gradient descent
struggles with ill-conditioning: it needs thousands of tiny steps, while
Newton's method needs only a handful, because rescaling each step by the
*inverse Hessian* lets it use the curvature directly instead of guessing.
The Hessian here isn't approximated or hand-derived: it comes out of
``hypergrad.hessian``, which gets it exactly via forward-over-reverse-mode
automatic differentiation over the same computation graph used for the
gradient.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..core import Value
from ..hessian import grad, hessian
from ..viz import plot_convergence

OPTIMUM = (1.0, 1.0)


def rosenbrock(v):
    x, y = v
    return (1 - x) ** 2 + 100 * (y - x * x) ** 2


def _value_of(f, x):
    return f([Value(float(xi)) for xi in x]).data


def _dist_to_optimum(x):
    return math.hypot(x[0] - OPTIMUM[0], x[1] - OPTIMUM[1])


def gradient_descent(f, x0, lr=1.0e-3, iters=20_000):
    x = list(x0)
    history = [_dist_to_optimum(x)]
    for _ in range(iters):
        g = grad(f, x)
        x = [xi - lr * gi for xi, gi in zip(x, g, strict=True)]
        history.append(_dist_to_optimum(x))
        if history[-1] < 1e-10:
            break
    return x, history


def newton(f, x0, iters=50, damping=1e-6, tol=1e-14):
    x = list(x0)
    history = [_dist_to_optimum(x)]
    for _ in range(iters):
        g = np.array(grad(f, x))
        H = np.array(hessian(f, x)) + damping * np.eye(len(x))
        step = np.linalg.solve(H, g)

        fx = _value_of(f, x)
        alpha = 1.0
        for _ in range(60):  # backtracking line search for robustness
            candidate = [xi - alpha * si for xi, si in zip(x, step, strict=True)]
            if _value_of(f, candidate) <= fx or alpha < 1e-12:
                x = candidate
                break
            alpha *= 0.5

        history.append(_dist_to_optimum(x))
        if history[-1] < tol:
            break
    return x, history


def main(assets_dir: str | Path = "assets") -> None:
    x0 = (-1.2, 1.0)  # the standard Rosenbrock starting point

    x_gd, hist_gd = gradient_descent(rosenbrock, x0)
    x_newton, hist_newton = newton(rosenbrock, x0)

    print(f"gradient descent : {len(hist_gd) - 1:>6} iterations -> x={x_gd}")
    print(f"Newton's method  : {len(hist_newton) - 1:>6} iterations -> x={x_newton}")

    out_path = Path(assets_dir) / "newton_vs_gd.png"
    plot_convergence(hist_gd, hist_newton, out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
