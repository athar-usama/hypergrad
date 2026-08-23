"""Gradients, Hessian-vector products, and full Hessians.

The key trick lives in ``hvp``: seed every leaf with a ``Dual`` whose tangent
component is the direction ``v``, run an *ordinary* reverse-mode backward
pass, and read off the tangent component of each leaf's gradient. That's
forward-over-reverse mode, and it costs one backward pass per Hessian-vector
product regardless of the number of parameters ``n``, versus ``n`` calls to
``grad`` (or ``2n`` function evaluations for a finite-difference Hessian).
``hessian`` just calls ``hvp`` once per basis vector to materialize the full
matrix, which is the cheap way to get one when you want to inspect it (as the
Newton's-method example does), even though for pure optimization you'd only
ever need ``hvp``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from .core import Value
from .dual import Dual

Vector = Sequence[float]
ScalarFn = Callable[[Sequence[Value]], Value]


def grad(f: ScalarFn, x: Vector) -> list[float]:
    """Ordinary reverse-mode gradient of a scalar function ``f`` at ``x``."""
    leaves = [Value(float(xi)) for xi in x]
    f(leaves).backward()
    return [leaf.grad for leaf in leaves]


def hvp(f: ScalarFn, x: Vector, v: Vector) -> list[float]:
    """Hessian-vector product ``H(x) @ v`` via forward-over-reverse mode."""
    if len(x) != len(v):
        raise ValueError("x and v must have the same length")
    leaves = [Value(Dual(float(xi), float(vi))) for xi, vi in zip(x, v, strict=True)]
    f(leaves).backward()
    return [leaf.grad.eps for leaf in leaves]


def hessian(f: ScalarFn, x: Vector) -> list[list[float]]:
    """Full (symmetric) Hessian of ``f`` at ``x``, one ``hvp`` per column."""
    n = len(x)
    columns = []
    for i in range(n):
        e_i = [1.0 if j == i else 0.0 for j in range(n)]
        columns.append(hvp(f, x, e_i))
    # columns[j][i] = d^2f / dx_i dx_j ; transpose so H[i][j] = d^2f/dx_i dx_j
    return [[columns[j][i] for j in range(n)] for i in range(n)]
