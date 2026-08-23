import math

import pytest

from hypergrad import Value


def fd_grad(f, xs, h=1e-6):
    grads = []
    for i in range(len(xs)):
        xp = list(xs)
        xp[i] += h
        xm = list(xs)
        xm[i] -= h
        grads.append((f(xp) - f(xm)) / (2 * h))
    return grads


def value_grad(expr, xs):
    vals = [Value(x) for x in xs]
    expr(vals).backward()
    return [v.grad for v in vals]


def as_float_fn(expr):
    def f(xs):
        return expr([Value(x) for x in xs]).data

    return f


EXPRESSIONS = [
    lambda v: v[0] + v[1],
    lambda v: v[0] - v[1],
    lambda v: v[0] * v[1],
    lambda v: v[0] / v[1],
    lambda v: v[0] ** 3,
    lambda v: v[0].exp(),
    lambda v: v[0].log(),
    lambda v: v[0].tanh(),
    lambda v: v[0].sigmoid(),
    lambda v: (v[0] * v[1] + v[0].tanh()).exp() / (v[1] + 2.0),
]


@pytest.mark.parametrize("expr", EXPRESSIONS)
def test_gradient_matches_finite_difference(expr):
    xs = [0.7, 1.3]
    analytic = value_grad(expr, xs)
    numeric = fd_grad(as_float_fn(expr), xs)
    for a, n in zip(analytic, numeric):
        assert math.isclose(a, n, rel_tol=1e-4, abs_tol=1e-5)


def test_backward_accumulates_over_shared_subexpressions():
    a = Value(3.0)
    b = a * a + a  # d/da = 2a + 1
    b.backward()
    assert math.isclose(a.grad, 2 * 3.0 + 1)


def test_relu_matches_expected_subgradient():
    a, b = Value(2.0), Value(-3.0)
    (a.relu() + b.relu()).backward()
    assert a.grad == 1.0
    assert b.grad == 0.0


def test_foreign_type_arithmetic_falls_back_correctly():
    """Value ops must return NotImplemented for non-Value/number operands so
    that e.g. Dual's reflected methods get a chance to run (see dual.py)."""
    from hypergrad.dual import Dual

    v = Value(2.0)
    d = Dual(3.0, 1.0)
    result = v * d  # dispatches to Dual.__rmul__, real/eps come back as Values
    assert isinstance(result, Dual)
    assert isinstance(result.real, Value)
    assert math.isclose(result.real.data, 6.0)
