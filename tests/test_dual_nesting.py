"""Dual is generic over what it wraps, so nesting it inside itself gives
exact second derivatives "for free" via forward-over-forward mode: see the
module docstring in ``hypergrad/dual.py``. These tests check that trick in
isolation (no Value/reverse-mode involved) against finite differences.
"""

import math

from hypergrad._generic import gexp, gtanh
from hypergrad.dual import Dual


def second_derivative_via_nested_dual(f, a):
    x = Dual(Dual(a, 1.0), Dual(1.0, 0.0))
    y = f(x)
    assert math.isclose(y.real.eps, y.eps.real, rel_tol=1e-9), "the two first-derivative slots must agree"
    return y.eps.eps


def first_derivative_via_dual(f, a):
    return f(Dual(a, 1.0)).eps


def fd_first_derivative(f_plain, a, h=1e-6):
    return (f_plain(a + h) - f_plain(a - h)) / (2 * h)


def fd_second_derivative(f_plain, a, h=1e-3):
    return (f_plain(a + h) - 2 * f_plain(a) + f_plain(a - h)) / (h * h)


def test_square_first_and_second_derivative_are_exact():
    f = lambda x: x * x
    a = 3.0
    assert math.isclose(first_derivative_via_dual(f, a), 2 * a, rel_tol=1e-9)
    assert math.isclose(second_derivative_via_nested_dual(f, a), 2.0, rel_tol=1e-9)


def test_cube_second_derivative_is_exact():
    f = lambda x: x**3
    a = 2.5
    assert math.isclose(second_derivative_via_nested_dual(f, a), 6 * a, rel_tol=1e-9)


def test_tanh_second_derivative_matches_finite_difference():
    a = 0.6
    analytic = second_derivative_via_nested_dual(gtanh, a)
    numeric = fd_second_derivative(math.tanh, a)
    assert math.isclose(analytic, numeric, rel_tol=1e-3, abs_tol=1e-4)


def test_exp_times_x_second_derivative_matches_finite_difference():
    f = lambda x: gexp(x) * x
    f_plain = lambda t: math.exp(t) * t
    a = 0.4
    analytic = second_derivative_via_nested_dual(f, a)
    numeric = fd_second_derivative(f_plain, a)
    assert math.isclose(analytic, numeric, rel_tol=1e-3, abs_tol=1e-4)
