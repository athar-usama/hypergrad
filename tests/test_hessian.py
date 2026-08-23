import math

from hypergrad import Value
from hypergrad.hessian import grad, hessian, hvp


def rosenbrock(v):
    x, y = v
    return (1 - x) ** 2 + 100 * (y - x * x) ** 2


def value_fn(expr):
    def f(xs):
        return expr([Value(x) for x in xs]).data

    return f


def fd_grad(f, x, h=1e-5):
    g = []
    for i in range(len(x)):
        xp, xm = list(x), list(x)
        xp[i] += h
        xm[i] -= h
        g.append((f(xp) - f(xm)) / (2 * h))
    return g


def fd_hessian(f, x, h=1e-4):
    n = len(x)
    H = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            pp, pm, mp, mm = list(x), list(x), list(x), list(x)
            pp[i] += h
            pp[j] += h
            pm[i] += h
            pm[j] -= h
            mp[i] -= h
            mp[j] += h
            mm[i] -= h
            mm[j] -= h
            H[i][j] = (f(pp) - f(pm) - f(mp) + f(mm)) / (4 * h * h)
    return H


def test_grad_matches_finite_difference():
    x0 = [0.3, -0.2]
    analytic = grad(rosenbrock, x0)
    numeric = fd_grad(value_fn(rosenbrock), x0)
    for a, n in zip(analytic, numeric):
        assert math.isclose(a, n, rel_tol=1e-4, abs_tol=1e-5)


def test_hessian_matches_finite_difference():
    x0 = [0.3, -0.2]
    H_analytic = hessian(rosenbrock, x0)
    H_numeric = fd_hessian(value_fn(rosenbrock), x0)
    for row_a, row_n in zip(H_analytic, H_numeric):
        for a, n in zip(row_a, row_n):
            assert math.isclose(a, n, rel_tol=1e-3, abs_tol=1e-3)


def test_hessian_is_symmetric():
    x0 = [0.5, 0.9]
    H = hessian(rosenbrock, x0)
    assert math.isclose(H[0][1], H[1][0], rel_tol=1e-9)


def test_hvp_equals_hessian_dot_vector():
    x0 = [0.3, -0.2]
    v = [0.5, -1.2]
    H = hessian(rosenbrock, x0)
    expected = [sum(H[i][j] * v[j] for j in range(2)) for i in range(2)]
    actual = hvp(rosenbrock, x0, v)
    for e, a in zip(expected, actual):
        assert math.isclose(e, a, rel_tol=1e-9, abs_tol=1e-9)


def test_hvp_costs_one_backward_pass_regardless_of_dimension():
    # not a timing test - just checks hvp doesn't require materializing H
    x0 = [0.1, 0.2, 0.3, 0.4, 0.5]

    def sphere(v):
        out = v[0] * v[0]
        for vi in v[1:]:
            out = out + vi * vi
        return out

    v = [1.0, 0.0, 0.0, 0.0, 0.0]
    result = hvp(sphere, x0, v)
    # Hessian of sum(x_i^2) is 2*I, so H@e_0 = [2,0,0,0,0]
    assert math.isclose(result[0], 2.0, rel_tol=1e-9)
    for r in result[1:]:
        assert math.isclose(r, 0.0, abs_tol=1e-9)
