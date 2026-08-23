"""End-to-end correctness check for the PINN's "Dual-of-Value" composition.

This is the test that matters most for the PINN example: it checks that
``physics_residual(...).backward()`` (reverse-mode through a graph built out
of a forward-mode Dual-of-Value evaluation) produces the *same* gradient
w.r.t. every network parameter as directly finite-differencing the forward
computation. The finite-difference side never touches ``.backward()`` or
``.grad`` at all, it only perturbs ``Value.data`` and re-runs the forward
pass, so this is an independent check rather than a circular one.
"""

import math
import random

from hypergrad.demos.pinn import physics_residual
from hypergrad.nn import MLP


def test_physics_residual_gradient_matches_finite_difference():
    rng = random.Random(42)
    net = MLP(1, [4, 4, 1], nonlin="tanh", rng=rng)
    t0 = 1.3

    for p in net.parameters():
        p.grad = 0.0
    loss = physics_residual(net, t0) ** 2
    loss.backward()

    h = 1e-5
    checked = 0
    for p in net.parameters():
        original = p.data
        analytic_grad = p.grad

        p.data = original + h
        loss_plus = physics_residual(net, t0).data ** 2
        p.data = original - h
        loss_minus = physics_residual(net, t0).data ** 2
        p.data = original

        numeric_grad = (loss_plus - loss_minus) / (2 * h)
        assert math.isclose(analytic_grad, numeric_grad, rel_tol=1e-2, abs_tol=1e-3), (
            f"analytic={analytic_grad} numeric={numeric_grad}"
        )
        checked += 1

    assert checked > 0


def test_u_and_du_are_exact_against_finite_difference():
    from hypergrad.demos.pinn import u_and_du

    rng = random.Random(7)
    net = MLP(1, [4, 4, 1], nonlin="tanh", rng=rng)
    t0 = 0.85
    h = 1e-6

    _, du = u_and_du(net, t0)
    u_plus, _ = u_and_du(net, t0 + h)
    u_minus, _ = u_and_du(net, t0 - h)
    numeric_du = (u_plus.data - u_minus.data) / (2 * h)

    assert math.isclose(du.data, numeric_du, rel_tol=1e-3, abs_tol=1e-4)
