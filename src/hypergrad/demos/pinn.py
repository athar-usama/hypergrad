"""A from-scratch Physics-Informed Neural Network (PINN).

Trains a network u(t; theta) to satisfy the damped-harmonic-oscillator ODE

    u''(t) + 2*zeta*omega0*u'(t) + omega0^2*u(t) = 0,   u(0) = 1, u'(0) = 0

by minimizing the ODE *residual* directly, rather than fitting labeled data,
the textbook PINN setup. That requires exact second derivatives of the
network's output with respect to its *input* while keeping the whole thing
differentiable with respect to the *parameters* so it can be trained. This
is the "Dual-of-Value" composition described in ``hypergrad.dual``: wrapping
the scalar input as ``Dual(Value(t0), Value(1.0))`` makes both u(t) and its
exact derivative u'(t) come out as ordinary ``Value`` nodes, still hooked
into the parameter graph. u''(t) is a central difference of two such exact
first-derivative evaluations, the one place in this example that is
numerically approximated rather than exact (see the package README for
why, and for the alternative that would make it exact too).
"""

from __future__ import annotations

import math
import random
from pathlib import Path

from ..core import Value
from ..dual import Dual
from ..nn import MLP
from ..optim import Adam
from ..viz import plot_loss_curve, plot_pinn_solution

ZETA = 0.5
OMEGA0 = 1.0
OMEGA_D = OMEGA0 * math.sqrt(1 - ZETA**2)
T_MAX = 8.0
FD_STEP = 1e-3  # step for the central-difference second derivative


def exact_solution(t: float) -> float:
    return math.exp(-ZETA * OMEGA0 * t) * (
        math.cos(OMEGA_D * t) + (ZETA * OMEGA0 / OMEGA_D) * math.sin(OMEGA_D * t)
    )


def u_and_du(net: MLP, t0: float) -> tuple[Value, Value]:
    """Exact u(t0) and u'(t0), both differentiable w.r.t. network params."""
    y = net([Dual(t0, 1.0)])
    return y.real, y.eps


def physics_residual(net: MLP, t0: float) -> Value:
    u, du = u_and_du(net, t0)
    _, du_plus = u_and_du(net, t0 + FD_STEP)
    _, du_minus = u_and_du(net, t0 - FD_STEP)
    d2u = (du_plus - du_minus) * (1.0 / (2 * FD_STEP))
    return d2u + (2 * ZETA * OMEGA0) * du + (OMEGA0**2) * u


def train(
    net: MLP,
    collocation_pts: list[float],
    *,
    ic_weight: float = 25.0,
    lr: float = 0.01,
    iters: int = 600,
) -> list[float]:
    opt = Adam(net.parameters(), lr=lr)
    losses = []
    for it in range(iters):
        opt.zero_grad()
        physics_loss = Value(0.0)
        for t0 in collocation_pts:
            r = physics_residual(net, t0)
            physics_loss = physics_loss + r * r
        physics_loss = physics_loss * (1.0 / len(collocation_pts))

        u0, du0 = u_and_du(net, 0.0)
        ic_loss = (u0 - 1.0) ** 2 + du0**2

        total = physics_loss + ic_weight * ic_loss
        total.backward()
        opt.step()

        losses.append(total.data)
        if it % 50 == 0 or it == iters - 1:
            print(f"iter {it:4d}  loss={total.data:.6e}  (physics={physics_loss.data:.3e}, ic={ic_loss.data:.3e})")
    return losses


def main(assets_dir: str | Path = "assets") -> None:
    # A pure-Python "Dual-of-Value" forward pass costs several times more than
    # plain reverse-mode autodiff (see the README), so the network and
    # collocation grid here are kept deliberately small to finish in about a
    # minute rather than demonstrate scale.
    rng = random.Random(0)
    net = MLP(1, [16, 1], nonlin="tanh", rng=rng)
    collocation = [T_MAX * i / 24 for i in range(25)]

    losses = train(net, collocation, lr=0.03, iters=500)

    ts = [T_MAX * i / 200 for i in range(201)]
    u_pred = [net([t]).data for t in ts]
    u_exact = [exact_solution(t) for t in ts]
    mse = sum((p - e) ** 2 for p, e in zip(u_pred, u_exact, strict=True)) / len(ts)
    print(f"final MSE vs. analytical solution: {mse:.6e}")

    assets = Path(assets_dir)
    assets.mkdir(exist_ok=True, parents=True)
    plot_pinn_solution(ts, u_pred, u_exact, assets / "pinn_oscillator.png")
    plot_loss_curve(losses, assets / "pinn_loss.png", title="PINN training loss (physics + IC residual)")
    print(f"wrote {assets / 'pinn_oscillator.png'}")
    print(f"wrote {assets / 'pinn_loss.png'}")


if __name__ == "__main__":
    main()
