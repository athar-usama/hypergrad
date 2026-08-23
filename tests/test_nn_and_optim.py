import random

from hypergrad import Value
from hypergrad.nn import MLP
from hypergrad.optim import Adam


def test_mlp_overfits_xor():
    rng = random.Random(0)
    net = MLP(2, [8, 1], nonlin="tanh", rng=rng)
    xs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    ys = [0.0, 1.0, 1.0, 0.0]

    opt = Adam(net.parameters(), lr=0.05)
    loss = None
    for _ in range(400):
        opt.zero_grad()
        loss = Value(0.0)
        for x, y in zip(xs, ys):
            pred = net([Value(xi) for xi in x])
            loss = loss + (pred - y) ** 2
        loss = loss * (1.0 / len(xs))
        loss.backward()
        opt.step()

    assert loss.data < 0.05


def test_pinn_training_reduces_loss():
    from hypergrad.demos.pinn import train

    rng = random.Random(1)
    net = MLP(1, [8, 8, 1], nonlin="tanh", rng=rng)
    collocation = [0.2 * i for i in range(6)]
    losses = train(net, collocation, iters=80, lr=0.02)
    assert losses[-1] < losses[0]
