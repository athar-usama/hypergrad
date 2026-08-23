"""A tiny multi-layer perceptron built on top of ``Value``.

Deliberately minimal, since this is not the point of the package: it exists
so the Hessian machinery and the PINN example have a trainable function to
work with. Activations default to ``tanh`` because it is smooth to all orders,
which matters for the PINN example (a ``relu`` network's output has a zero
second derivative almost everywhere, which would make the physics loss
meaningless).
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from .core import Value


class Module:
    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self) -> list[Value]:
        return []


class Neuron(Module):
    def __init__(self, n_in: int, nonlin: str | None = "tanh", *, rng: random.Random | None = None):
        rng = rng or random
        self.w = [Value(rng.uniform(-1, 1)) for _ in range(n_in)]
        self.b = Value(0.0)
        self.nonlin = nonlin

    def __call__(self, x: Sequence):
        act = self.b
        for wi, xi in zip(self.w, x, strict=True):
            act = act + wi * xi
        if self.nonlin == "tanh":
            return act.tanh()
        if self.nonlin == "relu":
            return act.relu()
        if self.nonlin == "sigmoid":
            return act.sigmoid()
        return act

    def parameters(self) -> list[Value]:
        return [*self.w, self.b]


class Layer(Module):
    def __init__(self, n_in: int, n_out: int, **kwargs):
        self.neurons = [Neuron(n_in, **kwargs) for _ in range(n_out)]

    def __call__(self, x: Sequence):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self) -> list[Value]:
        return [p for n in self.neurons for p in n.parameters()]


class MLP(Module):
    """``MLP(n_in, [h1, h2, ..., n_out])``: hidden layers use ``nonlin``,
    the final layer is linear (so outputs are unbounded), matching the usual
    convention for a regression network."""

    def __init__(self, n_in: int, n_outs: Sequence[int], nonlin: str = "tanh", *, rng: random.Random | None = None):
        sizes = [n_in, *n_outs]
        self.layers = [
            Layer(sizes[i], sizes[i + 1], nonlin=(nonlin if i < len(n_outs) - 1 else None), rng=rng)
            for i in range(len(n_outs))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> list[Value]:
        return [p for layer in self.layers for p in layer.parameters()]
