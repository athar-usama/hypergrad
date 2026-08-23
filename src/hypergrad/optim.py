"""Minimal optimizers over a list of ``Value`` parameters."""

from __future__ import annotations

from collections.abc import Sequence

from .core import Value


class SGD:
    def __init__(self, params: Sequence[Value], lr: float = 0.01):
        self.params = list(params)
        self.lr = lr

    def step(self) -> None:
        for p in self.params:
            p.data -= self.lr * p.grad

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = 0.0


class Adam:
    """Adam (Kingma & Ba, 2015) over plain-float ``Value`` gradients."""

    def __init__(
        self,
        params: Sequence[Value],
        lr: float = 0.01,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        self.params = list(params)
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.m = [0.0] * len(self.params)
        self.v = [0.0] * len(self.params)
        self.t = 0

    def step(self) -> None:
        self.t += 1
        b1, b2 = self.b1, self.b2
        bias1 = 1 - b1**self.t
        bias2 = 1 - b2**self.t
        for i, p in enumerate(self.params):
            g = p.grad
            self.m[i] = b1 * self.m[i] + (1 - b1) * g
            self.v[i] = b2 * self.v[i] + (1 - b2) * g * g
            m_hat = self.m[i] / bias1
            v_hat = self.v[i] / bias2
            p.data -= self.lr * m_hat / (v_hat**0.5 + self.eps)

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = 0.0
