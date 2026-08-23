"""Reverse-mode automatic differentiation over an arbitrary scalar ring.

``Value`` is a computation-graph node in the style of Karpathy's ``micrograd``:
it records the operation that produced it and a ``_backward`` closure that
knows how to push a gradient onto its parents. Calling ``.backward()`` on the
output of a computation does a topological sort and runs those closures in
reverse order.

The one deliberate difference from a typical from-scratch autograd: nothing
here assumes ``self.data`` is a ``float``. Every op is written in terms of
``+``, ``-``, ``*``, ``/``, ``**`` and the ``_generic`` dispatch helpers, so
``data`` (and therefore ``grad``) can just as well be a ``hypergrad.dual.Dual``
number. That's what makes exact Hessians (``hessian.py``) possible: seed the
leaves with dual numbers and the *gradient* that comes out of an ordinary
backward pass is itself a dual number, whose tangent component is a
Hessian-vector product. See ``hessian.py`` for the seeding logic.

Operators only auto-promote plain ``int``/``float`` operands. Anything else
(e.g. a ``Dual``) makes the operator return ``NotImplemented`` so Python
falls back to the other operand's reflected method, which is what lets a
``Value`` (weights) and a ``Dual`` (an input variable) meet in either order
inside the same expression, as used in the PINN example.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from ._generic import gexp, glog, gtanh


class Value:
    __slots__ = ("data", "grad", "_backward", "_prev", "_op")

    def __init__(self, data, _children: Iterable[Value] = (), _op: str = ""):
        self.data = data
        self.grad = 0.0
        self._backward: Callable[[], None] = lambda: None
        self._prev = set(_children)
        self._op = _op

    # -- coercion -----------------------------------------------------
    @staticmethod
    def _ensure(x):
        """Wrap plain numbers as constants; refuse to touch foreign types."""
        if isinstance(x, Value):
            return x
        if isinstance(x, (int, float)):
            return Value(x)
        return NotImplemented

    # -- arithmetic -----------------------------------------------------
    def __add__(self, other):
        other = Value._ensure(other)
        if other is NotImplemented:
            return NotImplemented
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad = self.grad + out.grad
            other.grad = other.grad + out.grad

        out._backward = _backward
        return out

    __radd__ = __add__

    def __neg__(self):
        out = Value(-self.data, (self,), "neg")

        def _backward():
            self.grad = self.grad - out.grad

        out._backward = _backward
        return out

    def __sub__(self, other):
        other = Value._ensure(other)
        if other is NotImplemented:
            return NotImplemented
        return self + (-other)

    def __rsub__(self, other):
        other = Value._ensure(other)
        if other is NotImplemented:
            return NotImplemented
        return other + (-self)

    def __mul__(self, other):
        other = Value._ensure(other)
        if other is NotImplemented:
            return NotImplemented
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad = self.grad + other.data * out.grad
            other.grad = other.grad + self.data * out.grad

        out._backward = _backward
        return out

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = Value._ensure(other)
        if other is NotImplemented:
            return NotImplemented
        return self * other**-1

    def __rtruediv__(self, other):
        other = Value._ensure(other)
        if other is NotImplemented:
            return NotImplemented
        return other * self**-1

    def __pow__(self, power):
        if not isinstance(power, (int, float)):
            return NotImplemented
        out = Value(self.data**power, (self,), f"**{power}")

        def _backward():
            self.grad = self.grad + (power * self.data ** (power - 1)) * out.grad

        out._backward = _backward
        return out

    # -- elementary functions --------------------------------------------
    def exp(self):
        e = gexp(self.data)
        out = Value(e, (self,), "exp")

        def _backward():
            self.grad = self.grad + out.data * out.grad

        out._backward = _backward
        return out

    def log(self):
        out = Value(glog(self.data), (self,), "log")

        def _backward():
            self.grad = self.grad + out.grad / self.data

        out._backward = _backward
        return out

    def tanh(self):
        t = gtanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad = self.grad + (1 - out.data * out.data) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        # NOTE: relu is not twice-differentiable, so it is unsuitable for the
        # Hessian/PINN demos in this package (which need smooth activations).
        # It's provided for completeness and assumes plain-float data.
        out = Value(self.data if self.data > 0 else 0.0, (self,), "relu")

        def _backward():
            self.grad = self.grad + (out.grad if out.data > 0 else 0.0)

        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1 / (1 + gexp(-self.data))
        out = Value(s, (self,), "sigmoid")

        def _backward():
            self.grad = self.grad + out.data * (1 - out.data) * out.grad

        out._backward = _backward
        return out

    # -- graph traversal --------------------------------------------------
    def backward(self):
        topo: list[Value] = []
        visited = set()

        def build(v: Value):
            if id(v) not in visited:
                visited.add(id(v))
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    def __repr__(self):
        return f"Value(data={self.data!r}, grad={self.grad!r})"
