"""Forward-mode automatic differentiation via dual numbers.

A dual number ``a + b*eps`` (with ``eps**2 == 0``) carries a value ``a`` and
its exact derivative ``b`` along some direction through every arithmetic
operation, by construction (the ``eps**2 = 0`` truncation rule *is* the chain
rule). ``Dual`` implements that algebra generically. ``real``/``eps`` can be
plain floats, ``Value`` objects from ``core.py`` (the whole point of this
package), or even another ``Dual``: nesting ``Dual`` inside itself gives
exact second derivatives for free, see the test suite for a worked check.

Two ways this composes with ``Value``, both used elsewhere in this package:

- **Value-of-Dual** (``hessian.py``): every leaf is a ``Value`` whose
  ``.data`` is a ``Dual``. An ordinary reverse-mode backward pass then
  produces a *dual-valued* gradient, whose tangent component is a
  Hessian-vector product computed via forward-over-reverse mode.
- **Dual-of-Value** (the PINN example): a single input variable is wrapped
  as ``Dual(Value(t0), Value(1.0))`` while parameters stay plain ``Value``
  leaves. Running a network on it produces a ``Dual`` whose ``.real`` and
  ``.eps`` are themselves ``Value`` nodes, so both the function value and
  its exact derivative w.r.t. the input remain differentiable w.r.t. the
  parameters.

Like ``Value``, arithmetic here only auto-promotes things it recognizes:
plain numbers and ``Value`` instances, nothing with dual-number semantics
of its own. It never needs to promote a *foreign* AD type, because a
non-numeric "other" operand is simply wrapped with a zero tangent, which
is the correct behaviour whether that operand is a float or a Value.
"""

from __future__ import annotations

from ._generic import gexp, glog, gtanh


class Dual:
    __slots__ = ("real", "eps")

    def __init__(self, real, eps=0.0):
        self.real = real
        self.eps = eps

    @staticmethod
    def _c(x) -> Dual:
        return x if isinstance(x, Dual) else Dual(x, 0.0)

    # -- arithmetic -------------------------------------------------------
    def __add__(self, other):
        other = Dual._c(other)
        return Dual(self.real + other.real, self.eps + other.eps)

    __radd__ = __add__

    def __neg__(self):
        return Dual(-self.real, -self.eps)

    def __sub__(self, other):
        other = Dual._c(other)
        return Dual(self.real - other.real, self.eps - other.eps)

    def __rsub__(self, other):
        other = Dual._c(other)
        return other - self

    def __mul__(self, other):
        other = Dual._c(other)
        return Dual(self.real * other.real, self.eps * other.real + self.real * other.eps)

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = Dual._c(other)
        return Dual(
            self.real / other.real,
            (self.eps * other.real - self.real * other.eps) / (other.real * other.real),
        )

    def __rtruediv__(self, other):
        other = Dual._c(other)
        return other / self

    def __pow__(self, power):
        if not isinstance(power, (int, float)):
            return NotImplemented
        return Dual(self.real**power, power * self.real ** (power - 1) * self.eps)

    # -- elementary functions --------------------------------------------
    def exp(self):
        e = gexp(self.real)
        return Dual(e, e * self.eps)

    def log(self):
        return Dual(glog(self.real), self.eps / self.real)

    def tanh(self):
        t = gtanh(self.real)
        return Dual(t, (1 - t * t) * self.eps)

    def __repr__(self):
        return f"Dual(real={self.real!r}, eps={self.eps!r})"
