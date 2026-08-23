"""Generic elementary-function dispatch.

``Value`` (reverse-mode) and ``Dual`` (forward-mode) are both written to be
agnostic to *what* number-like type they wrap: a plain ``float``, a ``Dual``,
a ``Value``, or a nesting of either inside the other. That's what lets the
two modes compose in whichever order a problem needs (``Value``-of-``Dual``
for exact Hessians in ``hessian.py``, ``Dual``-of-``Value`` for derivatives
through a trainable graph in the PINN example) without either class knowing
the other exists.

The only place that generality has to be spelled out is the elementary
functions (exp/log/tanh), since ``math.exp`` only understands floats. These
three helpers dispatch to a wrapped type's own method when it has one, and
fall back to ``math`` otherwise.
"""

import math


def gexp(x):
    return x.exp() if hasattr(x, "exp") else math.exp(x)


def glog(x):
    return x.log() if hasattr(x, "log") else math.log(x)


def gtanh(x):
    return x.tanh() if hasattr(x, "tanh") else math.tanh(x)
