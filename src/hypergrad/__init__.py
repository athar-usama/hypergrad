"""hypergrad: a from-scratch autodiff engine with exact higher-order derivatives.

    from hypergrad import Value, Dual, grad, hvp, hessian
    from hypergrad.nn import MLP

See the package README for the two things a first-order-only engine
(e.g. micrograd) can't do that this one can.
"""

from .core import Value
from .dual import Dual
from .hessian import grad, hessian, hvp

__all__ = ["Value", "Dual", "grad", "hessian", "hvp"]
__version__ = "0.1.0"
