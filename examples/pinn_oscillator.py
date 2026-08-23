"""Runnable entry point: ``python examples/pinn_oscillator.py``.

The actual implementation lives in ``hypergrad.demos.pinn`` so the CLI
(``hypergrad demo pinn``) and this script share the same code.
"""

from hypergrad.demos.pinn import main

if __name__ == "__main__":
    main()
