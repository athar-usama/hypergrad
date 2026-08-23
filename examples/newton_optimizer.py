"""Runnable entry point: ``python examples/newton_optimizer.py``.

The actual implementation lives in ``hypergrad.demos.newton`` so the CLI
(``hypergrad demo newton``) and this script share the same code.
"""

from hypergrad.demos.newton import main

if __name__ == "__main__":
    main()
