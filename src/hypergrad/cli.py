"""Command-line entry point: ``hypergrad demo {newton,pinn}``."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hypergrad")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="run a bundled example")
    demo.add_argument("name", choices=["newton", "pinn"])
    args = parser.parse_args(argv)

    if args.command == "demo":
        if args.name == "newton":
            from .demos.newton import main as run_demo
        else:
            from .demos.pinn import main as run_demo
        run_demo()
    return 0


if __name__ == "__main__":
    sys.exit(main())
