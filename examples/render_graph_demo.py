"""Regenerates assets/computation_graph.svg for the README: the actual
live computation graph for L = tanh(a*b + c), after L.backward().
"""

from pathlib import Path

from hypergrad import Value
from hypergrad.viz import render_graph_svg

ASSETS = Path(__file__).resolve().parent.parent / "assets"


def main() -> None:
    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10.0)
    e = a * b
    d = e + c
    out = d.tanh()
    out.backward()

    labels = {id(a): "a", id(b): "b", id(c): "c", id(e): "a*b", id(d): "a*b+c", id(out): "tanh"}
    path = ASSETS / "computation_graph.svg"
    render_graph_svg(out, path, labels=labels)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
