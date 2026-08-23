"""Plotting and computation-graph rendering helpers used by the examples.

The numeric plots use matplotlib. The computation-graph renderer is a small,
dependency-free layered SVG layout (no Graphviz binary required), good
enough for the toy expressions shown in the README but not intended for
huge graphs.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .core import Value


def plot_convergence(
    gd_history: list[float],
    newton_history: list[float],
    path: str | Path,
    *,
    ylabel: str = "distance to optimum",
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot([i + 1 for i in range(len(gd_history))], gd_history, label="gradient descent", linewidth=2)
    ax.plot(
        [i + 1 for i in range(len(newton_history))],
        newton_history,
        label="Newton's method (exact Hessian)",
        linewidth=2,
        marker="o",
        markersize=4,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("iteration + 1 (log scale)")
    ax.set_ylabel(ylabel + " (log scale)")
    ax.set_title("Gradient descent vs. Newton's method on the Rosenbrock function")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pinn_solution(t: list[float], u_pred: list[float], u_exact: list[float], path: str | Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(t, u_exact, label="analytical solution", linewidth=2, linestyle="--")
    ax.plot(t, u_pred, label="PINN prediction", linewidth=2)
    ax.set_xlabel("t")
    ax.set_ylabel("u(t)")
    ax.set_title("Physics-informed network vs. closed-form damped oscillator")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_loss_curve(losses: list[float], path: str | Path, *, title: str = "training loss") -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(losses, linewidth=2)
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss (log scale)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _topo_layers(root: Value) -> list[list[Value]]:
    """Assign each node a layer = its longest distance from a leaf, so
    edges always point from a lower layer to a higher one."""
    depth: dict[int, int] = {}
    nodes: dict[int, Value] = {}

    def visit(v: Value) -> int:
        if id(v) in depth:
            return depth[id(v)]
        nodes[id(v)] = v
        if not v._prev:
            depth[id(v)] = 0
        else:
            depth[id(v)] = 1 + max(visit(child) for child in v._prev)
        return depth[id(v)]

    visit(root)
    by_layer: dict[int, list[Value]] = defaultdict(list)
    for node_id, d in depth.items():
        by_layer[d].append(nodes[node_id])
    return [by_layer[d] for d in sorted(by_layer)]


def render_graph_svg(root: Value, path: str | Path, *, labels: dict[int, str] | None = None) -> None:
    """Render the computation graph rooted at ``root`` to a standalone SVG.

    Intended for small, illustrative expressions (a handful of nodes): a
    teaching diagram, not a debugger for a full network graph.
    """
    labels = labels or {}
    layers = _topo_layers(root)
    box_w, box_h = 190, 46
    x_gap, y_gap = 50, 90
    width = max(len(layer) for layer in layers) * (box_w + x_gap) + x_gap
    height = len(layers) * (box_h + y_gap) + y_gap

    pos: dict[int, tuple[float, float]] = {}
    for layer_idx, layer in enumerate(layers):
        y = height - (layer_idx * (box_h + y_gap) + y_gap)
        row_w = len(layer) * (box_w + x_gap) - x_gap
        x0 = (width - row_w) / 2
        for i, node in enumerate(layer):
            pos[id(node)] = (x0 + i * (box_w + x_gap), y)

    def fmt(v: Value) -> str:
        d = v.data
        try:
            return f"{float(d):.4g}"
        except (TypeError, ValueError):
            return str(d)

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace" font-size="13">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
    ]

    edges = []
    for layer in layers:
        for node in layer:
            x, y = pos[id(node)]
            for child in node._prev:
                cx, cy = pos[id(child)]
                edges.append((cx + box_w / 2, cy, x + box_w / 2, y + box_h))
    for x1, y1, x2, y2 in edges:
        svg.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#888" stroke-width="1.5"/>')

    def fmt_grad(v: Value) -> str:
        g = v.grad
        try:
            return f"{float(g):.4g}"
        except (TypeError, ValueError):
            return "n/a"

    for layer in layers:
        for node in layer:
            x, y = pos[id(node)]
            label = labels.get(id(node), node._op or "leaf")
            svg.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{box_w}" height="{box_h}" rx="8" '
                       f'fill="#eef3ff" stroke="#3355aa" stroke-width="1.5"/>')
            svg.append(f'<text x="{x + box_w/2:.1f}" y="{y + 18:.1f}" text-anchor="middle" fill="#111" '
                       f'font-size="13">{label}</text>')
            svg.append(f'<text x="{x + box_w/2:.1f}" y="{y + 36:.1f}" text-anchor="middle" fill="#555" '
                       f'font-size="11">data={fmt(node)} grad={fmt_grad(node)}</text>')

    svg.append("</svg>")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(svg), encoding="utf-8")
