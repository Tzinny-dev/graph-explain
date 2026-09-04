from __future__ import annotations

from ..core.explanation import Explanation


def visualize_interactive(
    explanation: Explanation,
    output_path: str = "explanation.html",
    threshold: float | None = None,
    show_labels: bool = True,
    title: str | None = None,
    node_size_scale: float = 120.0,
) -> str:
    try:
        from pyvis.network import Network
    except ImportError:
        raise ImportError(
            "visualize_interactive requiere pyvis: pip install pyvis (extras: [interactive])."
        )

    threshold = threshold if threshold is not None else explanation.mask_threshold
    G = explanation.to_networkx(threshold=threshold)

    net = Network(
        height="700px",
        width="100%",
        directed=False,
        notebook=False,
        heading=(title or "Graph Explain"),
        select_menu=True,
        filter_menu=False,
    )

    edge_weights = {
        (u, v): float(w) for u, v, w in G.edges(data="weight", default=0.0)
    }
    max_w = max(edge_weights.values(), default=0.0) or 1.0

    node_imp = {}
    if explanation.node_importance is not None:
        for n in G.nodes():
            node_imp[int(n)] = float(explanation.node_importance[int(n)])

    for n in G.nodes():
        label = str(n) if show_labels else ""
        if explanation.node_idx is not None and n == int(explanation.node_idx):
            color, size = "#d62728", 1.6 * node_size_scale
            title_attr = f"<b>nodo objetivo {n}</b>"
        else:
            imp = node_imp.get(int(n), 0.0)
            scale = 0.5 + 1.8 * imp
            color = _importance_color(imp)
            size = scale * node_size_scale
            title_attr = f"nodo {n}<br>importancia: {imp:.3f}"
        net.add_node(int(n), label=label, color=color, size=size, title=title_attr)

    for (u, v), w in edge_weights.items():
        width = 1 + 8 * (w / max_w)
        color = f"rgba(200, 40, 40, {0.3 + 0.7 * w / max_w})"
        net.add_edge(int(u), int(v), value=w, width=width, color=color, title=f"peso: {w:.3f}")

    net.write_html(output_path, open_browser=False, notebook=False)
    return output_path


def _importance_color(value: float) -> str:
    lo = "#dcedc1"
    hi = "#d62728"
    t = max(0.0, min(1.0, value))
    r = int(float(int(lo[1:3], 16)) + t * (int(hi[1:3], 16) - int(lo[1:3], 16)))
    g = int(float(int(lo[3:5], 16)) + t * (int(hi[3:5], 16) - int(lo[3:5], 16)))
    b = int(float(int(lo[5:7], 16)) + t * (int(hi[5:7], 16) - int(lo[5:7], 16)))
    return f"#{(r << 16) | (g << 8) | b:06x}"