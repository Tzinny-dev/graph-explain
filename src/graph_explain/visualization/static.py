from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import networkx as nx

from ..core.explanation import Explanation


def visualize_static(
    explanation: Explanation,
    threshold: float | None = None,
    show_labels: bool = False,
    node_size: int = 400,
    title: str | None = None,
    ax: Any | None = None,
    seed: int = 42,
    cmap: str = "YlOrRd",
) -> Any:
    threshold = threshold if threshold is not None else explanation.mask_threshold
    G = explanation.to_networkx(threshold=threshold)
    if explanation.node_idx is not None and explanation.node_idx in G.nodes():
        target = int(explanation.node_idx)
    else:
        target = None

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    pos = nx.spring_layout(G, seed=seed)
    edge_weights = {}
    for u, v, w in G.edges(data="weight", default=0.0):
        edge_weights[(u, v)] = float(w)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#888", alpha=0.6)

    if target is not None:
        others = [n for n in G.nodes() if n != target]
        if others:
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=others,
                node_size=node_size,
                node_color="#aaddff",
                ax=ax,
                node_shape="o",
            )
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=[target],
            node_size=node_size * 1.4,
            node_color="#d62728",
            ax=ax,
        )
    else:
        nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color="#aaddff", ax=ax)

    if explanation.edge_importance is not None:
        vals = list(edge_weights.values())
        if vals:
            vmin, vmax = min(vals), max(vals)
            span = vmax - vmin or 1.0
            cmap_obj = plt.colormaps[cmap]
            for (u, v), w in edge_weights.items():
                t = (w - vmin) / span
                width = 0.5 + 4.0 * t
                nx.draw_networkx_edges(
                    G,
                    pos,
                    edgelist=[(u, v)],
                    width=width,
                    edge_color=cmap_obj(t),
                    ax=ax,
                    alpha=0.9,
                )

    if show_labels:
        nx.draw_networkx_labels(G, pos, ax=ax)

    ax.set_axis_off()
    if title:
        ax.set_title(title)
    return ax


def show(explanation: Explanation, **kwargs) -> None:
    visualize_static(explanation, **kwargs)
    plt.show()
