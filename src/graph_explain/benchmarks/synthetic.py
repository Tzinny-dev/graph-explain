from __future__ import annotations

import networkx as nx
import numpy as np
import torch


def _house_motif(offset: int, anchor_in_motif: int = 3):
    edges_motif = [
        (0, 1),
        (0, 2),
        (1, 2),
        (1, 3),
        (2, 4),
        (3, 4),
    ]
    edges = [(u + offset, v + offset) for u, v in edges_motif]
    labels = {}
    for i in range(5):
        labels[i + offset] = 0
    labels[offset + anchor_in_motif] = 1
    labels[offset + 0] = 2
    for i in range(5):
        if i != anchor_in_motif and i != 0:
            labels[i + offset] = 3
    return edges, labels


def ba_shapes(
    base_nodes: int = 300,
    num_houses: int = 80,
    m: int = 5,
    seed: int = 0,
    num_features: int = 10,
    feature_style: str = "degree",
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    rng = np.random.default_rng(seed)
    g = nx.barabasi_albert_graph(base_nodes, m, seed=seed)
    g = g.to_undirected()

    node_count = base_nodes
    labels: dict[int, int] = {}

    anchors = rng.choice(base_nodes, size=num_houses, replace=False)
    for i, anchor in enumerate(anchors):
        edges_m, lab = _house_motif(node_count)
        g.add_nodes_from(range(node_count, node_count + 5))
        g.add_edges_from(edges_m)
        g.add_edge(anchor, node_count + 3)
        labels.update(lab)
        node_count += 5

    degrees = np.array([d for _, d in g.degree()], dtype=np.float64)
    max_deg = int(degrees.max()) + 1
    feat_dim = max(max_deg, num_features)
    if feature_style == "random":
        x = rng.normal(0.0, 1.0, size=(node_count, num_features)).astype(np.float32)
    else:
        x = np.zeros((node_count, feat_dim), dtype=np.float32)
        x[np.arange(node_count), np.minimum(degrees.astype(np.int64), feat_dim - 1)] = (
            1.0
        )

    edge_index = torch.tensor(np.array(g.edges(), dtype=np.int64).T, dtype=torch.long)
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    y = torch.zeros(node_count, dtype=torch.long)
    for n, l in labels.items():
        y[n] = l
    x = torch.from_numpy(x)

    house_anchors = torch.from_numpy(anchors)

    perm = rng.permutation(node_count)
    train_mask = torch.zeros(node_count, dtype=torch.bool)
    test_mask = torch.zeros(node_count, dtype=torch.bool)
    split = int(0.3 * node_count)
    train_mask[perm[:split]] = True
    test_mask[perm[split:]] = True
    train_mask[list(anchors)] = True
    test_mask[list(anchors)] = True

    return x, edge_index, y, train_mask, test_mask, house_anchors


def build_data(
    base_nodes: int = 300,
    num_houses: int = 80,
    m: int = 5,
    seed: int = 0,
    num_features: int = 10,
    feature_style: str = "degree",
):
    from torch_geometric.data import Data

    x, edge_index, y, train_mask, test_mask, house_anchors = ba_shapes(
        base_nodes=base_nodes,
        num_houses=num_houses,
        m=m,
        seed=seed,
        num_features=num_features,
        feature_style=feature_style,
    )
    return Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        test_mask=test_mask,
        house_anchors=house_anchors,
        base_nodes=base_nodes,
        num_houses=num_houses,
    )


HOUSE_SIZE = 5


def ground_truth_nodes(data, node: int) -> list[int]:
    base_nodes = int(getattr(data, "base_nodes", 0))
    if base_nodes <= 0:
        return []
    node = int(node)
    if node >= base_nodes:
        house_idx = (node - base_nodes) // HOUSE_SIZE
        return list(
            range(
                base_nodes + house_idx * HOUSE_SIZE,
                base_nodes + (house_idx + 1) * HOUSE_SIZE,
            )
        )
    anchors = data.house_anchors
    if anchors is not None and anchors.numel() and node in anchors.tolist():
        idx = int((anchors == node).nonzero(as_tuple=False).flatten()[0])
        members = list(
            range(base_nodes + idx * HOUSE_SIZE, base_nodes + (idx + 1) * HOUSE_SIZE)
        )
        return members + [node]
    return []


def ground_truth_edge_ids(data, node: int, edge_index) -> list[int]:
    import torch

    gt_nodes = set(ground_truth_nodes(data, node))
    if not gt_nodes:
        return []
    src = edge_index[0]
    dst = edge_index[1]
    both = torch.isin(src, torch.as_tensor(list(gt_nodes))) & torch.isin(
        dst, torch.as_tensor(list(gt_nodes))
    )
    if torch.is_tensor(both):
        return both.nonzero(as_tuple=False).flatten().tolist()
    return [i for i, ok in enumerate(both) if ok]
