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


def build_graph_classification(
    num_pos: int = 20,
    num_neg: int = 20,
    base_nodes_range: tuple[int, int] = (15, 30),
    m: int = 2,
    seed: int = 0,
    num_features: int = 8,
    feature_style: str = "random",
):
    """Graph classification dataset with a known 'house' motif.

    Returns a list of graph-level `Data` with a binary label `y` (1 if the graph
    contains the house motif). Each graph carries `gt_edge_mask` (bool over the
    directed edges, both directions included) and `gt_nodes` with the motif
    nodes (+ hub).
    """
    from torch_geometric.data import Data

    rng = np.random.default_rng(seed)
    graphs: list[Data] = []
    for cls in (1, 0):
        count = num_pos if cls == 1 else num_neg
        for _ in range(count):
            base = int(rng.integers(base_nodes_range[0], base_nodes_range[1] + 1))
            tree_seed = int(rng.integers(0, 2**31 - 1))
            g = nx.barabasi_albert_graph(base, m, seed=tree_seed)
            g = g.to_undirected()

            gt_nodes: list[int] = []
            gt_edges: list[tuple[int, int]] = []
            if cls == 1:
                offset = g.number_of_nodes()
                anchor = int(rng.integers(0, base))
                edges_m, _ = _house_motif(offset)
                g.add_nodes_from(range(offset, offset + 5))
                g.add_edges_from(edges_m)
                g.add_edge(anchor, offset + 3)
                gt_nodes = list(range(offset, offset + 5)) + [anchor]
                gt_edges = list(edges_m) + [(anchor, offset + 3)]

            if feature_style == "degree":
                degrees = np.array([d for _, d in g.degree()], dtype=np.float64)
                feat_dim = max(int(degrees.max()) + 1, num_features)
                x = np.zeros((g.number_of_nodes(), feat_dim), dtype=np.float32)
                x[
                    np.arange(g.number_of_nodes()),
                    np.minimum(degrees.astype(np.int64), feat_dim - 1),
                ] = 1.0
            else:
                x = rng.normal(
                    0.0,
                    1.0,
                    size=(g.number_of_nodes(), num_features),
                ).astype(np.float32)

            edge_index = torch.tensor(
                np.array(g.edges(), dtype=np.int64).T, dtype=torch.long
            )
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

            gt_pairs = set(gt_edges) | {(v, u) for u, v in gt_edges}
            gt_edge_mask = torch.zeros(edge_index.size(1), dtype=torch.bool)
            if cls == 1:
                for i, (s, d) in enumerate(
                    zip(edge_index[0].tolist(), edge_index[1].tolist())
                ):
                    if (s, d) in gt_pairs:
                        gt_edge_mask[i] = True

            graphs.append(
                Data(
                    x=torch.from_numpy(x),
                    edge_index=edge_index,
                    y=torch.tensor([cls], dtype=torch.long),
                    num_nodes=g.number_of_nodes(),
                    gt_edge_mask=gt_edge_mask,
                    gt_nodes=sorted(gt_nodes),
                )
            )
    return graphs


def ground_truth_edges_graph(data) -> list[int]:
    """(Directed) indices of the motif edges for a graph of the dataset."""
    mask = getattr(data, "gt_edge_mask", None)
    if mask is None:
        return []
    return mask.nonzero(as_tuple=False).flatten().tolist()


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
