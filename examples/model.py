from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from graph_explain.benchmarks.synthetic import build_data


def build_default_model(num_features: int, num_classes: int = 4, hidden: int = 32):
    torch.manual_seed(0)
    return GCN(num_features, hidden, num_classes)


class GCN(torch.nn.Module):
    task_level = "node"

    def __init__(self, in_channels: int, hidden: int = 32, num_classes: int = 4):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden, add_self_loops=False, bias=False)
        self.conv2 = GCNConv(hidden, hidden, add_self_loops=False, bias=False)
        self.conv3 = GCNConv(hidden, num_classes, add_self_loops=False, bias=False)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        x = F.dropout(x, 0.3, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_weight=edge_weight))
        x = F.dropout(x, 0.3, training=self.training)
        return self.conv3(x, edge_index, edge_weight=edge_weight)


def train(model, data, epochs: int = 600, lr: float = 0.005) -> float:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index).argmax(dim=-1)
    return float((pred[data.test_mask] == data.y[data.test_mask]).float().mean())


def default_data_and_model(base_nodes: int = 300, num_houses: int = 40):
    data = build_data(
        base_nodes=base_nodes,
        num_houses=num_houses,
        seed=0,
        num_features=20,
        feature_style="random",
    )
    model = build_default_model(data.x.size(1))
    return data, model


class GraphGCN(torch.nn.Module):
    task_level = "graph"

    def __init__(self, in_channels: int, hidden: int = 32, num_classes: int = 2):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden, add_self_loops=False)
        self.conv2 = GCNConv(hidden, hidden, add_self_loops=False)
        self.lin = torch.nn.Linear(hidden, num_classes)

    def forward(self, x, edge_index, batch=None, edge_weight=None):
        from torch_geometric.nn import global_mean_pool

        x = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long)
        return self.lin(global_mean_pool(x, batch))


def train_graph(model, graphs, epochs: int = 300, lr: float = 0.005) -> float:
    from torch_geometric.nn import global_mean_pool

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        total = None
        for g in graphs:
            batch = torch.zeros(g.num_nodes, dtype=torch.long)
            h = F.relu(model.conv1(g.x, g.edge_index))
            h = model.conv2(h, g.edge_index)
            out = model.lin(global_mean_pool(h, batch))
            loss = F.cross_entropy(out, g.y)
            total = loss if total is None else total + loss
        total.backward()
        optimizer.step()
    model.eval()
    correct = 0
    with torch.no_grad():
        for g in graphs:
            pred = model(g.x, g.edge_index)
            correct += int(pred.argmax(dim=-1).item() == g.y.item())
    return correct / max(1, len(graphs))
