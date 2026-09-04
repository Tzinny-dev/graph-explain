from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import pytest

dgl = pytest.importorskip("dgl", exc_type=Exception)

import torch
from dgl.nn import GraphConv

from graph_explain import Explainer, GNNExplainer, get_backend
from graph_explain.benchmarks.synthetic import build_data


class DGLGCN(torch.nn.Module):
    def __init__(self, in_feats: int, hidden: int = 16, out: int = 4):
        super().__init__()
        self.conv1 = GraphConv(in_feats, hidden)
        self.conv2 = GraphConv(hidden, out)

    def forward(self, g, feat):
        w = g.edata.get("w")
        if w is not None:
            h = torch.relu(self.conv1(g, feat, edge_weight=w))
            return self.conv2(g, h, edge_weight=w)
        h = torch.relu(self.conv1(g, feat))
        return self.conv2(g, h)


@pytest.fixture(scope="module")
def dgl_graph_data():
    torch.manual_seed(0)
    base = build_data(base_nodes=60, num_houses=10, m=3, seed=0)
    g = dgl.graph(
        (base.edge_index[0], base.edge_index[1]),
        num_nodes=base.num_nodes,
    )
    g.ndata["feat"] = base.x
    g.ndata["label"] = base.y
    g.ndata["train_mask"] = base.train_mask
    g.ndata["test_mask"] = base.test_mask
    return g


def _train(model, g, epochs: int = 150, lr: float = 0.01) -> float:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        out = model(g, g.ndata["feat"])
        loss = torch.nn.functional.cross_entropy(
            out[g.ndata["train_mask"]], g.ndata["label"][g.ndata["train_mask"]]
        )
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        out = model(g, g.ndata["feat"])
    pred = out[g.ndata["test_mask"]].argmax(-1)
    acc = (pred == g.ndata["label"][g.ndata["test_mask"]]).float().mean().item()
    return acc


def test_dgl_adapter_forward_and_prediction(dgl_graph_data):
    g = dgl_graph_data
    backend = get_backend("dgl")
    assert backend.name == "dgl"
    assert backend.num_nodes(g) == g.num_nodes()
    uu, vv = g.edges()
    assert torch.equal(backend.edge_index(g)[0], uu)
    assert torch.equal(backend.edge_index(g)[1], vv)

    model = DGLGCN(g.ndata["feat"].shape[1])
    acc = _train(model, g)
    assert acc > 0.3, f"el modelo DGL no aprendió (acc={acc:.3f})"

    ei = backend.edge_index(g)
    ones = torch.ones(ei.size(1))
    with torch.no_grad():
        full = backend.forward(model, g.ndata["feat"], ei, edge_weight=ones)
        masked = backend.forward(model, g.ndata["feat"], ei, edge_weight=ones * 0.01)
    assert full.shape == (g.num_nodes(), 4)
    assert (full - masked).abs().mean().item() > 1e-4


def test_dgl_gnnexplainer_end_to_end(dgl_graph_data):
    g = dgl_graph_data
    model = DGLGCN(g.ndata["feat"].shape[1])
    _train(model, g)

    node = int(g.ndata["label"].shape[0] // 2)
    explainer = Explainer(
        algorithm=GNNExplainer(epochs=50, lr=0.01),
        backend="dgl",
    )
    expl = explainer.explain_node(g, model, node)

    assert expl.node_importance.shape == (g.num_nodes(),)
    assert expl.edge_importance.shape == (g.num_edges(),)
    assert torch.isfinite(expl.node_importance).all()
    assert torch.isfinite(expl.edge_importance).all()
    assert expl.edge_importance.min().item() >= 0.0
    assert expl.edge_importance.max().item() <= 1.0001
    assert expl.edge_importance.sum().item() > 0.0
