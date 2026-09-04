from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

from graph_explain import Explainer, evaluate_gea_graph
from graph_explain.backends import PyGAdapter
from graph_explain.benchmarks.synthetic import (
    build_graph_classification,
    ground_truth_edges_graph,
)
from graph_explain.core.registry import get_algorithm
from graph_explain.methods import (
    GNNExplainer,
    GraphLIME,
    GuidedBackprop,
    IntegratedGradients,
    RandomBaseline,
    Saliency,
)


class GraphGCN(torch.nn.Module):
    task_level = "graph"

    def __init__(self, in_channels, hidden=32):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.lin = torch.nn.Linear(hidden, 2)

    def forward(self, x, edge_index, batch=None, edge_weight=None):
        h = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        h = self.conv2(h, edge_index, edge_weight=edge_weight)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long)
        return self.lin(global_mean_pool(h, batch))


def _train_graphs(graphs, model, epochs=200, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        total = None
        for g in graphs:
            out = model(g.x, g.edge_index)
            loss = F.cross_entropy(out, g.y)
            total = loss if total is None else total + loss
        total.backward()
        optimizer.step()
    model.eval()
    return model


class TestGraphDataset:
    def setup_method(self):
        self.graphs = build_graph_classification(
            num_pos=8, num_neg=8, seed=0
        )

    def test_shapes_and_labels(self):
        assert len(self.graphs) == 16
        assert [int(g.y.item()) for g in self.graphs].count(1) == 8
        assert all(hasattr(g, "gt_edge_mask") for g in self.graphs)

    def test_ground_truth_present_only_on_positives(self):
        for g in self.graphs:
            gt = ground_truth_edges_graph(g)
            if g.y.item() == 1:
                assert len(gt) >= 2 * 6, gt  # ambas direcciones de las 6 aristas
            else:
                assert gt == []


class TestGraphLevelExplanation:
    def setup_method(self):
        graphs = build_graph_classification(num_pos=8, num_neg=8, seed=1)
        model = GraphGCN(graphs[0].x.size(1))
        self.model = _train_graphs(graphs, model)
        self.graph = graphs[0]  # positivo

    def test_explain_graph_shape(self):
        expl = Explainer(algorithm=GNNExplainer(epochs=10)).explain_graph(
            self.graph, self.model
        )
        assert expl.edge_importance.shape == (self.graph.edge_index.size(1),)
        assert expl.node_idx is None

    def test_gea_graph_positive(self):
        from graph_explain import GradXInput

        expl = Explainer(algorithm=GradXInput(), backend=PyGAdapter()).explain_graph(
            self.graph, self.model
        )
        assert expl.edge_importance is not None
        gea = evaluate_gea_graph(expl, data=self.graph, top_k=13)
        assert 0.0 < gea <= 1.0

    def test_gradient_methods_accept_graph(self):
        backend = PyGAdapter()
        for algo in (
            Saliency(),
            IntegratedGradients(steps=10),
            GuidedBackprop(),
            RandomBaseline(seed=0),
        ):
            expl = Explainer(algorithm=algo, backend=backend).explain_graph(
                self.graph, self.model
            )
            assert expl.prediction_original.numel() == 2

    def test_graph_level_flag(self):
        assert GNNExplainer.graph_level is True
        assert Saliency.graph_level is True
        assert GraphLIME.graph_level is False
        assert get_algorithm("random").graph_level is True


class TestCLIGraphLevel:
    def test_explain_without_node_graph(self, tmp_path, capsys):
        from graph_explain.cli import main

        graphs = build_graph_classification(num_pos=3, num_neg=3, seed=0)
        model = _train_graphs(graphs, GraphGCN(graphs[0].x.size(1)), epochs=50)
        model_path = tmp_path / "model.pt"
        data_path = tmp_path / "data.pt"
        torch.save(model, model_path)
        torch.save(graphs[0], data_path)
        rc = main(
            [
                "explain",
                "--model",
                str(model_path),
                "--data",
                str(data_path),
                "--method",
                "saliency",
                "--metrics",
                "fidelity_plus,gea",
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "Metrics:" in out

    def test_node_only_method_rejected_on_graph(self, tmp_path, capsys):
        from graph_explain.cli import main

        graphs = build_graph_classification(num_pos=2, num_neg=2, seed=0)
        model = GraphGCN(graphs[0].x.size(1))
        model_path = tmp_path / "model.pt"
        data_path = tmp_path / "data.pt"
        torch.save(model, model_path)
        torch.save(graphs[0], data_path)
        rc = main(
            [
                "explain",
                "--model",
                str(model_path),
                "--data",
                str(data_path),
                "--method",
                "graph_lime",
            ]
        )
        assert rc == 2
        assert "does not support graph-level" in capsys.readouterr().err

    def test_bench_graph(self, tmp_path, capsys):
        from graph_explain.cli import main

        graphs = build_graph_classification(num_pos=2, num_neg=2, seed=0)
        model = GraphGCN(graphs[0].x.size(1))
        model_path = tmp_path / "model.pt"
        data_path = tmp_path / "data.pt"
        torch.save(model, model_path)
        torch.save(graphs[0], data_path)
        rc = main(
            [
                "bench",
                "--model",
                str(model_path),
                "--data",
                str(data_path),
                "--methods",
                "saliency,random,graph_lime",
                "--no-stability",
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "Benchmark graph-level" in out
        assert "node-level only" in out