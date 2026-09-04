from __future__ import annotations

import json

import pytest
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from graph_explain import __version__
from graph_explain.benchmarks.synthetic import build_data
from graph_explain.cli import main


class GCN(torch.nn.Module):
    task_level = "node"

    def __init__(self, in_channels):
        super().__init__()
        self.c1 = GCNConv(in_channels, 16)
        self.c2 = GCNConv(16, 4)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.c1(x, edge_index, edge_weight=edge_weight))
        return self.c2(x, edge_index, edge_weight=edge_weight)


def _save_model_and_data(tmp_path):
    torch.manual_seed(0)
    data = build_data(base_nodes=30, num_houses=5, m=2, seed=0)
    model = GCN(data.x.size(1))
    model_path = tmp_path / "model.pt"
    data_path = tmp_path / "data.pt"
    torch.save(model, model_path)
    torch.save(data, data_path)
    return str(model_path), str(data_path)


class TestCLI:
    def test_version(self, capsys):
        with pytest.raises(SystemExit) as err:
            main(["--version"])
        assert err.value.code == 0
        assert f"graph-explain {__version__}" in capsys.readouterr().out

    def test_explain_counterfactual_describe_and_json(self, tmp_path, capsys):
        model_path, data_path = _save_model_and_data(tmp_path)
        json_path = tmp_path / "report.json"
        rc = main(
            [
                "explain",
                "--model",
                model_path,
                "--data",
                data_path,
                "--method",
                "counterfactual",
                "--node",
                "0",
                "--describe",
                "--json",
                str(json_path),
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "Method: counterfactual" in out
        assert "Explicación del nodo 0" in out

        report = json.loads(json_path.read_text())
        assert report["method"] == "counterfactual"
        assert report["node"] == 0
        assert report["version"] == __version__
        assert report["summary"]["counterfactual"] in (True, False)
        assert isinstance(report["summary"]["important_nodes"], list)
        assert isinstance(report["summary"]["important_edges"], list)
        assert "important_edges" in report["summary"]

    def test_explain_gnnexplainer_with_metrics(self, tmp_path, capsys):
        model_path, data_path = _save_model_and_data(tmp_path)
        rc = main(
            [
                "explain",
                "--model",
                model_path,
                "--data",
                data_path,
                "--method",
                "gnn_explainer",
                "--node",
                "0",
                "--epochs",
                "10",
                "--metrics",
                "fidelity_plus,gea",
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "fidelity_plus" in out

    def test_explain_alias_lrp(self, tmp_path):
        model_path, data_path = _save_model_and_data(tmp_path)
        assert (
            main(
                [
                    "explain",
                    "--model",
                    model_path,
                    "--data",
                    data_path,
                    "--method",
                    "lrp",
                    "--node",
                    "0",
                ]
            )
            == 0
        )

    def test_explain_new_methods(self, tmp_path, capsys):
        model_path, data_path = _save_model_and_data(tmp_path)
        for method in ("dl", "gx"):
            rc = main(
                [
                    "explain",
                    "--model",
                    model_path,
                    "--data",
                    data_path,
                    "--method",
                    method,
                    "--node",
                    "0",
                ]
            )
            assert rc == 0, method
        # attention requiere un modelo con GATConv -> error controlado (rc 2)
        rc = main(
            [
                "explain",
                "--model",
                model_path,
                "--data",
                data_path,
                "--method",
                "attention",
                "--node",
                "0",
            ]
        )
        assert rc == 2
        assert "GATConv" in capsys.readouterr().err

    def test_node_required(self, tmp_path, capsys):
        model_path, data_path = _save_model_and_data(tmp_path)
        rc = main(
            [
                "explain",
                "--model",
                model_path,
                "--data",
                data_path,
                "--method",
                "saliency",
            ]
        )
        assert rc == 2
        assert "you must pass --node" in capsys.readouterr().err

    def test_metrics_stability_and_fidelity(self, tmp_path, capsys):
        model_path, data_path = _save_model_and_data(tmp_path)
        rc = main(
            [
                "explain",
                "--model",
                model_path,
                "--data",
                data_path,
                "--method",
                "saliency",
                "--node",
                "0",
                "--metrics",
                "stability,fidelity_plus,gea",
                "--num-perturbations",
                "2",
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "stability" in out and "fidelity_plus" in out and "gea" in out
        assert "Metrics:" in out

    def test_unknown_method(self, tmp_path):
        model_path, data_path = _save_model_and_data(tmp_path)
        with pytest.raises(SystemExit):
            main(
                [
                    "explain",
                    "--model",
                    model_path,
                    "--data",
                    data_path,
                    "--method",
                    "no_such",
                ]
            )

    def test_bench_json(self, tmp_path, capsys):
        model_path, data_path = _save_model_and_data(tmp_path)
        json_path = tmp_path / "bench.json"
        rc = main(
            [
                "bench",
                "--model",
                model_path,
                "--data",
                data_path,
                "--node",
                "0",
                "--methods",
                "dl,gx,attention",
                "--no-stability",
                "--json",
                str(json_path),
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "fid+" in out and "deep_lift" in out

        report = json.loads(json_path.read_text())
        assert report["_meta"]["node"] == 0
        assert report["_meta"]["version"] == __version__
        assert "deep_lift" in report and "grad_x_input" in report
        assert report["attention"]["skipped"]
        for key in ("fidelity_plus", "fidelity_minus", "sparsity", "stability"):
            assert key in report["deep_lift"]["metrics"]
        assert report["deep_lift"]["summary"]
        assert report["deep_lift"]["metrics"]["stability"] is None

    def test_bench_html(self, tmp_path, capsys):
        model_path, data_path = _save_model_and_data(tmp_path)
        html_path = tmp_path / "bench.html"
        rc = main(
            [
                "bench",
                "--model",
                model_path,
                "--data",
                data_path,
                "--node",
                "0",
                "--methods",
                "saliency",
                "--no-stability",
                "--html",
                str(html_path),
            ]
        )
        assert rc == 0
        html = html_path.read_text()
        assert "<table>" in html and "fid+" in html
        assert capsys.readouterr().out.count("saliency") >= 1
