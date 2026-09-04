from __future__ import annotations

import argparse
import sys
from typing import Any


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="graph-explain",
        description="Explicabilidad de modelos basados en grafos (GNN).",
    )
    sub = parser.add_subparsers(dest="command")

    explain = sub.add_parser("explain", help="Explicar la predicción de un nodo/grafo")
    explain.add_argument("--model", required=True, help="Ruta al modelo guardado (.pt)")
    explain.add_argument("--data", required=True, help="Ruta al Data guardado (.pt)")
    explain.add_argument(
        "--method",
        default="gnn_explainer",
        choices=[
            "gnn_explainer",
            "gnnexplainer",
            "saliency",
            "gradient",
            "pg_explainer",
            "pgexplainer",
            "subgraphx",
            "integrated_gradients",
            "ig",
        ],
    )
    explain.add_argument("--node", type=int, default=None, help="Índice del nodo a explicar (node-level)")
    explain.add_argument("--target-class", type=int, default=None)
    explain.add_argument("--epochs", type=int, default=200)
    explain.add_argument("--backend", default="pyg", choices=["pyg", "dgl"])
    explain.add_argument("--output", default=None, help="Guardar la explicación en .pt")
    explain.add_argument("--plot", default=None, help="Guardar visualización estática en .png/.pdf")
    explain.add_argument("--html", default=None, help="Guardar visualización interactiva en .html")
    explain.add_argument("--threshold", type=float, default=0.5)

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "explain":
        return _cmd_explain(args)
    return 1


def _cmd_explain(args: argparse.Namespace) -> int:
    import torch

    from graph_explain import Explainer
    from graph_explain.core.registry import get_algorithm
    from graph_explain.visualization import visualize_static

    model = torch.load(args.model, map_location="cpu", weights_only=False)
    data = torch.load(args.data, map_location="cpu", weights_only=False)
    model.eval()

    algorithm_cls = get_algorithm(args.method)
    init_params = getattr(algorithm_cls.__init__, "__code__", None)
    if init_params is not None and "epochs" in init_params.co_varnames:
        algorithm = algorithm_cls(epochs=args.epochs)
    else:
        algorithm = algorithm_cls()
    explainer = Explainer(
        algorithm=algorithm,
        backend=args.backend,
        mask_threshold=args.threshold,
    )
    explanation = explainer.explain(
        data,
        model,
        index=args.node,
        target_class=args.target_class,
    )
    metrics = {}
    if explanation.prediction_explanation is not None:
        metrics.update(explanation.evaluate(metrics=["fidelity"]))
    if explanation.edge_importance is not None or explanation.node_importance is not None:
        metrics.update(explanation.evaluate(metrics=["sparsity"]))
    print(f"Predicción original: {_fmt(explanation.prediction_original)}")
    print(f"Métricas: {metrics}")
    if args.output:
        torch.save(explanation, args.output)
        print(f"Explicación guardada en {args.output}")
    if args.plot:
        visualize_static(explanation, threshold=args.threshold)
        import matplotlib.pyplot as plt

        plt.savefig(args.plot, bbox_inches="tight")
        print(f"Visualización guardada en {args.plot}")
    if args.html:
        from graph_explain.visualization import visualize_interactive

        visualize_interactive(explanation, output_path=args.html, threshold=args.threshold)
        print(f"Visualización interactiva guardada en {args.html}")
    return 0


def _fmt(pred: Any) -> Any:
    if hasattr(pred, "tolist"):
        return [round(float(v), 4) for v in pred.flatten().tolist()]
    return pred


if __name__ == "__main__":
    sys.exit(main())