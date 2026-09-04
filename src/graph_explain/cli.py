from __future__ import annotations

import argparse
import inspect
import json
import sys
from typing import Any

from graph_explain import __version__

_METHODS = [
    "gnn_explainer",
    "gnnexplainer",
    "saliency",
    "gradient",
    "grad",
    "pg_explainer",
    "pgexplainer",
    "subgraphx",
    "subgraph_x",
    "integrated_gradients",
    "ig",
    "gnn_lrp",
    "gnn-lrp",
    "lrp",
    "deep_lift",
    "deeplift",
    "dl",
    "attention",
    "gat",
    "attention_explainer",
    "grad_x_input",
    "gradient_x_input",
    "gx",
    "counterfactual",
    "cf",
]

_METRICS = ["fidelity", "fidelity_plus", "fidelity_minus", "gea", "stability"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="graph-explain",
        description="Explicabilidad de modelos basados en grafos (GNN).",
    )
    parser.add_argument(
        "--version", action="version", version=f"graph-explain {__version__}"
    )
    sub = parser.add_subparsers(dest="command")

    explain = sub.add_parser("explain", help="Explicar la predicción de un nodo/grafo")
    explain.add_argument("--model", required=True, help="Ruta al modelo guardado (.pt)")
    explain.add_argument("--data", required=True, help="Ruta al Data guardado (.pt)")
    explain.add_argument("--method", default="gnn_explainer", choices=_METHODS)
    explain.add_argument(
        "--node", type=int, default=None, help="Índice del nodo a explicar (node-level)"
    )
    explain.add_argument("--target-class", type=int, default=None)
    explain.add_argument("--epochs", type=int, default=200)
    explain.add_argument("--lr", type=float, default=None)
    explain.add_argument(
        "--mode", default="edge", choices=["edge", "feature"], help="Modo contrafactual"
    )
    explain.add_argument("--hops", type=int, default=2)
    explain.add_argument("--max-steps", type=int, default=10)
    explain.add_argument("--eps", type=float, default=None)
    explain.add_argument("--steps", type=int, default=50)
    explain.add_argument("--normalize", action="store_true", help="GNN-LRP normalizado")
    explain.add_argument("--backend", default="pyg", choices=["pyg", "dgl"])
    explain.add_argument("--output", default=None, help="Guardar la explicación en .pt")
    explain.add_argument(
        "--plot", default=None, help="Guardar visualización en .png/.pdf"
    )
    explain.add_argument(
        "--html", default=None, help="Guardar visualización interactiva en .html"
    )
    explain.add_argument("--threshold", type=float, default=0.5)
    explain.add_argument(
        "--top-k", type=int, default=5, help="Top-k para GEA/stability/JSON"
    )
    explain.add_argument(
        "--metrics",
        default="",
        help=f"Lista separada por comas: {', '.join(_METRICS)}",
    )
    explain.add_argument("--num-perturbations", type=int, default=10)
    explain.add_argument("--noise-std", type=float, default=0.05)
    explain.add_argument("--describe", action="store_true", help="Imprimir narración")
    explain.add_argument(
        "--json", default=None, help="Exportar resumen + métricas a .json"
    )
    return parser


def _params(cls: type) -> set[str]:
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return set()
    names = set()
    for name, p in sig.parameters.items():
        if name in ("self", "kwargs", "args"):
            continue
        if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY):
            names.add(name)
    return names


def _instantiate(name: str, args: argparse.Namespace):
    from graph_explain.core.registry import get_algorithm

    cls = get_algorithm(name)
    accepted = _params(cls)
    kw: dict[str, Any] = {}
    for attr, param in (
        ("epochs", "epochs"),
        ("lr", "lr"),
        ("mode", "mode"),
        ("hops", "hops"),
        ("max_steps", "max_steps"),
        ("eps", "eps"),
        ("steps", "steps"),
    ):
        val = getattr(args, attr)
        if param in accepted and val is not None:
            kw[param] = val
    if "normalize" in accepted and args.normalize:
        kw["normalize"] = True
    return cls(**kw)


def _make_explainer(args: argparse.Namespace):
    from graph_explain import Explainer

    algorithm = _instantiate(args.method, args)
    return Explainer(
        algorithm=algorithm,
        backend=args.backend,
        mask_threshold=args.threshold,
    )


def _fmt(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return [round(float(v), 4) for v in value.reshape(-1).tolist()]
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    import torch

    if torch.is_tensor(value):
        return _json_safe(value.tolist())
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def _eval_metric(name: str, args, model, data, explanation) -> float | None:
    from graph_explain.core.evaluation import (
        evaluate_fidelity_minus,
        evaluate_fidelity_plus,
        evaluate_gea,
        evaluate_stability,
    )

    try:
        if name in ("fidelity", "fidelity_plus"):
            return float(evaluate_fidelity_plus(model, explanation))
        if name == "fidelity_minus":
            return float(evaluate_fidelity_minus(model, explanation))
        if name == "gea":
            return float(evaluate_gea(explanation, data=data, top_k=args.top_k))
        if name == "stability":
            if args.node is None:
                raise ValueError("stability requiere --node")

            def _again(d):
                return _make_explainer(args).explain_node(d, model, args.node)

            return float(
                evaluate_stability(
                    _again,
                    data,
                    num_perturbations=args.num_perturbations,
                    noise_std=args.noise_std,
                    top_k=args.top_k,
                )
            )
    except Exception as exc:  # noqa: BLE001
        print(f"  * métrica {name} no disponible: {exc}", file=sys.stderr)
        return None
    raise ValueError(f"métrica desconocida: {name}")


def _cmd_explain(args: argparse.Namespace) -> int:
    import torch

    from graph_explain.core.registry import get_algorithm

    if args.node is None:
        print(
            "Para explicar es necesario indicar --node (explicación por nodo).",
            file=sys.stderr,
        )
        return 2

    model = torch.load(args.model, map_location="cpu", weights_only=False)
    data = torch.load(args.data, map_location="cpu", weights_only=False)
    model.eval()

    algorithm = _instantiate(args.method, args)
    explainer = _make_explainer(args)
    try:
        explanation = explainer.explain(
            data,
            model,
            index=args.node,
            target_class=args.target_class,
        )
    except ValueError as exc:
        print(f"Error con {args.method}: {exc}", file=sys.stderr)
        return 2
    algorithm_class = get_algorithm(args.method)

    print(f"Método: {args.method} ({algorithm_class.__name__})")
    print(f"Predicción original: {_fmt(explanation.prediction_original)}")
    if explanation.prediction_explanation is not None:
        print(
            f"Predicción tras explicación: {_fmt(explanation.prediction_explanation)}"
        )

    metrics: dict[str, float | None] = {}
    if args.metrics:
        for name in args.metrics.split(","):
            name = name.strip()
            if not name:
                continue
            metrics[name] = _eval_metric(name, args, model, data, explanation)
        print(f"Métricas: {metrics}")

    if args.describe:
        from graph_explain import describe

        print(f"Narración: {describe(explanation, data=data, top_k=args.top_k)}")

    if args.json:
        from graph_explain import summarize

        report = {
            "version": __version__,
            "method": algorithm.name,
            "backend": args.backend,
            "node": args.node,
            "target_class": args.target_class,
            "threshold": args.threshold,
            "prediction_original": _fmt(explanation.prediction_original),
            "prediction_explanation": _fmt(explanation.prediction_explanation),
            "metrics": metrics,
            "summary": summarize(explanation, data=data, top_k=args.top_k),
        }
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(_json_safe(report), fh, indent=2, ensure_ascii=False)
        print(f"Informe JSON guardado en {args.json}")

    if args.output:
        torch.save(explanation, args.output)
        print(f"Explicación guardada en {args.output}")
    if args.plot:
        from graph_explain.visualization import visualize_static

        visualize_static(explanation, threshold=args.threshold)
        import matplotlib.pyplot as plt

        plt.savefig(args.plot, bbox_inches="tight")
        print(f"Visualización guardada en {args.plot}")
    if args.html:
        from graph_explain.visualization import visualize_interactive

        visualize_interactive(
            explanation, output_path=args.html, threshold=args.threshold
        )
        print(f"Visualización interactiva guardada en {args.html}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 1
    if args.command == "explain":
        return _cmd_explain(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
