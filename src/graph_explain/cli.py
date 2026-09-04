from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from graph_explain import __version__
from graph_explain.core.registry import get_algorithm, instantiate

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
    "graph_lime",
    "glime",
    "gl",
    "node_mask",
    "nodemask",
    "nm",
    "guided_backprop",
    "guided-backprop",
    "gbp",
    "random",
    "random_baseline",
    "rand",
    "counterfactual",
    "cf",
]

_METRICS = ["fidelity", "fidelity_plus", "fidelity_minus", "gea", "stability"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="graph-explain",
        description="Explainability for graph-based models (GNNs).",
    )
    parser.add_argument(
        "--version", action="version", version=f"graph-explain {__version__}"
    )
    sub = parser.add_subparsers(dest="command")

    explain = sub.add_parser(
        "explain", help="Explain the prediction of a node/graph"
    )
    explain.add_argument("--model", required=True, help="Path to the saved model (.pt)")
    explain.add_argument(
        "--data", required=True, help="Path to the saved Data (.pt)"
    )
    explain.add_argument("--method", default="gnn_explainer", choices=_METHODS)
    explain.add_argument(
        "--node",
        type=int,
        default=None,
        help="Node index to explain (node-level)",
    )
    explain.add_argument("--target-class", type=int, default=None)
    explain.add_argument("--epochs", type=int, default=200)
    explain.add_argument("--lr", type=float, default=None)
    explain.add_argument(
        "--mode",
        default="edge",
        choices=["edge", "feature"],
        help="Counterfactual mode",
    )
    explain.add_argument("--hops", type=int, default=2)
    explain.add_argument("--max-steps", type=int, default=10)
    explain.add_argument("--eps", type=float, default=None)
    explain.add_argument("--steps", type=int, default=50)
    explain.add_argument("--normalize", action="store_true", help="Normalized GNN-LRP")
    explain.add_argument("--backend", default="pyg", choices=["pyg", "dgl"])
    explain.add_argument("--output", default=None, help="Save the explanation to .pt")
    explain.add_argument(
        "--plot", default=None, help="Save visualization to .png/.pdf"
    )
    explain.add_argument(
        "--html",
        default=None,
        help="Save interactive visualization to .html",
    )
    explain.add_argument("--threshold", type=float, default=0.5)
    explain.add_argument(
        "--top-k", type=int, default=5, help="Top-k for GEA/stability/JSON"
    )
    explain.add_argument(
        "--metrics",
        default="",
        help=f"Comma-separated list: {', '.join(_METRICS)}",
    )
    explain.add_argument("--num-perturbations", type=int, default=10)
    explain.add_argument("--noise-std", type=float, default=0.05)
    explain.add_argument("--describe", action="store_true", help="Print the narration")
    explain.add_argument(
        "--json", default=None, help="Export summary + metrics to .json"
    )

    bench = sub.add_parser(
        "bench", help="Comparative benchmark of methods over a node"
    )
    bench.add_argument("--model", required=True, help="Path to the saved model (.pt)")
    bench.add_argument(
        "--data", required=True, help="Path to the saved Data (.pt)"
    )
    bench.add_argument(
        "--node",
        type=int,
        default=None,
        help="Node index (node-level); omit for graph-level",
    )
    bench.add_argument("--target-class", type=int, default=None)
    bench.add_argument(
        "--methods",
        default="all",
        help=f"Comma-separated methods (or 'all'). Aliases: {', '.join(_METHODS)}",
    )
    bench.add_argument("--backend", default="pyg", choices=["pyg", "dgl"])
    bench.add_argument("--epochs", type=int, default=200)
    bench.add_argument("--lr", type=float, default=None)
    bench.add_argument("--top-k", type=int, default=5)
    bench.add_argument("--num-perturbations", type=int, default=5)
    bench.add_argument("--noise-std", type=float, default=0.05)
    bench.add_argument("--threshold", type=float, default=0.5)
    bench.add_argument("--seed", type=int, default=0)
    bench.add_argument("--no-stability", action="store_true")
    bench.add_argument("--json", default=None, help="Export results to .json")
    bench.add_argument(
        "--html", default=None, help="Export comparative report to .html"
    )
    return parser


def _instantiate(name: str, args: argparse.Namespace):
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
        if val is not None:
            kw[param] = val
    if "normalize" in dir(args) and args.normalize:
        kw["normalize"] = True
    return instantiate(name, **kw)


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
            if getattr(model, "task_level", "node") == "graph":
                from graph_explain.core.evaluation import evaluate_gea_graph

                return float(
                    evaluate_gea_graph(explanation, data=data, top_k=args.top_k)
                )
            return float(evaluate_gea(explanation, data=data, top_k=args.top_k))
        if name == "stability":
            if args.node is None:
                raise ValueError("stability requires --node")

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
        print(f"  * metric {name} unavailable: {exc}", file=sys.stderr)
        return None
    raise ValueError(f"unknown metric: {name}")


def _cmd_explain(args: argparse.Namespace) -> int:
    import torch

    from graph_explain.core.registry import get_algorithm

    model = torch.load(args.model, map_location="cpu", weights_only=False)
    data = torch.load(args.data, map_location="cpu", weights_only=False)
    model.eval()

    task = getattr(model, "task_level", "node")
    if args.node is None:
        if task == "graph":
            if not get_algorithm(args.method).graph_level:
                print(
                    f"Error: {args.method} does not support graph-level explanations "
                    "(node-level only).",
                    file=sys.stderr,
                )
                return 2
            index = None
        else:
            print(
                "To explain a node you must pass --node (per-node explanation).",
                file=sys.stderr,
            )
            return 2
    else:
        index = args.node

    algorithm = _instantiate(args.method, args)
    explainer = _make_explainer(args)
    try:
        explanation = explainer.explain(
            data,
            model,
            index=index,
            target_class=args.target_class,
        )
    except ValueError as exc:
        print(f"Error with {args.method}: {exc}", file=sys.stderr)
        return 2
    algorithm_class = get_algorithm(args.method)

    print(f"Method: {args.method} ({algorithm_class.__name__})")
    print(f"Original prediction: {_fmt(explanation.prediction_original)}")
    if explanation.prediction_explanation is not None:
        print(
            f"Prediction after explanation: {_fmt(explanation.prediction_explanation)}"
        )

    metrics: dict[str, float | None] = {}
    if args.metrics:
        for name in args.metrics.split(","):
            name = name.strip()
            if not name:
                continue
            metrics[name] = _eval_metric(name, args, model, data, explanation)
        print(f"Metrics: {metrics}")

    if args.describe:
        from graph_explain import describe

        print(f"Narration: {describe(explanation, data=data, top_k=args.top_k)}")

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
        print(f"JSON report saved to {args.json}")

    if args.output:
        torch.save(explanation, args.output)
        print(f"Explanation saved to {args.output}")
    if args.plot:
        from graph_explain.visualization import visualize_static

        visualize_static(explanation, threshold=args.threshold)
        import matplotlib.pyplot as plt

        plt.savefig(args.plot, bbox_inches="tight")
        print(f"Visualization saved to {args.plot}")
    if args.html:
        from graph_explain.visualization import visualize_interactive

        visualize_interactive(
            explanation, output_path=args.html, threshold=args.threshold
        )
        print(f"Interactive visualization saved to {args.html}")
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    import torch

    from graph_explain.core.benchmark import DEFAULT_METHODS, compare, report_html

    model = torch.load(args.model, map_location="cpu", weights_only=False)
    data = torch.load(args.data, map_location="cpu", weights_only=False)
    model.eval()

    task = getattr(model, "task_level", "node")
    if args.node is None and task != "graph":
        print("For node-level you must pass --node.", file=sys.stderr)
        return 2

    if args.methods.strip().lower() == "all":
        methods = list(DEFAULT_METHODS)
    else:
        methods = []
        for item in args.methods.split(","):
            item = item.strip()
            if not item:
                continue
            try:
                methods.append(get_algorithm(item).name)
            except ValueError as exc:
                print(f"Error: {exc}", file=sys.stderr)
                return 2

    methods = list(dict.fromkeys(methods))

    print(
        f"Benchmark {'over node ' + str(args.node) if args.node is not None else 'graph-level'}"
        f" ({len(methods)} methods) - backend {args.backend}\n"
    )
    results = compare(
        data,
        model,
        node=args.node,
        target_class=args.target_class,
        backend=args.backend,
        methods=methods,
        top_k=args.top_k,
        num_perturbations=args.num_perturbations,
        noise_std=args.noise_std,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        mask_threshold=args.threshold,
        stability=not args.no_stability,
    )

    headers = ("Method", "fid+", "fid-", "GEA", "sparsity", "stab")
    widths = [len(h) for h in headers]
    rows: list[tuple[Any, ...]] = []
    for name, entry in results.items():
        if name.startswith("_"):
            continue
        if entry["skipped"]:
            rows.append((name, "not applicable", "", "", "", ""))
            continue
        m = entry["metrics"]
        rows.append(
            (
                name,
                "" if m["fidelity_plus"] is None else f"{m['fidelity_plus']:.3f}",
                "" if m["fidelity_minus"] is None else f"{m['fidelity_minus']:.3f}",
                "" if m["gea"] is None else f"{m['gea']:.3f}",
                "" if m["sparsity"] is None else f"{m['sparsity']:.3f}",
                "" if m["stability"] is None else f"{m['stability']:.3f}",
            )
        )

    widths[0] = max(widths[0], max((len(r[0]) for r in rows), default=0))
    for i in range(1, len(headers)):
        widths[i] = max(widths[i], max((len(r[i]) for r in rows if r[i]), default=0))

    for i, h in enumerate(headers):
        widths[i] = max(widths[i], len(h))
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(line)
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)))

    skipped = results["_meta"]["skipped"]
    if skipped:
        print("\nSkipped:")
        for name, reason in skipped.items():
            print(f"  {name}: {reason}")

    if args.json:
        out = {
            name: entry for name, entry in results.items() if not name.startswith("_")
        }
        out["_meta"] = results["_meta"]
        out["_meta"]["version"] = __version__
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(_json_safe(out), fh, indent=2, ensure_ascii=False)
        print(f"\nResults saved to JSON at {args.json}")
    if args.html:
        report_html(results, args.html)
        print(f"HTML report saved to {args.html}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 1
    if args.command == "explain":
        return _cmd_explain(args)
    if args.command == "bench":
        return _cmd_bench(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
