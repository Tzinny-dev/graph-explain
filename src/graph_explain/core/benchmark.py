from __future__ import annotations

import torch

from ..narration import summarize
from .evaluation import (
    evaluate_fidelity_minus,
    evaluate_fidelity_plus,
    evaluate_gea,
    evaluate_gea_graph,
    evaluate_sparsity,
    evaluate_stability,
)
from .explainer import Explainer
from .registry import get_algorithm, instantiate

DEFAULT_METHODS = [
    "gnn_explainer",
    "pg_explainer",
    "subgraphx",
    "saliency",
    "integrated_gradients",
    "gnn_lrp",
    "deep_lift",
    "grad_x_input",
    "graph_lime",
    "node_mask",
    "guided_backprop",
    "random",
    "counterfactual",
    "attention",
]

_METRICS = (
    "fidelity_plus",
    "fidelity_minus",
    "gea",
    "sparsity",
    "sparsity_local",
    "stability",
)


def _has_gat(model) -> bool:
    try:
        from torch_geometric.nn import GATConv

        return any(isinstance(m, GATConv) for m in model.modules())
    except ImportError:
        return False


def _method_kwargs(epochs, lr, seed, top_k) -> dict:
    kw: dict = {}
    for name, value in (
        ("epochs", epochs),
        ("lr", lr),
        ("seed", seed),
        ("top_k", top_k),
    ):
        if value is not None:
            kw[name] = value
    return kw


def compare(
    data,
    model,
    node: int | None = None,
    target_class: int | None = None,
    backend: str = "pyg",
    methods: list[str] | None = None,
    top_k: int = 5,
    num_perturbations: int = 5,
    noise_std: float = 0.05,
    epochs: int = 200,
    lr: float = 0.01,
    seed: int = 0,
    mask_threshold: float = 0.5,
    stability: bool = True,
) -> dict:
    """Runs several explanation methods and compares their metrics.

    Uses `node` for node-level (required) or `node=None` for graph-level
    (node-only methods are marked `skipped`). Returns a dictionary with one
    entry per method: class, predictions, metrics (fidelity±, GEA, sparsity,
    stability) and the structured `summarize` summary. Non-applicable methods
    or failing metrics are marked as `skipped`/`None` without aborting the
    rest.
    """
    from ..backends import get_backend

    if methods is None:
        methods = list(DEFAULT_METHODS)
    backend_obj = get_backend(backend)
    model.eval()

    if node is None:
        stability = False

    torch.manual_seed(seed)
    results: dict = {}
    ran: list[str] = []
    skipped: dict[str, str] = {}

    for name in methods:
        cls = get_algorithm(name)
        entry = {
            "method": name,
            "class": cls.__name__,
            "node": node,
            "target_class": target_class,
            "prediction_original": None,
            "prediction_explanation": None,
            "metrics": {m: None for m in _METRICS},
            "summary": None,
            "skipped": None,
        }
        if node is None and not cls.graph_level:
            entry["skipped"] = "node-level only"
            results[name] = entry
            skipped[name] = entry["skipped"]
            continue
        if name == "attention" and not _has_gat(model):
            entry["skipped"] = "requires a model with GATConv layers"
            results[name] = entry
            skipped[name] = entry["skipped"]
            continue
        algo = instantiate(name, **_method_kwargs(epochs, lr, seed, top_k))
        explainer = Explainer(
            algorithm=algo,
            backend=backend_obj,
            mask_threshold=mask_threshold,
        )
        torch.manual_seed(seed)
        try:
            expl = (
                explainer.explain_node(data, model, node, target_class=target_class)
                if node is not None
                else explainer.explain_graph(data, model, target_class=target_class)
            )
        except (ValueError, TypeError) as exc:
            entry["skipped"] = str(exc)
            results[name] = entry
            skipped[name] = str(exc)
            continue

        entry["prediction_original"] = _fmt(expl.prediction_original)
        entry["prediction_explanation"] = _fmt(expl.prediction_explanation)
        entry["summary"] = summarize(expl, data=data, top_k=top_k)

        m = entry["metrics"]
        expl_arg = expl
        m["fidelity_plus"] = _safe(
            lambda expl=expl_arg: float(evaluate_fidelity_plus(model, expl))
        )
        m["fidelity_minus"] = _safe(
            lambda expl=expl_arg: float(evaluate_fidelity_minus(model, expl))
        )
        if node is not None:
            m["gea"] = _safe(
                lambda expl=expl_arg: float(evaluate_gea(expl, data=data, top_k=top_k))
            )
        else:
            m["gea"] = _safe(
                lambda expl=expl_arg: float(
                    evaluate_gea_graph(expl, data=data, top_k=top_k)
                )
            )
        m["sparsity"] = _safe(lambda expl=expl_arg: float(evaluate_sparsity(expl)))
        m["sparsity_local"] = _safe(
            lambda expl=expl_arg: float(evaluate_sparsity(expl, local=True))
        )
        if stability:

            def _again(d, name=name):
                algo_r = instantiate(name, **_method_kwargs(epochs, lr, seed, top_k))
                return Explainer(
                    algorithm=algo_r,
                    backend=backend_obj,
                    mask_threshold=mask_threshold,
                ).explain_node(d, model, node)

            m["stability"] = _safe(
                lambda: float(
                    evaluate_stability(
                        _again,
                        data,
                        num_perturbations=num_perturbations,
                        noise_std=noise_std,
                        top_k=top_k,
                    )
                )
            )
        results[name] = entry
        ran.append(name)

    return {
        "_meta": {
            "node": node,
            "target_class": target_class,
            "backend": backend,
            "methods": ran,
            "skipped": skipped,
        },
        **{name: results[name] for name in methods},
    }


def report_html(results: dict, output_path: str) -> None:
    """Builds a self-contained HTML report (comparative table)."""
    meta = results["_meta"]
    rows = []
    for name, entry in results.items():
        if name.startswith("_"):
            continue
        m = entry["metrics"]
        if entry["skipped"]:
            rows.append(
                f"<tr><td>{name}</td>"
                f"<td colspan='7' class='skip'>not applicable: {entry['skipped']}</td></tr>"
            )
            continue
        cells = "".join(
            f"<td>{'-' if m[k] is None else f'{m[k]:.4f}'}</td>"
            for k in ("fidelity_plus", "fidelity_minus", "gea", "sparsity", "stability")
        )
        rows.append(
            f"<tr><td>{name} <small>({entry['class']})</small></td>{cells}</tr>"
        )

    body = "\n".join(rows)
    html = f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Benchmark - graph-explain</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 2rem; }}
  table {{ border-collapse: collapse; width: 100%; max-width: 900px; }}
  th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: right; }}
  th {{ background: #f0f0f0; }}
  td:first-child {{ text-align: left; }}
  td.skip {{ text-align: left; color: #888; font-style: italic; }}
  .meta {{ color: #555; margin-bottom: 1rem; }}
  code {{ background: #f4f4f4; padding: 0 4px; }}
</style>
</head>
<body>
<h1>Comparative explanation benchmark</h1>
<p class="meta">
  node <code>{meta["node"]}</code> &middot; target class
  <code>{meta["target_class"]}</code> &middot; backend <code>{meta["backend"]}</code>
</p>
<table>
<tr>
  <th>Method</th><th>fid+</th><th>fid-</th><th>GEA</th><th>sparsity</th><th>stability</th>
</tr>
{body}
</table>
<p class="meta">
  Generated with <code>graph-explain</code>. fid+ = necessity (drop in P(c) after
  removing top-k), fid- = sufficiency, GEA = overlap with ground truth, sparsity =
  global sparsity, stability = mean similarity under perturbations.
</p>
</body>
</html>"""
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(html)


def _fmt(value):
    if value is None:
        return None
    if hasattr(value, "tolist"):
        return [round(float(v), 4) for v in value.reshape(-1).tolist()]
    return value


def _safe(fn):
    try:
        return fn()
    except Exception:  # noqa: BLE001
        return None
