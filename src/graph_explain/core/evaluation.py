from __future__ import annotations

import math


def _resolve_target(explanation) -> int:
    target = explanation.target_class
    if target is None:
        import torch

        pred = explanation.prediction_original
        if torch.is_tensor(pred):
            pred = pred.reshape(-1)
            target = int(pred.argmax().item())
        else:
            target = 0
    return int(target)


def _mask_top_k(explanation, top_k: int | None, keep_ratio: float, kind: str):
    importance = (
        explanation.edge_importance if kind == "edge" else explanation.node_importance
    )
    if importance is None:
        raise ValueError(
            f"La explicación no tiene edge_importance/node_importance para "
            f"fidelidad por {kind}."
        )
    import torch

    imp = importance.detach().cpu().reshape(-1)
    n = imp.shape[0]
    if top_k is None:
        top_k = round(n * keep_ratio)
    top_k = int(max(0, min(top_k, n)))
    if top_k == 0:
        return torch.zeros(n, dtype=torch.bool)
    idx = imp.argsort(descending=True)[:top_k]
    mask = torch.zeros(n, dtype=torch.bool)
    mask[idx] = True
    return mask


def _context(model, explanation):
    backend = explanation.metadata.get("backend")
    data = explanation.metadata.get("backing_data")
    if backend is None or data is None:
        raise ValueError(
            "Para fidelidad± es necesario que la explicación tenga backend y "
            "backing_data en metadata (úsala a través de Explainer)."
        )
    return backend, data


def evaluate_fidelity_plus(
    model,
    explanation,
    top_k: int | None = None,
    keep_ratio: float = 0.1,
    kind: str = "edge",
) -> float:
    """Necesidad: caída de P(c) al eliminar los top-k elementos importantes."""
    target = _resolve_target(explanation)
    backend, data = _context(model, explanation)
    import torch

    x = backend.node_features(data)
    edge_index = backend.edge_index(data)
    device = x.device
    with torch.no_grad():
        logits_orig = backend.forward(model, x, edge_index)
        node = _explained_node(explanation)
        p_orig = _prob(logits_orig[node], target, softmax=True)
        mask = _mask_top_k(explanation, top_k, keep_ratio, kind)
        if kind == "edge":
            edge_weight = torch.ones(edge_index.size(1), device=device)
            edge_weight[mask] = 0.0
            masked = backend.forward(model, x, edge_index, edge_weight=edge_weight)
            p_masked = _prob(masked[node], target, softmax=True)
        else:
            node_mask = torch.ones(x.size(0), device=device)
            node_mask[mask] = 0.0
            masked = backend.forward(model, x, edge_index, node_mask=node_mask)
            p_masked = _prob(masked[node], target, softmax=True)
    return float(p_orig - p_masked)


def evaluate_fidelity_minus(
    model,
    explanation,
    top_k: int | None = None,
    keep_ratio: float = 0.1,
    kind: str = "edge",
) -> float:
    """Suficiencia: P(c) conservada al conservar SOLO los top-k elementos."""
    target = _resolve_target(explanation)
    backend, data = _context(model, explanation)
    import torch

    x = backend.node_features(data)
    edge_index = backend.edge_index(data)
    device = x.device
    with torch.no_grad():
        node = _explained_node(explanation)
        mask = _mask_top_k(explanation, top_k, keep_ratio, kind)
        if kind == "edge":
            edge_weight = torch.zeros(edge_index.size(1), device=device)
            edge_weight[mask] = 1.0
            masked = backend.forward(model, x, edge_index, edge_weight=edge_weight)
        else:
            node_mask = torch.zeros(x.size(0), device=device)
            node_mask[mask] = 1.0
            masked = backend.forward(model, x, edge_index, node_mask=node_mask)
        p_kept = _prob(masked[node], target, softmax=True)
    return float(p_kept)


def _explained_node(explanation) -> int:
    if explanation.node_idx is None:
        return 0
    return int(explanation.node_idx)


def _perturbed(data, perturbation: str, noise_std: float, num_edges: int | None, rng):
    import copy

    import torch

    d = copy.deepcopy(data)
    if perturbation == "feature":
        x = d.x
        noise = torch.randn_like(x) * noise_std
        d.x = x + noise
    elif perturbation == "edge" and hasattr(d, "edge_index"):
        ei = d.edge_index
        n = ei.size(1)
        if num_edges is None:
            num_edges = max(1, n // 10)
        num_edges = min(num_edges, n)
        drop = rng.choice(n, size=num_edges, replace=False)
        keep = [i for i in range(n) if i not in set(drop.tolist())]
        d.edge_index = ei[:, keep]
        if hasattr(d, "edge_weight") and d.edge_weight is not None:
            d.edge_weight = d.edge_weight[keep]
    else:
        raise ValueError(f"perturbación desconocida: {perturbation}")
    return d


def evaluate_stability(
    get_explanation,
    data,
    num_perturbations: int = 10,
    perturbation: str = "feature",
    noise_std: float = 0.05,
    num_edges: int | None = None,
    top_k: int | None = None,
    seed: int = 0,
) -> float:
    """Estabilidad: similitud media entre explicaciones dadas perturbaciones
    pequeñas del grafo. `get_explanation` recibe un Data y devuelve Explanation."""
    from itertools import pairwise

    import numpy as np

    rng = np.random.default_rng(seed)
    exps = [
        get_explanation(_perturbed(data, perturbation, noise_std, num_edges, rng))
        for _ in range(num_perturbations)
    ]
    if not exps:
        return 1.0
    sims = []
    for a, b in pairwise(exps):
        sims.append(_explanation_similarity(a, b, top_k))
    valid = [
        s for s in sims if math.isfinite(s)
    ]  # descarta NaN (p.ej. perturbación de aristas)
    return float(np.mean(valid)) if valid else 1.0


def _explanation_similarity(a, b, top_k: int | None) -> float:
    import torch

    va = _importance_vector(a)
    vb = _importance_vector(b)
    if va is None or vb is None or va.numel() != vb.numel():
        return float("nan")
    if top_k is not None:
        ta = set(va.argsort(descending=True)[:top_k].tolist())
        tb = set(vb.argsort(descending=True)[:top_k].tolist())
        union = ta | tb
        if not union:
            return 1.0
        return float(len(ta & tb) / len(union))
    if va.norm().item() < 1e-9 or vb.norm().item() < 1e-9:
        return float("nan")
    cos = float(torch.nn.functional.cosine_similarity(va, vb, dim=0).item())
    return max(0.0, min(1.0, cos))


def _importance_vector(explanation):
    parts = []
    if explanation.node_importance is not None:
        parts.append(explanation.node_importance.detach().reshape(-1).cpu().float())
    if explanation.edge_importance is not None:
        parts.append(explanation.edge_importance.detach().reshape(-1).cpu().float())
    if not parts:
        return None
    import torch

    return torch.cat(parts)


def evaluate_gea(explanation, data=None, top_k: int | None = None) -> float:
    """Graph Explanation Accuracy: solape entre los top-k elementos de la
    explicación y el subgrafo relevante de ground truth del benchmark."""
    if data is None:
        data = explanation.metadata.get("backing_data")
    if data is None:
        raise ValueError("evaluate_gea necesita `data` (o backing_data en metadata).")
    node = _explained_node(explanation)
    gt_nodes, gt_edges = _ground_truth(data, node, explanation.metadata.get("backend"))
    if explanation.edge_importance is not None and gt_edges:
        imp = explanation.edge_importance.detach().reshape(-1)
        k = top_k if top_k is not None else len(gt_edges)
        k = int(max(0, min(k, imp.shape[0])))
        if k == 0:
            return 0.0
        top = set(imp.argsort(descending=True)[:k].tolist())
        return float(len(top & set(gt_edges)) / max(1, k))
    if explanation.node_importance is not None and gt_nodes:
        imp = explanation.node_importance.detach().reshape(-1)
        k = top_k if top_k is not None else len(gt_nodes)
        k = int(max(0, min(k, imp.shape[0])))
        if k == 0:
            return 0.0
        top = set(imp.argsort(descending=True)[:k].tolist())
        return float(len(top & set(gt_nodes)) / max(1, k))
    raise ValueError("evaluate_gea necesita edge_importance o node_importance.")


def _ground_truth(data, node, backend=None):
    try:
        from ..benchmarks.synthetic import ground_truth_edge_ids, ground_truth_nodes
    except ImportError:
        raise ValueError("Ground truth disponible solo con el benchmark sintético.")
    edge_index = backend.edge_index(data) if backend is not None else data.edge_index
    gt_nodes = ground_truth_nodes(data, node)
    gt_edges = ground_truth_edge_ids(data, node, edge_index)
    return gt_nodes, gt_edges


def evaluate_fidelity(explanation, keep_ratio: float = 0.2) -> float:
    if (
        explanation.prediction_original is None
        or explanation.prediction_explanation is None
    ):
        raise ValueError(
            "Para evaluar fidelidad la explicación debe contener "
            "prediction_original y prediction_explanation."
        )
    target = explanation.target_class
    if target is None:
        import torch

        pred = explanation.prediction_original
        if torch.is_tensor(pred):
            pred = pred.reshape(-1)
            target = int(pred.argmax().item())
        else:
            target = 0
    p_orig = _prob(explanation.prediction_original, target, softmax=True)
    p_expl = _prob(explanation.prediction_explanation, target, softmax=True)
    return float(p_orig - p_expl)


def evaluate_sparsity(explanation, local: bool = False, local_hops: int = 3) -> float:
    node_mask = explanation.node_importance
    edge_mask = explanation.edge_importance
    if node_mask is None and edge_mask is None:
        raise ValueError(
            "Para evaluar esparcidad la explicación necesita node_importance "
            "y/o edge_importance."
        )
    if local and explanation.node_idx is not None:
        node_ids, edge_ids = _local_scope(explanation, local_hops)
    else:
        node_ids = None
        edge_ids = None
    masked = 0.0
    total = 0.0
    threshold = explanation.mask_threshold
    if node_mask is not None:
        total += len(node_ids) if node_ids is not None else node_mask.shape[0]
        if node_ids is not None:
            masked += float((node_mask[node_ids] < threshold).sum())
        else:
            masked += float((node_mask < threshold).sum())
    if edge_mask is not None:
        total += len(edge_ids) if edge_ids is not None else edge_mask.shape[0]
        if edge_ids is not None:
            masked += float((edge_mask[edge_ids] < threshold).sum())
        else:
            masked += float((edge_mask < threshold).sum())
    return 1.0 - masked / total if total > 0 else 1.0


def _local_scope(explanation, hops: int) -> tuple[list[int], list[int]]:
    backend = explanation.metadata.get("backend")
    data = explanation.metadata.get("backing_data")
    edge_index = None
    num_nodes = None
    if backend is not None and data is not None:
        edge_index = backend.edge_index(data)
        num_nodes = backend.num_nodes(data)
    else:
        edge_index = getattr(data, "edge_index", None) if data is not None else None
        num_nodes = getattr(data, "num_nodes", None)
    if edge_index is None:
        edge_index = explanation.metadata.get("edge_index")
        num_nodes = num_nodes or explanation.metadata.get("num_nodes")
    if edge_index is None or num_nodes is None:
        return None, None
    node_idx = int(explanation.node_idx)
    visited = {node_idx}
    frontier = {node_idx}
    for _ in range(hops):
        nxt = set()
        for u in frontier:
            m = (edge_index[0] == u) | (edge_index[1] == u)
            nxt.update(edge_index[:, m].flatten().tolist())
        frontier = nxt - visited
        visited |= frontier
    nodes = sorted(v for v in visited if v < num_nodes)
    edge_ids = [
        i
        for i in range(edge_index.size(1))
        if int(edge_index[0, i]) in visited and int(edge_index[1, i]) in visited
    ]
    return nodes, edge_ids


def _prob(pred, class_idx, softmax: bool = False):
    import torch

    if torch.is_tensor(pred) and pred.dim() > 1:
        pred = pred.reshape(-1)
    if torch.is_tensor(pred) and pred.dim() == 1 and softmax:
        pred = pred.softmax(dim=0)
    if class_idx is None:
        if torch.is_tensor(pred) and pred.numel() == 1:
            return float(pred)
        return float(pred)
    if torch.is_tensor(pred):
        if pred.dim() == 0:
            return float(pred)
        return float(pred[class_idx])
    return float(pred)
