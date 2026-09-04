from __future__ import annotations


def evaluate_fidelity(explanation, keep_ratio: float = 0.2) -> float:
    if explanation.prediction_original is None or explanation.prediction_explanation is None:
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