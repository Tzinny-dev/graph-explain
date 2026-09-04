from __future__ import annotations

from typing import Any

import torch


def capture_node_embeddings(
    model: Any,
    backend: Any,
    data: Any,
    layer_index: int = -2,
) -> torch.Tensor:
    children = list(model.children())
    modules = list(model.modules())
    if not children:
        raise ValueError("El modelo no tiene submódulos para capturar embeddings")
    target = children[layer_index] if layer_index < 0 else modules[layer_index]
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _input, output):
        if isinstance(output, tuple):
            output = output[0]
        captured["emb"] = output.detach()

    handle = target.register_forward_hook(hook)
    try:
        x = backend.node_features(data)
        edge_index = backend.edge_index(data)
        edge_weight = backend.edge_weight(data)
        backend.forward(model, x, edge_index, edge_weight=edge_weight)
    finally:
        handle.remove()
    emb = captured.get("emb")
    if emb is None:
        raise RuntimeError("No se capturaron embeddings del modelo")
    return emb


def edge_embeddings(
    embeddings: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    return torch.cat(
        [embeddings[edge_index[0]], embeddings[edge_index[1]]], dim=-1
    )