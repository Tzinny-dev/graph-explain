from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ...core.explanation import Explanation
from ...core.registry import register
from ..base import ExplanationAlgorithm

_ACTIVATION_GATES = (
    nn.ReLU,
    nn.ReLU6,
    nn.LeakyReLU,
)


@register("gnn_lrp", "gnn-lrp", "lrp")
class GNNGatedLRP(ExplanationAlgorithm):
    """GNN-LRP (Layer-wise Relevance Propagation para GNNs).

    Propaga la relevancia desde el logit de la clase objetivo hacia atrás,
    capa a capa, redistribuyéndola según las contribuciones positivas de cada
    neurona (regla LRP-0 / z+). Para cada `GCNConv` la relevancia se reparte en
    dos pasos: (a) el transformador lineal `W` sobre las features agregadas y
    (b) la convolución, atribuyendo la relevancia a los nodos/aristas vecinas
    proporcionalmente a su contribución en el paso de mensajes (norma GCN
    incluida). Soporta arquitecturas GCN (`GCNConv` + `ReLU` + `Linear`).

    La relevancia resultante es no negativa (reglas positivas) y se devuelve
    como `node_importance` (suma por nodo) y `edge_importance` (por arista
    dirigida, alineada con los índices de `edge_index`).
    """

    def __init__(
        self,
        eps: float = 1e-6,
        normalize: bool = False,
        node_mask_type: str | None = None,
    ):
        self.eps = float(eps)
        self.normalize = bool(normalize)
        self.node_mask_type = node_mask_type

    def explain(
        self,
        backend: Any,
        model: Any,
        data: Any,
        index: int | list[int] | torch.Tensor | None = None,
        target_class: int | None = None,
        **kwargs,
    ) -> Explanation:
        model.eval()
        x = backend.node_features(data).detach().requires_grad_(False)
        edge_index = backend.edge_index(data)
        edge_weight = backend.edge_weight(data)
        num_nodes = int(x.size(0))

        nodes = self._to_node_ids(index, num_nodes)
        order: list[tuple[nn.Module, tuple[Any, ...]]] = []

        def _pre(module: nn.Module, args: tuple[Any, ...]):
            order.append((module, args))

        handles = [
            module.register_forward_pre_hook(_pre)
            for module in model.modules()
            if module is not model
        ]

        out = backend.forward(model, x, edge_index, edge_weight=edge_weight)
        for handle in handles:
            handle.remove()

        logits = out
        if logits.dim() != 2:
            raise ValueError(
                "GNN-LRP requiere predicciones node-level (logits (N, C))."
            )
        target = target_class
        if target is None:
            target = int(logits[nodes[0]].argmax().item())
        target_cls = max(0, min(int(target), logits.size(1) - 1))

        seed = torch.zeros_like(logits)
        seed[nodes, target_cls] = 1.0
        relevance = seed  # (N, C)

        edge_rel = torch.zeros(edge_index.size(1), device=x.device)
        eps = self.eps

        for module, args in reversed(order):
            if isinstance(module, nn.Linear):
                relevance = self._linear_lrp(module.weight, args[0], relevance, eps)
            elif self._is_gcn(module):
                rel_out, rel_edge = self._conv_lrp(
                    module, args[0], edge_index, edge_weight, relevance, eps
                )
                relevance = rel_out
                num_expanded = int(rel_edge.numel())
                if num_expanded >= edge_index.size(1):
                    edge_rel = edge_rel + rel_edge[: edge_index.size(1)]
            elif self._is_activation(module):
                gate = (args[0] > 0).to(relevance.dtype)
                relevance = relevance * gate
            else:
                continue

        node_importance = relevance.sum(dim=-1)
        if self.normalize:
            total = float(node_importance.sum().item())
            if total > 0:
                node_importance = node_importance / total
                total_e = float(edge_rel.sum().item())
                if total_e > 0:
                    edge_rel = edge_rel / total_e

        return Explanation(
            node_importance=node_importance.detach().cpu(),
            edge_importance=edge_rel.detach().cpu(),
            feature_importance=(
                relevance.detach().cpu()
                if self.node_mask_type == "attributes"
                else None
            ),
            prediction_original=logits[nodes[0]].detach().cpu(),
            prediction_explanation=None,
            node_idx=int(nodes[0]) if nodes.shape[0] == 1 else index,
            target_class=target_cls,
        )

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _to_node_ids(index, num_nodes: int) -> torch.Tensor:
        if index is None:
            return torch.zeros(1, dtype=torch.long)
        if isinstance(index, int):
            return torch.tensor([index], dtype=torch.long)
        idx = torch.as_tensor(index, dtype=torch.long)
        return idx.reshape(-1) if idx.numel() else torch.zeros(1, dtype=torch.long)

    @staticmethod
    def _is_gcn(module: nn.Module) -> bool:
        try:
            from torch_geometric.nn import GCNConv

            return isinstance(module, GCNConv)
        except ImportError:
            return False

    @staticmethod
    def _is_activation(module: nn.Module) -> bool:
        return isinstance(module, _ACTIVATION_GATES)

    @staticmethod
    def _linear_lrp(
        weight: torch.Tensor,
        x: torch.Tensor,
        r: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        """Regla z+ (LRP-0 positiva) para una transformación lineal y = Wx."""
        wp = weight.clamp(min=0)  # (out, in)
        xp = x.clamp(min=0)  # (N, in)
        contrib = xp[:, None, :] * wp[None, :, :]  # (N, out, in)
        denom = contrib.sum(dim=-1).clamp(min=eps)  # (N, out)
        return (contrib / denom[:, :, None] * r[:, :, None]).sum(dim=1)  # (N, in)

    def _conv_lrp(
        self,
        conv: nn.Module,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
        r: torch.Tensor,
        eps: float,
    ):
        """Relevancia a través de un GCNConv: lineal `W` + mensajes (norma GCN)."""
        from torch_geometric.nn.conv.gcn_conv import gcn_norm

        num_nodes = int(x.size(0))
        if getattr(conv, "normalize", True):
            ei, norm = gcn_norm(
                edge_index,
                edge_weight=edge_weight,
                num_nodes=num_nodes,
                improved=getattr(conv, "improved", False),
                add_self_loops=getattr(conv, "add_self_loops", True),
                flow=getattr(conv, "flow", "source_to_target"),
            )
        else:
            if getattr(conv, "add_self_loops", True):
                from torch_geometric.utils import add_self_loops

                ei = add_self_loops(edge_index, num_nodes=num_nodes)[0]
            else:
                ei = edge_index
            norm = (
                edge_weight
                if edge_weight is not None
                else torch.ones(ei.size(1), device=x.device)
            )
        src, dst = ei[0], ei[1]

        xp = x.clamp(min=0)
        msg = norm[:, None] * xp[src]  # (E, F)
        agg = torch.zeros(num_nodes, x.size(1), device=x.device)
        agg.index_add_(0, dst, norm[:, None] * x[src])
        agg_pos = agg.clamp(min=0)

        r_agg = self._linear_lrp(conv.lin.weight, agg_pos, r, eps)  # (N, F)

        frac = msg / agg_pos[dst].clamp(min=eps)  # (E, F)
        r_msg = (r_agg[dst] * frac).sum(dim=-1)  # (E,)
        r_to_src = torch.zeros(num_nodes, x.size(1), device=x.device)
        r_to_src.index_add_(0, src, r_agg[dst] * frac)
        return r_to_src, r_msg
