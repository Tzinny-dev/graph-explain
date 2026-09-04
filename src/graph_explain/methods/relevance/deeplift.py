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


@register("deep_lift", "deeplift", "dl")
class DeepLift(ExplanationAlgorithm):
    """DeepLIFT (rescale rule) for GCNs + ReLU + Linear.

    It is an additive rule: each input feature receives a contribution (delta)
    proportional to how much the target-class output changes when moving from a
    baseline (zero, by default) to the actual instance. The multiplier is
    propagated backwards layer by layer: exact for linear layers and GCN
    messages, and with the rescale rule (delta_out / delta_in) for elementwise
    nonlinearities.

    Returns `node_importance` (absolute contribution per node), `edge_importance`
    (contributions through the message passing of each GCNConv, per directed
    edge) and `feature_importance` (contribution per feature).
    """

    def __init__(
        self,
        eps: float = 1e-7,
        normalize: bool = False,
        node_mask_type: str | None = "attributes",
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
        x = backend.node_features(data).detach()
        edge_index = backend.edge_index(data)
        edge_weight = backend.edge_weight(data)
        num_nodes = int(x.size(0))

        nodes = self._to_node_ids(index, num_nodes)

        order, order0, logits, logits0 = self._capture(
            model, backend, x, edge_index, edge_weight
        )
        if logits.dim() != 2:
            raise ValueError(
                "DeepLift requiere predicciones node-level (logits (N, C))."
            )
        target = target_class
        if target is None:
            target = int(logits[nodes[0]].argmax().item())
        target_cls = max(0, min(int(target), logits.size(1) - 1))

        mul = torch.zeros_like(logits)
        mul[nodes, target_cls] = 1.0
        edge_rel_full = torch.zeros(edge_index.size(1), device=x.device)
        eps = self.eps

        for m, m0 in reversed(list(zip(order, order0))):
            in_x = m["args"][0]
            in_x0 = m0["args"][0]
            delta_in = in_x - in_x0
            name = m["name"]
            mod = m["module"]
            if name == "linear":
                mul = mul @ mod.weight
            elif name == "gcn":
                mul, edge_rel = self._conv_back(
                    mod, in_x, in_x0, edge_index, edge_weight, mul
                )
                edge_rel_full = edge_rel_full + edge_rel[: edge_index.size(1)]
            elif name == "activation":
                delta_out = mod(in_x) - mod(in_x0)
                mul = mul * self._rescale_ratio(delta_in, delta_out, eps)
            else:
                continue

        delta_total = logits[nodes, target_cls].sum() - logits0[nodes, target_cls].sum()
        contrib = mul * x
        node_importance = contrib.abs().sum(dim=-1)
        edge_importance = edge_rel_full.abs()
        self._last_delta_total = float(delta_total.item())

        if self.normalize:
            total = float(node_importance.sum().item())
            if total > 0:
                node_importance = node_importance / total
                total_e = float(edge_importance.sum().item())
                if total_e > 0:
                    edge_importance = edge_importance / total_e

        return Explanation(
            node_importance=node_importance.detach().cpu(),
            edge_importance=edge_importance.detach().cpu(),
            feature_importance=(
                contrib.detach().cpu() if self.node_mask_type == "attributes" else None
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
    def _is_linear(module: nn.Module) -> bool:
        return isinstance(module, nn.Linear)

    @staticmethod
    def _is_activation(module: nn.Module) -> bool:
        return isinstance(module, _ACTIVATION_GATES)

    @staticmethod
    def _is_gcn(module: nn.Module) -> bool:
        try:
            from torch_geometric.nn import GCNConv

            return isinstance(module, GCNConv)
        except ImportError:
            return False

    def _capture(self, model, backend, x, edge_index, edge_weight):
        order: list[dict[str, Any]] = []
        order0: list[dict[str, Any]] = []

        def _pre(module, args):
            order.append(
                {
                    "module": module,
                    "name": self._kind(module),
                    "args": tuple(
                        a.detach() if torch.is_tensor(a) else a for a in args
                    ),
                }
            )

        def _pre0(module, args):
            order0.append(
                {
                    "module": module,
                    "name": self._kind(module),
                    "args": tuple(
                        a.detach() if torch.is_tensor(a) else a for a in args
                    ),
                }
            )

        handles = [
            module.register_forward_pre_hook(_pre)
            for module in model.modules()
            if module is not model
        ]
        with torch.no_grad():
            logits = backend.forward(model, x, edge_index, edge_weight=edge_weight)
        for handle in handles:
            handle.remove()

        handles0 = [
            module.register_forward_pre_hook(_pre0)
            for module in model.modules()
            if module is not model
        ]
        baseline = torch.zeros_like(x)
        with torch.no_grad():
            logits0 = backend.forward(
                model, baseline, edge_index, edge_weight=edge_weight
            )
        for handle in handles0:
            handle.remove()
        return order, order0, logits, logits0

    @staticmethod
    def _kind(module: nn.Module) -> str:
        if DeepLift._is_linear(module):
            return "linear"
        if DeepLift._is_gcn(module):
            return "gcn"
        if DeepLift._is_activation(module):
            return "activation"
        return "other"

    def _conv_back(
        self,
        conv: nn.Module,
        x: torch.Tensor,
        x0: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
        mul: torch.Tensor,
    ):
        from torch_geometric.nn.conv.gcn_conv import gcn_norm
        from torch_geometric.utils import add_self_loops

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
            ei = (
                add_self_loops(edge_index, num_nodes=num_nodes)[0]
                if getattr(conv, "add_self_loops", True)
                else edge_index
            )
            norm = (
                edge_weight
                if edge_weight is not None
                else torch.ones(ei.size(1), device=x.device)
            )
        src, dst = ei[0], ei[1]
        W = conv.lin.weight  # (out, in)

        mul_agg = mul @ W  # (N, F_in)

        delta_in = x - x0
        edge_contrib = (mul_agg[dst] * (norm[:, None] * delta_in[src])).sum(dim=-1)

        mul_src = torch.zeros(num_nodes, delta_in.size(1), device=x.device)
        mul_src.index_add_(0, src, mul_agg[dst] * norm[:, None])
        return mul_src, edge_contrib

    @staticmethod
    def _rescale_ratio(delta_in: torch.Tensor, delta_out: torch.Tensor, eps: float):
        denom = delta_in.abs()
        safe = denom > eps
        ratio = torch.ones_like(delta_in)
        ratio[safe] = delta_out[safe] / delta_in[safe]
        return ratio
