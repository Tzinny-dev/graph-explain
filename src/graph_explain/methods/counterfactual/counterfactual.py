from __future__ import annotations

from collections import deque
from typing import Any

import torch
import torch.nn.functional as F

from ...core.explanation import Explanation
from ...core.registry import register
from ..base import ExplanationAlgorithm


def _softmax(logits, c: int) -> float:
    return float(F.softmax(logits[None], dim=-1)[0, c].item())


@register("counterfactual", "counterfactual_explainer", "cf")
class Counterfactual(ExplanationAlgorithm):
    """Counterfactual explanation: minimal perturbation that changes the prediction.

    Finds the minimal set of edges (mode='edge') or feature coordinates
    (mode='feature') whose removal/re-scaling makes the node's prediction change
    class (or reach `flip_to`). The search is greedy and deterministic: at each
    step it removes the candidate element that most reduces `P(original class)`;
    if the class does not change within `max_steps` steps it returns the current
    state (prediction unchanged).

    The returned importance marks the modified elements (edges 0/1, nodes from
    their incidence on removed edges, changed features 0/1), with
    `prediction_explanation` = logits after the perturbation.
    """

    def __init__(
        self,
        mode: str = "edge",
        flip_to: int | None = None,
        max_steps: int = 10,
        hops: int = 2,
        eps: float = 0.0,
        seed: int = 0,
    ):
        if mode not in ("edge", "feature"):
            raise ValueError("mode debe ser 'edge' o 'feature'.")
        self.mode = mode
        self.flip_to = flip_to
        self.max_steps = int(max_steps)
        self.hops = int(hops)
        self.eps = float(eps)
        torch.manual_seed(seed)

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
        node = self._single_node(index)
        x = backend.node_features(data).detach().clone()
        edge_index = backend.edge_index(data)
        edge_weight = backend.edge_weight(data)
        num_nodes = int(x.size(0))

        orig_logits = backend.forward(model, x, edge_index, edge_weight=edge_weight)
        orig_class = int(orig_logits[node].argmax().item())
        flip_class = target_class if self.flip_to is None else self.flip_to
        target_new = None if flip_class is None else int(flip_class)

        device = x.device
        if self.mode == "edge":
            edge_importance, final_logits = self._flip_edges(
                backend,
                model,
                x,
                edge_index,
                edge_weight,
                node,
                orig_class,
                target_new,
                device,
            )
            return self._build(
                backend,
                data,
                node,
                orig_class,
                orig_logits,
                final_logits,
                node_importance=torch.bincount(
                    torch.cat(
                        [
                            edge_index[0][edge_importance.bool()],
                            edge_index[1][edge_importance.bool()],
                        ]
                    ),
                    minlength=num_nodes,
                ).float(),
                edge_importance=edge_importance,
                feature_importance=None,
            )
        feature_importance, _final_x, final_logits = self._flip_features(
            backend,
            model,
            x,
            edge_index,
            edge_weight,
            node,
            orig_class,
            target_new,
            device,
        )
        return self._build(
            backend,
            data,
            node,
            orig_class,
            orig_logits,
            final_logits,
            node_importance=torch.zeros(num_nodes).index_fill(
                0, torch.tensor([node], device=device), 1.0
            ),
            edge_importance=None,
            feature_importance=feature_importance,
        )

    # ------------------------------------------------------------------ búsquedas
    def _flip_edges(
        self,
        backend,
        model,
        x,
        edge_index,
        edge_weight,
        node,
        orig_class,
        target_new,
        device,
    ):
        num_nodes = int(x.size(0))
        weight = (
            edge_weight.detach().clone()
            if edge_weight is not None
            else torch.ones(edge_index.size(1), device=device)
        )
        candidates = self._candidate_edges(edge_index, num_nodes, node, self.hops)
        removed: list[int] = []
        pred = backend.forward(model, x, edge_index, edge_weight=weight)[node]

        def flipped(logits, tc):
            pred_cls = int(logits.argmax().item())
            return pred_cls == tc if tc is not None else pred_cls != orig_class

        for _ in range(self.max_steps):
            if flipped(pred, target_new):
                break
            remaining = [e for e in candidates if e not in removed]
            if not remaining:
                break
            best_e, best_logits, best_p = None, None, None
            with torch.no_grad():
                for e in remaining:
                    w = weight.clone()
                    w[e] = 0.0
                    lg = backend.forward(model, x, edge_index, edge_weight=w)[node]
                    p = float(_softmax(lg, orig_class))
                    if best_p is None or p < best_p:
                        best_e, best_logits, best_p = e, lg, p
            if best_e is None or (
                best_p is not None
                and best_p >= float(_softmax(pred, orig_class)) - self.eps
            ):
                break
            removed.append(best_e)
            weight[best_e] = 0.0
            pred = best_logits

        importance = torch.zeros(edge_index.size(1), device=device)
        importance[removed] = 1.0
        return importance, pred.detach()

    def _flip_features(
        self,
        backend,
        model,
        x,
        edge_index,
        edge_weight,
        node,
        orig_class,
        target_new,
        device,
    ):
        baseline = x.mean(dim=0, keepdim=True)
        x_cur = x.detach().clone()
        changed: list[int] = []
        pred = backend.forward(model, x_cur, edge_index, edge_weight=edge_weight)[node]

        def flipped(logits, tc):
            pred_cls = int(logits.argmax().item())
            return pred_cls == tc if tc is not None else pred_cls != orig_class

        for _ in range(self.max_steps):
            if flipped(pred, target_new):
                break
            best_c, best_logits, best_p = None, None, None
            with torch.no_grad():
                for c in range(x.size(1)):
                    if c in changed:
                        continue
                    xn = x_cur.clone()
                    xn[node, c] = baseline[0, c]
                    lg = backend.forward(
                        model, xn, edge_index, edge_weight=edge_weight
                    )[node]
                    p = float(_softmax(lg, orig_class))
                    if best_p is None or p < best_p:
                        best_c, best_logits, best_p = c, lg, p
            if best_c is None or (
                best_p is not None
                and best_p >= float(_softmax(pred, orig_class)) - self.eps
            ):
                break
            changed.append(best_c)
            x_cur[node, best_c] = baseline[0, best_c]
            pred = best_logits

        importance = torch.zeros(x.size(1), device=device)
        importance[changed] = 1.0
        return importance, x_cur, pred.detach()

    # ------------------------------------------------------------------ utilidades
    @staticmethod
    def _single_node(index) -> int:
        if index is None:
            return 0
        if isinstance(index, (list, tuple)):
            index = index[0]
        return int(torch.as_tensor(index).reshape(-1)[0].item())

    @staticmethod
    def _candidate_edges(edge_index, num_nodes: int, node: int, hops: int) -> list[int]:
        adj: dict[int, list[int]] = {}
        for u, v in zip(edge_index[0].tolist(), edge_index[1].tolist()):
            adj.setdefault(u, []).append(v)
        distance = {node: 0}
        queue = deque([node])
        while queue:
            cur = queue.popleft()
            if distance[cur] >= hops:
                continue
            for nb in adj.get(cur, []):
                if nb not in distance:
                    distance[nb] = distance[cur] + 1
                    queue.append(nb)
        near = set(distance)
        edges = []
        for e in range(edge_index.size(1)):
            u = int(edge_index[0, e].item())
            v = int(edge_index[1, e].item())
            if u in near or v in near:
                edges.append(e)
        return edges

    def _build(
        self,
        backend,
        data,
        node,
        orig_class,
        orig_logits,
        final_logits,
        node_importance,
        edge_importance,
        feature_importance,
    ) -> Explanation:
        metadata = {
            "backend": backend,
            "backing_data": data,
            "counterfactual": True,
            "original_class": orig_class,
        }
        return Explanation(
            node_importance=node_importance.detach().cpu(),
            edge_importance=edge_importance.detach().cpu()
            if edge_importance is not None
            else None,
            feature_importance=(
                feature_importance.detach().cpu()
                if feature_importance is not None
                else None
            ),
            prediction_original=orig_logits[node].detach().cpu(),
            prediction_explanation=final_logits.detach().cpu(),
            node_idx=node,
            target_class=int(final_logits.argmax().item())
            if final_logits.numel()
            else None,
            metadata=metadata,
            mask_threshold=0.5,
        )
