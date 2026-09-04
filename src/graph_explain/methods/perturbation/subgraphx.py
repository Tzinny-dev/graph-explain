from __future__ import annotations

import math
from typing import Any

import torch

from ...core.explanation import Explanation
from ...core.registry import register
from ..base import ExplanationAlgorithm


class MCTSNode:
    __slots__ = ("children", "parent", "reward", "state", "value", "visits")

    def __init__(self, state, parent=None):
        self.state = frozenset(state)
        self.parent = parent
        self.children: dict[int, MCTSNode] = {}
        self.visits = 0
        self.value = 0.0
        self.reward = -math.inf


@register("subgraphx", "subgraph_x")
class SubgraphX(ExplanationAlgorithm):
    def __init__(
        self,
        num_hops: int = 3,
        rollout: int = 30,
        high_cpu: bool = False,
        reward_method: str = "mi",
        lambda_connect: float = 0.5,
        lambda_size: float = 0.05,
        max_nodes: int = 20,
        prune: bool = True,
        seed: int = 0,
        **kwargs,
    ):
        self.num_hops = num_hops
        self.rollout = rollout
        self.high_cpu = high_cpu
        self.reward_method = reward_method
        self.lambda_connect = lambda_connect
        self.lambda_size = lambda_size
        self.max_nodes = max_nodes
        self.prune = prune
        self.seed = seed

    def explain(
        self,
        backend: Any,
        model: Any,
        data: Any,
        index: int | torch.Tensor | None,
        target_class: int | None = None,
        **kwargs,
    ) -> Explanation:
        if index is None:
            raise ValueError("SubgraphX requiere un nodo de interés (index)")
        model.eval()
        x = backend.node_features(data)
        edge_index = backend.edge_index(data)
        num_nodes = backend.num_nodes(data)

        ni = int(index[0]) if torch.is_tensor(index) else int(index)
        device = x.device
        baseline = x.mean(dim=0, keepdim=True)

        logits = backend.forward(model, x, edge_index)
        if target_class is None:
            target_class = int(logits[ni].argmax().item()) if logits.dim() == 2 else 0

        candidates = self._neighborhood(
            edge_index, num_nodes, ni, self.num_hops, self.max_nodes
        )
        if not candidates:
            candidates = list(range(num_nodes))
        candidates.remove(ni)

        best_state, _ = self._mcts(
            backend, model, x, edge_index, ni, candidates, target_class, baseline
        )

        selected = set(best_state)
        selected = self._grow_if_unfaithful(
            backend,
            model,
            x,
            edge_index,
            ni,
            target_class,
            selected,
            candidates,
            baseline,
        )
        if self.prune:
            selected = self._prune(
                backend, model, x, edge_index, ni, target_class, selected, baseline
            )

        edge_mask = self._edge_mask(edge_index, selected, num_nodes, device)
        node_importance = self._node_importance(edge_index, selected, num_nodes, device)

        with torch.no_grad():
            x_eff = self._masked_features(x, selected, baseline)
            pred_masked = backend.forward(
                model, x_eff, edge_index, edge_weight=edge_mask
            )
            pred_node = (
                pred_masked[ni] if pred_masked.dim() == 2 else pred_masked.unsqueeze(0)
            )

        return Explanation(
            node_importance=node_importance.cpu(),
            edge_importance=edge_mask.cpu(),
            feature_importance=None,
            prediction_original=(
                logits[ni].detach().reshape(1, -1).cpu()
                if logits.dim() == 2
                else logits.detach().reshape(1, -1).cpu()
            ),
            prediction_explanation=pred_node.detach().reshape(1, -1).cpu(),
            node_idx=ni,
            target_class=target_class,
            metadata={"selected": sorted(selected)},
        )

    def _neighborhood(self, edge_index, num_nodes, node, hops, max_nodes) -> list[int]:
        visited = {node}
        frontier = {node}
        for _ in range(hops):
            nxt: set[int] = set()
            for u in frontier:
                mask = (edge_index[0] == u) | (edge_index[1] == u)
                nxt.update(edge_index[:, mask].flatten().tolist())
            frontier = nxt - visited
            visited |= frontier
        ordered = sorted(visited)
        if len(ordered) > max_nodes:
            gated = set(ordered[:max_nodes])
            gated.add(node)
            ordered = sorted(gated)
        return ordered

    def _mcts(
        self, backend, model, x, edge_index, node, candidates, target_class, baseline
    ):
        root_state = (node,)
        root = MCTSNode(root_state)
        for _ in range(self.rollout):
            leaf, path = self._select(root, node, candidates)
            reward = self._evaluate(
                backend, model, x, edge_index, leaf.state, node, target_class, baseline
            )
            self._backprop(path, reward)
        best_leaf = self._best_child(root)
        return best_leaf.state, best_leaf.reward

    def _select(self, root, node, candidates):
        current = root
        path = [root]
        while current.children:
            unexplored = [
                n
                for n in candidates
                if n not in current.state and n not in current.children
            ]
            if unexplored:
                nxt_node = unexplored[0]
                new_state = tuple(sorted(current.state | {nxt_node}))
                child = MCTSNode(new_state, parent=current)
                current.children[nxt_node] = child
                path.append(child)
                return child, path
            best_child, best_score = None, -math.inf
            for child in current.children.values():
                uct = (child.value / max(child.visits, 1)) + math.sqrt(
                    2 * math.log(max(root.visits, 1)) / (child.visits + 1)
                )
                if uct > best_score:
                    best_score, best_child = uct, child
            if best_child is None or len(current.state) >= len(candidates):
                return current, path
            current = best_child
            path.append(current)
        return current, path

    def _evaluate(
        self, backend, model, x, edge_index, state, node, target_class, baseline
    ):
        selected = set(state)
        if len(selected) == 0:
            return -1e6
        weight = self._edge_mask(edge_index, selected, x.size(0), x.device)
        x_eff = self._masked_features(x, selected, baseline)
        with torch.no_grad():
            pred = backend.forward(model, x_eff, edge_index, edge_weight=weight)
        if pred.dim() == 2:
            logits_n = pred[node]
            logp = logits_n.log_softmax(-1)[target_class].item()
            if int(logits_n.argmax().item()) != target_class:
                logp -= 20.0
        else:
            logp = float(pred[node])
        components = self._connectivity(edge_index, selected)
        size_penalty = self.lambda_size * (len(selected) - 1)
        return logp - self.lambda_connect * (components - 1) - size_penalty

    def _grow_if_unfaithful(
        self,
        backend,
        model,
        x,
        edge_index,
        node,
        target_class,
        selected,
        candidates,
        baseline,
    ):
        selected = set(selected)
        if not self._preserves_class(
            backend, model, x, edge_index, node, target_class, selected, baseline
        ):
            remaining = [c for c in candidates if c not in selected]
            added = set()
            while remaining:
                best, best_reward = None, -math.inf
                for u in remaining:
                    trial = selected | {u}
                    weight = self._edge_mask(edge_index, trial, x.size(0), x.device)
                    x_eff = self._masked_features(x, trial, baseline)
                    with torch.no_grad():
                        pred = backend.forward(
                            model, x_eff, edge_index, edge_weight=weight
                        )
                    if pred.dim() != 2:
                        continue
                    r = pred[node].log_softmax(-1)[target_class].item()
                    if int(pred[node].argmax().item()) != target_class:
                        r -= 20.0
                    if r > best_reward:
                        best_reward, best = r, u
                if best is None:
                    break
                selected.add(best)
                remaining.remove(best)
                added.add(best)
                if self._preserves_class(
                    backend,
                    model,
                    x,
                    edge_index,
                    node,
                    target_class,
                    selected,
                    baseline,
                ):
                    break
        return selected

    def _preserves_class(
        self, backend, model, x, edge_index, node, target_class, selected, baseline
    ) -> bool:
        if len(selected) == 0:
            return False
        weight = self._edge_mask(edge_index, selected, x.size(0), x.device)
        x_eff = self._masked_features(x, selected, baseline)
        with torch.no_grad():
            pred = backend.forward(model, x_eff, edge_index, edge_weight=weight)
        if pred.dim() != 2:
            return True
        return (
            not (node >= pred.size(0))
            and int(pred[node].argmax().item()) == target_class
        )

    @staticmethod
    def _masked_features(x, selected, baseline):
        sel = torch.zeros(x.size(0), dtype=torch.bool, device=x.device)
        sel[list(selected)] = True
        return baseline + (x - baseline) * sel.unsqueeze(-1)

    def _prune(
        self, backend, model, x, edge_index, node, target_class, selected, baseline
    ):
        selected = set(selected)
        weight = self._edge_mask(edge_index, selected, x.size(0), x.device)
        x_eff = self._masked_features(x, selected, baseline)
        with torch.no_grad():
            pred = backend.forward(model, x_eff, edge_index, edge_weight=weight)
        if pred.dim() != 2:
            return selected
        if int(pred[node].argmax().item()) != target_class:
            return selected
        keep = sorted(selected)
        if len(keep) > 1:
            ordered = self._removal_order(
                backend, model, x, edge_index, node, target_class, keep, baseline
            )
            for u in ordered:
                trial = set(keep) - {u}
                if u == node or not trial:
                    continue
                t_weight = self._edge_mask(edge_index, trial, x.size(0), x.device)
                t_x = self._masked_features(x, trial, baseline)
                with torch.no_grad():
                    t_pred = backend.forward(
                        model, t_x, edge_index, edge_weight=t_weight
                    )
                if int(t_pred[node].argmax().item()) == target_class:
                    keep = list(trial)
        return set(keep)

    def _removal_order(
        self, backend, model, x, edge_index, node, target_class, keep, baseline
    ):
        scores = []
        for u in keep:
            if u == node:
                continue
            trial = set(keep) - {u}
            t_weight = self._edge_mask(edge_index, trial, x.size(0), x.device)
            t_x = self._masked_features(x, trial, baseline)
            with torch.no_grad():
                t_pred = backend.forward(model, t_x, edge_index, edge_weight=t_weight)
            if int(t_pred[node].argmax().item()) != target_class:
                scores.append((u, -math.inf))
            else:
                logp_drop = t_pred[node].log_softmax(-1)[target_class].item()
                scores.append((u, logp_drop))
        return [u for u, _ in sorted(scores, key=lambda t: t[1])]

    def _backprop(self, path, reward):
        for n in reversed(path):
            n.visits += 1
            n.value += reward
            n.reward = max(n.reward, reward)

    def _best_child(self, root):
        best = root
        queue = [root]
        while queue:
            cur = queue.pop(0)
            if cur.reward > best.reward:
                best = cur
            queue.extend(cur.children.values())
        return best

    @staticmethod
    def _edge_mask(edge_index, selected, num_nodes, device):
        src = edge_index[0]
        dst = edge_index[1]
        both = torch.isin(
            src, torch.as_tensor(list(selected), device=device)
        ) & torch.isin(dst, torch.as_tensor(list(selected), device=device))
        return both.to(torch.float32)

    @staticmethod
    def _node_importance(edge_index, selected, num_nodes, device):
        imp = torch.zeros(num_nodes, dtype=torch.float32, device=device)
        sel = torch.as_tensor(list(selected), device=device)
        for s in sel:
            imp[s] = 1.0
        return imp

    @staticmethod
    def _connectivity(edge_index, selected):
        sel = set(selected)
        seen = set()
        comps = 0
        adj = {}
        src = edge_index[0].tolist()
        dst = edge_index[1].tolist()
        for u, v in zip(src, dst):
            if u in sel and v in sel:
                adj.setdefault(u, set()).add(v)
                adj.setdefault(v, set()).add(u)
        for s in sel:
            if s in seen:
                continue
            comps += 1
            stack = [s]
            while stack:
                cur = stack.pop()
                if cur in seen:
                    continue
                seen.add(cur)
                for nb in adj.get(cur, ()):
                    if nb not in seen:
                        stack.append(nb)
        return comps
