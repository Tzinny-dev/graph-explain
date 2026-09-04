from __future__ import annotations

from typing import Any

import torch

from .base import Backend

_FEATURE_KEYS = ("feat", "x", "features")
_LABEL_KEYS = ("label", "y", "labels")
_WEIGHT_KEYS = ("w", "weight", "edge_weight")


def _first_available(mapping: dict, keys: tuple[str, ...], default):
    for k in keys:
        if k in mapping:
            return mapping[k]
    return default


class DGLAdapter(Backend):
    """Adapter for `dgl.DGLGraph` graphs and DGL models.

    Data convention: node features live in `ndata['feat']`, labels in
    `ndata['label']` and edge weights in `edata['w']` ('x'/'weight' are also
    accepted). The DGL model must read `g.ndata['feat']` and `g.edata['w']` in
    its `forward(graph, feat)`.
    """

    name = "dgl"

    def __init__(
        self,
        feat_key: str | None = None,
        label_key: str | None = None,
        edge_weight_key: str | None = None,
    ):
        self.feat_key = feat_key
        self.label_key = label_key
        self.edge_weight_key = edge_weight_key

    @staticmethod
    def _require_dgl():
        try:
            import dgl
        except ImportError as exc:
            raise ImportError(
                "El backend 'dgl' requiere la librería DGL (pip install dgl). "
                "Además, DGL necesita una versión de PyTorch con librerías "
                "precompiladas de graphbolt (ver docs de instalación de DGL)."
            ) from exc
        return dgl

    def num_nodes(self, data: Any) -> int:
        if callable(getattr(data, "num_nodes", None)):
            return int(data.num_nodes())
        return int(data.num_nodes)

    def node_features(self, data: Any) -> torch.Tensor:
        ndata: dict = getattr(data, "ndata", {})
        feat = ndata.get(self.feat_key) if self.feat_key else ndata.get("feat")
        if feat is None:
            feat = _first_available(ndata, _FEATURE_KEYS, None)
        if feat is None:
            raise ValueError(
                "El grafo DGL no tiene features de nodo (ndata['feat']/['x'])."
            )
        return feat

    def edge_index(self, data: Any) -> torch.Tensor:
        u, v = data.edges()
        return torch.stack([u, v], dim=0)

    def edge_weight(self, data: Any) -> torch.Tensor | None:
        edata: dict = getattr(data, "edata", {})
        w = edata.get(self.edge_weight_key) if self.edge_weight_key else edata.get("w")
        if w is None:
            w = _first_available(edata, _WEIGHT_KEYS, None)
        return w

    def node_labels(self, data: Any) -> torch.Tensor | None:
        ndata: dict = getattr(data, "ndata", {})
        lab = ndata.get(self.label_key) if self.label_key else ndata.get("label")
        if lab is None:
            lab = _first_available(ndata, _LABEL_KEYS, None)
        return lab

    def to_networkx(self, data: Any):
        return data.to_networkx()

    def supports_edge_weight(self, model: Any) -> bool:
        return True

    def forward(
        self,
        model: Any,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
        node_mask: torch.Tensor | None = None,
        **model_kwargs: Any,
    ) -> torch.Tensor:
        dgl = self._require_dgl()
        x_masked = x
        if node_mask is not None:
            expand = (-1,) * x_masked.dim()
            node_mask = node_mask.to(x_masked.dtype)
            x_masked = x_masked * node_mask.view((node_mask.shape[0], *expand[1:]))
        num_nodes = max(int(x_masked.size(0)), int(edge_index.max().item()) + 1)
        g = dgl.graph(
            (edge_index[0], edge_index[1]),
            num_nodes=num_nodes,
        )
        g.ndata["feat"] = x_masked
        g.ndata["x"] = x_masked
        if edge_weight is not None:
            g.edata["w"] = edge_weight
            g.edata["weight"] = edge_weight
        kwargs: dict = {}
        kwargs.update(model_kwargs)
        return model(g, x_masked, **kwargs)
