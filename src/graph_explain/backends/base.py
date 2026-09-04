from __future__ import annotations

from abc import ABC, abstractmethod
from inspect import signature
from typing import Any

import torch


class Backend(ABC):
    name: str = "base"

    @abstractmethod
    def num_nodes(self, data: Any) -> int: ...

    @abstractmethod
    def node_features(self, data: Any) -> torch.Tensor: ...

    @abstractmethod
    def edge_index(self, data: Any) -> torch.Tensor: ...

    @abstractmethod
    def edge_weight(self, data: Any) -> torch.Tensor | None: ...

    @abstractmethod
    def node_labels(self, data: Any) -> torch.Tensor | None: ...

    @abstractmethod
    def to_networkx(self, data: Any): ...

    def supports_edge_weight(self, model: Any) -> bool:
        try:
            params = signature(model.forward).parameters
        except (TypeError, ValueError):
            return False
        return "edge_weight" in params

    def forward(
        self,
        model: Any,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
        node_mask: torch.Tensor | None = None,
        **model_kwargs: Any,
    ) -> torch.Tensor:
        x_masked = x
        if node_mask is not None:
            expand = (-1,) * x_masked.dim()
            node_mask = node_mask.to(x_masked.dtype)
            x_masked = x_masked * node_mask.view((node_mask.shape[0], *expand[1:]))
        if edge_weight is not None and not self.supports_edge_weight(model):
            raise ValueError(
                "El modelo no acepta edge_weight. Los métodos por perturbación de "
                "aristas requieren modelos GNN con soporte para edge_weight "
                "(p.ej. GCNConv, GATConv)."
            )
        kwargs: dict = {}
        if edge_weight is not None:
            kwargs["edge_weight"] = edge_weight
        kwargs.update(model_kwargs)
        return model(x_masked, edge_index, **kwargs)


class PyGAdapter(Backend):
    name = "pyg"

    def num_nodes(self, data: Any) -> int:
        return int(data.num_nodes)

    def node_features(self, data: Any) -> torch.Tensor:
        return data.x

    def edge_index(self, data: Any) -> torch.Tensor:
        return data.edge_index

    def edge_weight(self, data: Any) -> torch.Tensor | None:
        return getattr(data, "edge_weight", None)

    def node_labels(self, data: Any) -> torch.Tensor | None:
        return getattr(data, "y", None)

    def to_networkx(self, data: Any):
        from torch_geometric.utils import to_networkx

        return to_networkx(data, to_undirected=True)


def get_backend(name: str) -> Backend:
    if name == "pyg":
        return PyGAdapter()
    if name == "dgl":
        from ..backends.dgl import DGLAdapter

        return DGLAdapter()
    raise ValueError(f"Backend desconocido: {name}. Disponibles: ['pyg', 'dgl']")


def default_mask_type(model: Any, data: Any) -> tuple[str | None, str | None]:
    edge_mask_type = "object" if getattr(data, "edge_index", None) is not None else None
    x = getattr(data, "x", None)
    node_mask_type = "attributes" if x is not None else None
    return node_mask_type, edge_mask_type
