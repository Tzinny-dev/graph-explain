from .base import ExplanationAlgorithm
from .gradient.integrated_gradients import IntegratedGradients
from .gradient.saliency import Saliency
from .perturbation.gnn_explainer import GNNExplainer
from .perturbation.pg_explainer import PGExplainer
from .perturbation.subgraphx import SubgraphX
from .relevance.gnn_lrp import GNNGatedLRP

__all__ = [
    "ExplanationAlgorithm",
    "GNNExplainer",
    "GNNGatedLRP",
    "IntegratedGradients",
    "PGExplainer",
    "Saliency",
    "SubgraphX",
]


def _available_methods():
    from ..core.registry import _available_methods

    return _available_methods()
