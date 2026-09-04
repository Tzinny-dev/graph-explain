from .base import ExplanationAlgorithm
from .gradient.integrated_gradients import IntegratedGradients
from .gradient.saliency import Saliency
from .perturbation.gnn_explainer import GNNExplainer
from .perturbation.pg_explainer import PGExplainer
from .perturbation.subgraphx import SubgraphX

__all__ = [
    "ExplanationAlgorithm",
    "GNNExplainer",
    "IntegratedGradients",
    "PGExplainer",
    "Saliency",
    "SubgraphX",
]


def _available_methods():
    from ..core.registry import _available_methods

    return _available_methods()