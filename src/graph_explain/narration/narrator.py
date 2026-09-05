from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import torch


def _top_values(importance, k: int) -> list[tuple[int, float]]:
    imp = importance.detach().reshape(-1)
    n = int(imp.numel())
    k = max(1, min(int(k), n))
    idx = imp.argsort(descending=True)[:k]
    return [(int(i), float(imp[i])) for i in idx.tolist()]


def _data_context(explanation, data: Any | None):
    if data is None:
        data = explanation.metadata.get("backing_data")
    backend = explanation.metadata.get("backend")
    return backend, data


def _labels(explanation, data: Any | None) -> Any | None:
    backend, data = _data_context(explanation, data)
    if backend is None or data is None:
        return None
    try:
        return backend.node_labels(data)
    except Exception:  # noqa: BLE001
        return None


def _edge_index(data, backend) -> Any | None:
    if backend is not None and data is not None:
        try:
            return backend.edge_index(data)
        except Exception:  # noqa: BLE001
            return None
    return None


_SUPPORTED_LANGS = ("es", "en")

_TEMPLATES: dict[str, dict[str, str]] = {
    "es": {
        "graph_head": "Explicación a nivel de grafo.",
        "node_head": "Explicación del nodo {node}.",
        "target": " La clase objetivo es {target}.",
        "correct": " La predicción del modelo es correcta.",
        "incorrect": " La predicción del modelo difiere de la etiqueta real.",
        "nodes": "Los nodos más relevantes son {nodes}.",
        "edges": " Las aristas más relevantes son {edges}.",
        "node_item": "nodo {i} (importancia {v:.3f})",
        "edge_item": "arista {u}-{v} ({w:.3f})",
        "edge_item_float": "arista con relevancia {v:.3f}",
        "cf_intro": " Se necesitaron {n} cambios (aristas/features eliminadas) "
        "para cambiar la predicción de {change}.",
        "cf_change": "la clase {orig} a la clase {new}",
        "cf_change_solo": "la clase {orig}",
        "nodata": "Sin datos suficientes para describir la explicación.",
        "prompt": "Eres un asistente que explica predicciones de GNNs en lenguaje "
        "natural. Dado este resumen de una explicación (JSON), escribe un párrafo "
        "breve en español (2-4 oraciones) describiendo qué hace el modelo y qué "
        "evidencia respalda su predicción. Resumen:\n",
        "llm_fallback": "[LLM no disponible: {exc}]",
    },
    "en": {
        "graph_head": "Graph-level explanation.",
        "node_head": "Explanation of node {node}.",
        "target": " The target class is {target}.",
        "correct": " The model prediction is correct.",
        "incorrect": " The model prediction differs from the true label.",
        "nodes": "The most relevant nodes are {nodes}.",
        "edges": " The most relevant edges are {edges}.",
        "node_item": "node {i} (importance {v:.3f})",
        "edge_item": "edge {u}-{v} ({w:.3f})",
        "edge_item_float": "edge with relevance {v:.3f}",
        "cf_intro": " {n} changes (removed edges/features) were needed to change the "
        "prediction from {change}.",
        "cf_change": "class {orig} to class {new}",
        "cf_change_solo": "class {orig}",
        "nodata": "Not enough data to describe the explanation.",
        "prompt": "You are an assistant that explains GNN predictions in natural "
        "language. Given this JSON summary of an explanation, write a brief "
        "paragraph (2-4 sentences) in English describing what the model does and "
        "what evidence supports its prediction. Summary:\n",
        "llm_fallback": "[LLM unavailable: {exc}]",
    },
}


def _templates(lang: str) -> dict[str, str]:
    if lang not in _SUPPORTED_LANGS:
        raise ValueError(f"lang must be one of {_SUPPORTED_LANGS}, got {lang!r}")
    return _TEMPLATES[lang]


def summarize(explanation, data: Any | None = None, top_k: int = 5) -> dict[str, Any]:
    """Structured summary of an explanation (for narration or JSON)."""
    backend, data = _data_context(explanation, data)
    node = explanation.node_idx
    target = explanation.target_class
    pred = None
    if explanation.prediction_original is not None:
        p = explanation.prediction_original
        if torch.is_tensor(p):
            pred = int(p.reshape(-1).argmax().item())
    labels = _labels(explanation, data)

    true_label = None
    if labels is not None and node is not None:
        try:
            true_label = int(labels[node].item())
        except Exception:  # noqa: BLE001
            true_label = None

    summary: dict[str, Any] = {
        "node": None if node is None else int(node),
        "target_class": target,
        "predicted_class": pred,
        "true_class": true_label,
        "correct": (
            None if pred is None or true_label is None else bool(pred == true_label)
        ),
        "important_nodes": (
            _top_values(explanation.node_importance, top_k)
            if explanation.node_importance is not None
            else []
        ),
        "important_edges": [],
        "counterfactual": bool(explanation.metadata.get("counterfactual", False)),
    }
    if explanation.edge_importance is not None:
        ei = _edge_index(data, backend)
        top = _top_values(explanation.edge_importance, top_k)
        if ei is None:
            summary["important_edges"] = [v for _, v in top]
        else:
            summary["important_edges"] = [
                (int(ei[0, i]), int(ei[1, i]), v) for i, v in top
            ]
    if summary["counterfactual"]:
        summary["original_class"] = explanation.metadata.get("original_class")
    return summary


def describe(
    explanation, data: Any | None = None, top_k: int = 5, lang: str = "es"
) -> str:
    """Deterministic template-based narration of an explanation.

    Args:
        lang: Template language, ``"es"`` (default) or ``"en"``.
    """
    _T = _templates(lang)
    s = summarize(explanation, data, top_k)
    node = s["node"]
    target = (
        s["target_class"] if s["target_class"] is not None else s["predicted_class"]
    )

    if node is None:
        head = _T["graph_head"]
    else:
        head = _T["node_head"].format(node=node)
    if target is not None:
        head += _T["target"].format(target=target)
    if s["correct"] is True:
        head += _T["correct"]
    elif s["correct"] is False:
        head += _T["incorrect"]

    nodes_txt = ", ".join(
        _T["node_item"].format(i=i, v=v) for i, v in s["important_nodes"]
    )
    tail = _T["nodes"].format(nodes=nodes_txt) if nodes_txt else ""

    if s["important_edges"]:
        pieces = []
        for e in s["important_edges"]:
            if len(e) == 3:
                u, v, w = e
                pieces.append(_T["edge_item"].format(u=u, v=v, w=w))
            else:
                pieces.append(_T["edge_item_float"].format(v=float(e)))
        tail += _T["edges"].format(edges=", ".join(pieces))

    if s["counterfactual"]:
        n = len(s["important_edges"])
        new_class = s["predicted_class"]
        if new_class is not None:
            change = _T["cf_change"].format(orig=s["original_class"], new=new_class)
        else:
            change = _T["cf_change_solo"].format(orig=s["original_class"])
        tail += _T["cf_intro"].format(n=n, change=change)
    if tail:
        head += " " + tail.strip()
    return head.strip() or _T["nodata"]


def _prompt(summary: dict[str, Any], lang: str = "es") -> str:
    _T = _templates(lang)
    return _T["prompt"] + json.dumps(summary, ensure_ascii=False, indent=2)


def narrate(
    explanation,
    llm: Callable[[str], str] | None = None,
    data: Any | None = None,
    top_k: int = 5,
    lang: str = "es",
) -> str:
    """Narrates an explanation. With `llm` (a `prompt -> text` callable) it uses the
    generative model's output; otherwise it falls back to deterministic
    template-based narration."""
    _T = _templates(lang)
    summary = summarize(explanation, data, top_k)
    deterministic = describe(explanation, data, top_k, lang=lang)
    if llm is None:
        return deterministic
    try:
        return llm(_prompt(summary, lang)).strip()
    except Exception as exc:  # noqa: BLE001
        return f"{deterministic}\n\n{_T['llm_fallback'].format(exc=exc)}"


class Narrator:
    """Reusable narrator; lets you inject the LLM just once."""

    def __init__(
        self,
        llm: Callable[[str], str] | None = None,
        top_k: int = 5,
        lang: str = "es",
    ):
        self.llm = llm
        self.top_k = top_k
        self.lang = lang

    def describe(self, explanation, data: Any | None = None) -> str:
        return describe(explanation, data, self.top_k, lang=self.lang)

    def narrate(self, explanation, data: Any | None = None) -> str:
        return narrate(explanation, self.llm, data, self.top_k, lang=self.lang)
