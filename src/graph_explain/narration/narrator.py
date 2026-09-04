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


def summarize(explanation, data: Any | None = None, top_k: int = 5) -> dict[str, Any]:
    """Resumen estructurado de una explicación (para narración o JSON)."""
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


def describe(explanation, data: Any | None = None, top_k: int = 5) -> str:
    """Narración determinista (plantillas) en español de una explicación."""
    s = summarize(explanation, data, top_k)
    node = s["node"]
    target = (
        s["target_class"] if s["target_class"] is not None else s["predicted_class"]
    )

    if node is None:
        head = "Explicación a nivel de grafo."
    else:
        head = f"Explicación del nodo {node}."
    if target is not None:
        head += f" La clase objetivo es {target}."
    if s["correct"] is True:
        head += " La predicción del modelo es correcta."
    elif s["correct"] is False:
        head += " La predicción del modelo difiere de la etiqueta real."

    nodes_txt = ", ".join(
        f"nodo {i} (importancia {v:.3f})" for i, v in s["important_nodes"]
    )
    tail = f"Los nodos más relevantes son {nodes_txt}." if nodes_txt else ""

    if s["important_edges"]:
        pieces = []
        for e in s["important_edges"]:
            if len(e) == 3:
                u, v, w = e
                pieces.append(f"arista {u}-{v} ({w:.3f})")
            else:
                pieces.append(f"arista con relevancia {float(e):.3f}")
        tail += " Las aristas más relevantes son " + ", ".join(pieces) + "."

    if s["counterfactual"]:
        n = len(s["important_edges"])
        new_class = s["predicted_class"]
        change = (
            f"la clase {s['original_class']} a la clase {new_class}"
            if new_class is not None
            else f"la clase {s['original_class']}"
        )
        tail += (
            f" Se necesitaron {n} cambios"
            f" (aristas/features eliminadas) para cambiar la predicción de"
            f" {change}."
        )
    if tail:
        head += " " + tail.strip()
    return head.strip() or "Sin datos suficientes para describir la explicación."


def _prompt(summary: dict[str, Any]) -> str:
    return (
        "Eres un asistente que explica predicciones de GNNs en lenguaje natural. "
        "Dado este resumen de una explicación (JSON), escribe un párrafo breve "
        "en español (2-4 oraciones) describiendo qué hace el modelo y qué "
        "evidencia respalda su predicción. Resumen:\n"
        + json.dumps(summary, ensure_ascii=False, indent=2)
    )


def narrate(
    explanation,
    llm: Callable[[str], str] | None = None,
    data: Any | None = None,
    top_k: int = 5,
) -> str:
    """Narra una explicación. Con `llm` (callable prompt->text) usa la salida
    del modelo generativo; sin él, usa la narración determinista por plantilla."""
    summary = summarize(explanation, data, top_k)
    deterministic = describe(explanation, data, top_k)
    if llm is None:
        return deterministic
    try:
        return llm(_prompt(summary)).strip()
    except Exception as exc:  # noqa: BLE001
        return f"{deterministic}\n\n[LLM no disponible: {exc}]"


class Narrator:
    """Narrador reutilizable; permite inyectar el LLM una sola vez."""

    def __init__(self, llm: Callable[[str], str] | None = None, top_k: int = 5):
        self.llm = llm
        self.top_k = top_k

    def describe(self, explanation, data: Any | None = None) -> str:
        return describe(explanation, data, self.top_k)

    def narrate(self, explanation, data: Any | None = None) -> str:
        return narrate(explanation, self.llm, data, self.top_k)
