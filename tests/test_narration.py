from __future__ import annotations

import torch

from graph_explain import (
    Counterfactual,
    Explainer,
    GNNExplainer,
    Narrator,
    describe,
    narrate,
)
from graph_explain.benchmarks.synthetic import build_data
from tests.test_core import make_model


class TestNarration:
    def _setup(self):
        torch.manual_seed(0)
        data = build_data(base_nodes=60, num_houses=10, m=3, seed=0)
        model = make_model()(data.x.size(1))
        node = int(data.house_anchors[0])
        expl = Explainer(algorithm=GNNExplainer(epochs=15, lr=0.01)).explain_node(
            data, model, node
        )
        return data, expl

    def test_describe(self):
        data, expl = self._setup()
        text = describe(expl, data=data)
        assert "nodo" in text
        assert "clase" in text
        assert "relevantes" in text
        assert text == describe(expl, data=data)  # determinista

    def test_narrate_without_llm(self):
        data, expl = self._setup()
        text = narrate(expl, data=data)
        assert text == describe(expl, data=data)

    def test_narrate_with_llm(self):
        data, expl = self._setup()
        calls = []

        def fake_llm(prompt):
            calls.append(prompt)
            return "NARRACIÓN PERSONALIZADA"

        text = narrate(expl, llm=fake_llm, data=data)
        assert text == "NARRACIÓN PERSONALIZADA"
        assert calls and "JSON" in calls[0]

    def test_narrate_fallback(self):
        _, expl = self._setup()

        def broken_llm(_prompt):
            raise RuntimeError("sin API")

        text = narrate(expl, llm=broken_llm)
        assert "LLM no disponible" in text
        assert "Explicación del nodo" in text

    def test_counterfactual_describe(self):
        torch.manual_seed(0)
        data = build_data(base_nodes=50, num_houses=8, m=3, seed=0)
        model = make_model()(data.x.size(1))
        expl = Explainer(
            algorithm=Counterfactual(mode="edge", max_steps=5, hops=2)
        ).explain_node(data, model, int(data.house_anchors[0]))
        text = describe(expl, data=data)
        assert "cambiar la predicción" in text

    def test_narrator_class(self):
        _, expl = self._setup()
        narrator = Narrator(llm=None)
        assert narrator.describe(expl) == describe(expl)
