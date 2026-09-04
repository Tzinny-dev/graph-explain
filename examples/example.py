from __future__ import annotations

import os
import warnings

import torch

from examples.model import default_data_and_model, train
from graph_explain import (
    Explainer,
    GNNExplainer,
    IntegratedGradients,
    PGExplainer,
    Saliency,
    SubgraphX,
)
from graph_explain.visualization import show, visualize_interactive

warnings.filterwarnings("ignore")


def main():
    torch.manual_seed(0)
    data, model = default_data_and_model(num_houses=40)

    acc = train(model, data, epochs=600)
    print(f"Test accuracy: {acc:.3f}")

    with torch.no_grad():
        pred = model(data.x, data.edge_index).argmax(dim=-1)
    class2 = (data.y == 2).nonzero(as_tuple=False).flatten()
    correct = class2[pred[class2] == data.y[class2]]
    anchor = int(correct[0].item() if correct.numel() else class2[0].item())
    print(f"Explaining house node (class 2) index={anchor}")

    methods = [
        ("GNNExplainer", GNNExplainer(epochs=120, lr=0.01)),
        ("PGExplainer", PGExplainer(epochs=120, lr=0.01)),
        ("SubgraphX", SubgraphX(rollout=20, num_hops=3, max_nodes=15)),
        ("Saliency", Saliency()),
        ("IntegratedGradients", IntegratedGradients(steps=25)),
    ]

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    for name, algo in methods:
        explainer = Explainer(
            algorithm=algo,
            mask_threshold=0.4 if not isinstance(algo, SubgraphX) else 0.5,
        )
        expl = explainer.explain_node(data, model, node_idx=anchor)
        metrics = expl.evaluate(metrics=["fidelity", "sparsity"]) if expl.prediction_explanation is not None else expl.evaluate(metrics=["sparsity"])
        local = expl.evaluate(metrics=["sparsity"], local=True)
        print(f"{name:20s} -> {expl}")
        print(f"{'':20s}    métricas: {metrics} | local: {local}")

    expl = Explainer(algorithm=GNNExplainer(epochs=120)).explain_node(data, model, node_idx=anchor)
    visualize_interactive(expl, output_path=f"{output_dir}/explicacion.html", threshold=0.4)
    show(expl, threshold=0.4, show_labels=True, title=f"GNNExplainer (nodo {anchor}, clase 2)")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()