from __future__ import annotations

import os
import warnings

import torch

from examples.model import default_data_and_model, train
from graph_explain import (
    AttentionExplainer,
    Counterfactual,
    DeepLift,
    Explainer,
    GNNExplainer,
    GNNGatedLRP,
    GradXInput,
    GraphLIME,
    GuidedBackprop,
    IntegratedGradients,
    NodeMask,
    PGExplainer,
    RandomBaseline,
    Saliency,
    SubgraphX,
    describe,
)
from graph_explain.core.evaluation import (
    evaluate_fidelity_minus,
    evaluate_fidelity_plus,
    evaluate_gea,
    evaluate_stability,
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
        ("GNN-LRP", GNNGatedLRP(normalize=True)),
        ("DeepLIFT", DeepLift()),
        ("Attention", AttentionExplainer()),
        ("GradXInput", GradXInput()),
        ("GraphLIME", GraphLIME()),
        ("NodeMask", NodeMask(epochs=120)),
        ("GuidedBackprop", GuidedBackprop()),
        ("Random", RandomBaseline(seed=0)),
        ("Contrafactual", Counterfactual(mode="edge", max_steps=8, hops=2)),
        ("Saliency", Saliency()),
        ("IntegratedGradients", IntegratedGradients(steps=25)),
    ]

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    for name, algo in methods:
        mask_threshold = (
            0.05
            if isinstance(algo, GNNGatedLRP)
            else 0.4
            if not isinstance(algo, SubgraphX)
            else 0.5
        )
        explainer = Explainer(algorithm=algo, mask_threshold=mask_threshold)
        try:
            expl = explainer.explain_node(data, model, node_idx=anchor)
        except ValueError as exc:
            print(f"{name:20s} -> no aplicable aquí ({exc})")
            continue
        metrics = (
            expl.evaluate(metrics=["fidelity", "sparsity"])
            if expl.prediction_explanation is not None
            else expl.evaluate(metrics=["sparsity"])
        )
        local = expl.evaluate(metrics=["sparsity"], local=True)
        print(f"{name:20s} -> {expl}")
        print(f"{'':20s}    métricas: {metrics} | local: {local}")

    expl = Explainer(algorithm=GNNExplainer(epochs=120)).explain_node(
        data, model, node_idx=anchor
    )
    print(f"GNNExplainer completo -> {expl}")
    print(f"Narración: {describe(expl, data=data)}")
    print(
        f"{'':20s}    fid+= {evaluate_fidelity_plus(model, expl):+.3f}  "
        f"fid-= {evaluate_fidelity_minus(model, expl):.3f}  "
        f"GEA= {evaluate_gea(expl, data=data):.3f}  "
        f"stab= {evaluate_stability(lambda d: Explainer(algorithm=GNNExplainer(epochs=40)).explain_node(d, model, node_idx=anchor), data, num_perturbations=5, noise_std=0.02, seed=3):.3f}"
    )
    visualize_interactive(
        expl, output_path=f"{output_dir}/explicacion.html", threshold=0.4
    )
    show(
        expl,
        threshold=0.4,
        show_labels=True,
        title=f"GNNExplainer (nodo {anchor}, clase 2)",
    )

    graph_example(output_dir)


def graph_example(output_dir: str):
    """Explicación graph-level: clasificación de grafos con motivo house."""
    try:
        from examples.model import GraphGCN, train_graph
    except ImportError:
        from model import GraphGCN, train_graph

    from graph_explain import evaluate_gea_graph
    from graph_explain.benchmarks.synthetic import build_graph_classification
    from graph_explain.core.registry import instantiate

    graphs = build_graph_classification(num_pos=8, num_neg=8, seed=0)
    model = GraphGCN(graphs[0].x.size(1))
    acc = train_graph(model, graphs, epochs=200)
    print(f"\n[Graph-level] accuracy: {acc:.2f}")

    positive = graphs[0]
    for name, algo in (
        ("Saliency", instantiate("saliency")),
        ("GradXInput", instantiate("grad_x_input")),
        ("Random", instantiate("random", seed=0)),
    ):
        expl = Explainer(algorithm=algo).explain_graph(positive, model)
        gea = None
        if expl.edge_importance is not None:
            try:
                gea = evaluate_gea_graph(expl, data=positive, top_k=13)
            except ValueError:
                gea = None
        print(
            f"{name:12s} grafo -> nodos {expl.node_importance.shape[0]}, "
            f"aristas {expl.edge_importance.shape[0] if expl.edge_importance is not None else '-'}, "
            f"GEA(graph)={gea if gea is None else f'{gea:.3f}'}"
        )


if __name__ == "__main__":
    main()
