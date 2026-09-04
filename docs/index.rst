Graph-Explain
==============

An explainability library for graph-based models (GNNs).

Features
--------

- Unified API: a single ``Explainer`` object for every method.
- 15 algorithms: GNNExplainer, PGExplainer, SubgraphX, Saliency, Integrated
  Gradients, GNNGatedLRP, DeepLift, Attention/GAT, GradXInput, GraphLIME,
  NodeMask, GuidedBackprop, Random baseline and Counterfactual.
- **Node-level** and **graph-level** (graph classification) explanations.
- Metrics: fidelity+ / fidelity-, GEA (node and graph), sparsity and stability.
- PyTorch Geometric and DGL backends, static and interactive visualization,
  natural-language narration and a full CLI with comparative benchmarking.

Contents
--------

.. toctree::
   :maxdepth: 2

   api

Quick start
-----------

.. code-block:: python

   from graph_explain import Explainer, GNNExplainer, describe

   # node-level
   expl = Explainer(algorithm=GNNExplainer(epochs=120)).explain_node(
       data, model, node_idx=42)
   print(describe(expl, data=data))

   # graph-level (models with task_level="graph")
   expl_g = Explainer(algorithm=GradXInput()).explain_graph(graph, graph_model)
   from graph_explain import evaluate_gea_graph
   gea = evaluate_gea_graph(expl_g, data=graph, top_k=13)

CLI
---

.. code-block:: bash

   graph-explain explain --model model.pt --data data.pt \
       --method gnn_explainer --node 42 --describe --json report.json
   graph-explain bench --model model.pt --data data.pt --node 42 \
       --json bench.json --html bench.html

   # graph-level: omit --node
   graph-explain explain --model model.pt --data graph.pt --method grad_x_input