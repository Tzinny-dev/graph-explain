Graph-Explain
==============

Librería de explicabilidad para modelos basados en grafos (GNN).

Características
---------------

- API unificada: un solo objeto ``Explainer`` para todos los métodos.
- 15 algoritmos: GNNExplainer, PGExplainer, SubgraphX, Saliency, Integrated
  Gradients, GNNGatedLRP, DeepLift, Attention/GAT, GradXInput, GraphLIME,
  NodeMask, GuidedBackprop, Random baseline y Counterfactual.
- Explicaciones **node-level** y **graph-level** (graph classification).
- Métricas: fidelity+ / fidelity-, GEA (node y graph), sparsity y stability.
- Backends PyTorch Geometric y DGL, visualización estática e interactiva,
  narración en lenguaje natural y CLI completa con benchmark comparativo.

Contenido
---------

.. toctree::
   :maxdepth: 2

   api

Guía rápida
-----------

.. code-block:: python

   from graph_explain import Explainer, GNNExplainer, describe

   # node-level
   expl = Explainer(algorithm=GNNExplainer(epochs=120)).explain_node(
       data, model, node_idx=42)
   print(describe(expl, data=data))

   # graph-level (modelos con task_level="graph")
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

   # graph-level: omitir --node
   graph-explain explain --model model.pt --data graph.pt --method grad_x_input