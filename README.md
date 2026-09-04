# graph-explain

Explainability library for graph-based models (Graph Neural Networks).
Explains a GNN's predictions in terms of **important nodes, edges and subgraphs**,
with built-in metrics and visualization.

## Features

- **Unified API**: a single `Explainer` object for every method.
- **Node-level and graph-level**: `explain_node(...)` explains a node's
  prediction; `explain_graph(...)` (or CLI without `--node`) explains a whole
  graph with graph-level models (`task_level = "graph"`), including GEA
  graph-level metrics and comparative benchmarking.
- **Explanation methods**:
  - `GNNExplainer` — soft masks over nodes/edges (perturbation).
  - `PGExplainer` — MLP generating edge masks (inductive, fast at inference).
  - `SubgraphX` — MCTS search for subgraphs that maximize the prediction (high fidelity).
  - `Saliency` — gradient-based importance.
  - `Integrated Gradients` — gradient accumulation vs. a baseline (attribution paths).
  - `GNNGatedLRP` — layer-wise relevance propagation (LRP-0/z+) over GCNs;
    distributes relevance between nodes and edges from the positive contributions
    of each conv/linear layer; supports `GCNConv` + `ReLU` + `Linear`.
  - `DeepLift` — additive rescale rule vs. a (zero) baseline: each feature gets a
    contribution proportional to its effect on the target class; conservative
    (contributions sum ≈ Δ logits); supports `GCNConv` + `ReLU` + `Linear`.
  - `AttentionExplainer` — node/edge importance from a `GATConv` model's
    attention weights (softmax per neighbor, averaged over heads and layers).
  - `GradXInput` — gradient × activation (zero baseline) for nodes and edges.
  - `GraphLIME` — local linear (ridge) regression over the k-hop neighbors'
    features, weighted by similarity to the target node; gives directly
    interpretable feature importance without training.
  - `NodeMask` — node mask learned by optimization (tracking the prediction)
    over the k-hop subgraph, regularized toward sparsity.
  - `GuidedBackprop` — gradients guided by the ReLU mask (positive activations
    only); falls back to standard gradients if the model uses functional ReLUs.
  - `Random` — uniformly random importance baseline (seed-able) for benchmarks.
  - `Counterfactual` — minimal perturbation (edges or features) that changes a
    node's prediction (deterministic greedy search); returns the modified
    elements as importance plus the logits after the change.
- **Narration**: `describe(expl)` builds a template-based natural-language
  explanation (Spanish by default), and `narrate(expl, llm=...)` lets you plug
  in a generative model (a `prompt -> text` callable) for free-form text.
- **Metrics**:
  - `evaluate_sparsity` — global or local sparsity (`local=True`, over the k-hop subgraph).
  - `evaluate_fidelity_plus` — **necessity**: drop in `P(c)` when removing the top-k elements.
  - `evaluate_fidelity_minus` — **sufficiency**: `P(c)` preserved when keeping ONLY the top-k.
  - `evaluate_stability` — mean similarity between explanations under feature/edge perturbations.
  - `evaluate_gea` — **Graph Explanation Accuracy**: overlap of the top-k with the ground-truth subgraph (BA-Shapes).
- **Built-in benchmarks**: BA-Shapes synthetic generator with ground truth and
  `ground_truth_nodes` / `ground_truth_edge_ids` helpers; in addition,
  `build_graph_classification` builds a **graph classification** dataset (house
  motif) with per-graph `gt_edge_mask` for graph-level GEA
  (`evaluate_gea_graph`).
- **Visualization**: static (matplotlib + networkx) and interactive (pyvis → HTML).
- **Backends**: PyTorch Geometric and DGL (through an adapter; DGL requires a
  PyTorch version with pre-built graphbolt libraries).
- **CLI** to explain saved models without writing code, plus a **comparative
  benchmark** of all methods over a node (table, JSON and HTML).
- **Programmatic comparison**: `compare(...)` to evaluate and compare methods.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .[all]
```

Optional extras: `pyg` (PyTorch Geometric), `dgl` (DGL backend),
`interactive` (plotly/pyvis).

## Quick start

```python
from graph_explain import Explainer, GNNExplainer, Saliency
from graph_explain.benchmarks.synthetic import build_data
from graph_explain.visualization import show

data = build_data(base_nodes=300, num_houses=80)   # BA-Shapes with ground truth
model = GCN(in_channels=data.x.size(1))            # your trained GNN
model.eval()

explainer = Explainer(algorithm=GNNExplainer(epochs=150))
expl = explainer.explain_node(data, model, node_idx=42)

print(expl.evaluate(metrics=["fidelity", "sparsity"]))
print(expl.evaluate(metrics=["sparsity"], local=True))  # sparsity over the node's k-hop subgraph
show(expl, show_labels=True)                       # highlight the explanatory subgraph
```

## Sparsity tuning notes

- **Structural models**: perturbation-based explainers (GNNExplainer,
  PGExplainer, SubgraphX) assume the prediction depends on the neighborhood
  structure. A `GCNConv` with `add_self_loops=True` and `bias=True` can predict
  the class from biases/self-loops alone; in that case edge masks collapse to
  zero because edges do not matter. For meaningful demos use
  `GCNConv(..., add_self_loops=False, bias=False)` (see `examples/model.py`).
- **Benchmark split**: `build_data` splits train/test across **all** nodes
  (including motifs). If the model is trained on class 0 only, it learns to
  ignore structure.
- **`PGExplainer(temp=...)`**: with `temp=5` the Gumbel-sigmoid sampling
  gradient flattens (~0.05) and the mask collapses to zero. The default is `temp=1.0`.
- **Local sparsity**: `evaluate_sparsity(expl, local=True)` measures sparsity
  over the explained node's `k-hop` subgraph instead of the whole graph; when
  the mask is counted over the full graph, local explanations get diluted
  (sparsity appears near 1).

## CLI

```bash
# Save model and data first:
torch.save(model, "model.pt"); torch.save(data, "data.pt")

graph-explain explain \
  --model model.pt --data data.pt \
  --method gnn_explainer --node 42 \
  --plot explicacion.png
```

## The `Explanation` object

- `node_importance`: importance per node `(num_nodes,)`.
- `edge_importance`: importance per edge.
- `feature_importance`: importance per feature (method-dependent).
- `prediction_original` / `prediction_explanation`: logits for fidelity evaluation.
- Methods: `evaluate(metrics=[...])`, `to_networkx(threshold=...)`, `visualize_static(...)`.

## Structure

```
src/graph_explain/
├── core/         # Explainer, Explanation, registry, evaluation
├── methods/      # gnn_explainer, subgraphx, pg_explainer, saliency, integrated_gradients
├── backends/     # Backend API + PyGAdapter + DGLAdapter
├── benchmarks/   # BA-Shapes synthetic generator + ground-truth helpers
└── visualization/ # static plots

```

`get_backend(name)` returns `PyGAdapter` or `DGLAdapter`. For DGL, features go
in `ndata['feat']`, labels in `ndata['label']` and edge weights in `edata['w']`;
the model must read `g.ndata['feat']` and `g.edata['w']`.

**DGL validation against the real library**: DGL 2.1.0 only ships graphbolt C++
libraries for torch ≤ 2.2.1, so the real integration is tested in an isolated
virtual machine (`tests/test_dgl_integration.py`, skipped when dgl is not
available):

```bash
python3.12 -m venv /tmp/dgl-venv
/tmp/dgl-venv/bin/pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cpu \
    dgl==2.1.0 "numpy<2" "scipy<1.14" "pandas" "torchdata==0.7.1" \
    "torch-geometric==2.6.1" setuptools packaging
cd graph-explain && PYTHONPATH=. /tmp/dgl-venv/bin/python -m pytest tests -q
```

## Metrics (phase 3)

````python
from graph_explain.core.evaluation import (
    evaluate_fidelity_plus, evaluate_fidelity_minus,
    evaluate_stability, evaluate_gea,
)

fp = evaluate_fidelity_plus(model, expl)    # necessity: remove top-k elements → P(c) drops
fm = evaluate_fidelity_minus(model, expl)   # sufficiency: keep only top-k → P(c) is preserved
stab = evaluate_stability(
    lambda d: Explainer(algorithm=GNNExplainer(epochs=40)).explain_node(d, model, node_idx=42),
    data, num_perturbations=5, noise_std=0.02,
)
gea = evaluate_gea(expl, data=data)         # overlap with the BA-Shapes motif
```

Example in `examples/example.py`, benchmark with `num_houses=30`: GNNExplainer →
`fid+ 0.74 / fid- 0.99 / GEA 0.92 / stab 0.85`.
````

## CLI (phase 5)

The command-line interface covers all methods (including the `lrp`/`gnn_lrp`
and `cf`/`counterfactual` aliases), metrics, narration and JSON reports:

```bash
graph-explain --version

# Counterfactual explanation for node 42 + narration + JSON report
graph-explain explain --model model.pt --data data.pt \
    --method counterfactual --node 42 --mode feature \
    --hops 2 --max-steps 10 --describe --json report.json

# Normalized GNN-LRP with metrics
graph-explain explain --model model.pt --data data.pt \
    --method lrp --node 42 --normalize \
    --metrics fidelity_plus,fidelity_minus,gea,stability \
    --top-k 5 --num-perturbations 5

# GNNExplainer + static and interactive visualizations
graph-explain explain --model model.pt --data data.pt \
    --method gnn_explainer --node 42 --epochs 200 \
    --threshold 0.5 --plot expl.png --html expl.html
```

Main options: `--method`, `--node`, `--target-class`, `--epochs`, `--lr`,
`--mode` (edge/feature), `--hops`, `--max-steps`, `--eps`, `--steps`,
`--normalize`, `--backend` (pyg/dgl), `--threshold`, `--top-k`, `--metrics`,
`--num-perturbations`, `--noise-std`, `--describe`, `--json`, `--output`,
`--plot`, `--html`. The JSON report includes method, predictions, metrics and
the structured summary (`summarize`) with top-k nodes/edges.

## Comparative benchmark (phase 7)

`compare(data, model, node=...)` runs every method on a node, computes the
metric battery (fid+ / fid- / GEA / sparsity / stability) and returns a
structured dict; non-applicable methods (e.g. Attention without `GATConv`) and
meaningless metrics are marked as `skipped`/`None` without aborting the rest:

```python
from graph_explain import compare, report_html

results = compare(data, model, node=42, methods=None,   # None = all
                  num_perturbations=5, epochs=200)
report_html(results, "bench.html")                      # self-contained HTML report
```

The CLI ships an equivalent subcommand:

```bash
graph-explain bench --model model.pt --data data.pt --node 42 \
    --methods all --num-perturbations 5 \
    --json bench.json --html bench.html
```

Note: `gea` is only defined when the node belongs to a ground-truth subgraph of
the benchmark (BA-Shapes); otherwise it shows up empty in the table.

## Graph-level (phase 10)

Models that predict over whole graphs (`task_level = "graph"`, e.g. GCN +
global pooling). Without `--node`, the CLI explains the whole graph; methods
marked with `graph_level`:

```bash
# Explain a whole graph (graph-level model) + GEA over the motif
graph-explain explain --model model.pt --data graph.pt \
    --method grad_x_input --metrics fidelity_plus,gea

# Graph-level bench (shows skipped methods and only runs applicable ones)
graph-explain bench --model model.pt --data graph.pt \
    --methods all --no-stability --json bench_graph.json
```

In Python:

```python
from graph_explain import Explainer, evaluate_gea_graph
from graph_explain.benchmarks.synthetic import build_graph_classification

graphs = build_graph_classification(num_pos=8, num_neg=8, seed=0)  # binary y, gt_edge_mask
model = ...  # GraphGCN (task_level="graph")

expl = Explainer(algorithm=GradXInput()).explain_graph(graphs[0], model)
print(evaluate_gea_graph(expl, data=graphs[0], top_k=13))
```

Node-only methods (`GraphLIME`, `NodeMask`, `Attention`, `GNNGatedLRP`,
`Counterfactual`, `DeepLift`, `PGExplainer`, `SubgraphX`) are marked as
`skipped` at graph-level.

## Roadmap

- [x] Phase 2: PGExplainer, SubgraphX, Integrated Gradients
- [x] Phase 2: interactive visualization (pyvis → HTML)
- [x] Phase 3: full metrics (fidelity±, stability, GEA)
- [x] Phase 3: DGL backend (adapter; integration validated with DGL 2.1 + torch 2.2.1)
- [x] Phase 4: GNN-LRP (layer-wise relevance for GCNs; validates the house motif in BA-Shapes)
- [x] Phase 4: counterfactual explanations (minimal edge/feature removal that changes the class)
- [x] Phase 4: LLM narration (`describe` deterministic + pluggable `narrate` LLM)
- [x] Phase 5: full CLI (all methods, metrics, narration and JSON export)
- [x] Phase 6: more methods (DeepLIFT rescale, Attention/GAT, Gradient×Input)
- [x] Phase 7: comparative benchmark (`compare` + CLI `bench` subcommand, table and JSON/HTML reports)
- [x] Phase 8: more methods (GraphLIME, NodeMask, GuidedBackprop and Random baseline)
- [x] Phase 10: graph-level explanations (graph-classification dataset with house
  motif, graph-level GEA, CLI/bench without `--node` and `graph_level` flag)

## License

MIT