# Changelog

All notable versions of `graph-explain`.

## [0.7.0] - 2026-09-04

### Phase 10: graph-level
- Synthetic graph-classification dataset `build_graph_classification` (house
  motif, binary `y`, per-graph `gt_edge_mask`/`gt_nodes`).
- `evaluate_gea_graph`: graph-level GEA over the motif edges.
- `explain_graph` in the CLI and `bench` without `--node` for `task_level =
  "graph"` models; per-method `graph_level` flag (node-only methods are marked
  `skipped`).
- Example graph-level model and training (`GraphGCN`, `train_graph`).
- `Saliency` and `IntegratedGradients` accept `index=None` (graph-level).

## [0.6.0] - 2026-09-04

### Phase 8: more methods
- `GraphLIME` (`graph_lime`/`glime`/`gl`): local linear (ridge) regression over
  k-hop neighbors' features weighted by similarity; feature and node importance
  without training.
- `NodeMask` (`node_mask`/`nodemask`/`nm`): node mask learned by optimization
  over the k-hop subgraph (CE + entropy + top-k suppression).
- `GuidedBackprop` (`guided_backprop`/`guided-backprop`/`gbp`): gradients guided
  by the ReLU mask with fallback to standard gradients.
- `RandomBaseline` (`random`/`random_baseline`/`rand`): seed-able uniform
  baseline for benchmarks.
- Packaging: expanded project metadata (classifiers, `dev` extra).

### Phase 7: comparative benchmark
- `compare(...)` and `report_html(...)` to compare all methods over a node with
  a metric battery (fid±, GEA, sparsity, stability).
- CLI `bench` subcommand (terminal table, JSON/HTML reports).

## [0.5.0] - 2026-04-04

### Phase 6: more methods
- `DeepLift` (additive rescale rule, conservative, zero baseline).
- `AttentionExplainer` (`GATConv` attention weights).
- `GradXInput` (gradient × activation).

## [0.4.0] - 2025-01-01

### Phase 5: full CLI
- `explain` subcommand for all methods (including aliases), metrics, narration
  and JSON export.
- Unified version in `pyproject.toml` and `graph_explain.__version__`.

## [0.3.0] - 2024-01-01

### Phases 3 and 4
- Full metrics (fidelity±, stability, GEA) and DGL backend.
- `GNNGatedLRP`, counterfactual explanations and LLM narration.
- Synthetic BA-Shapes benchmark with ground truth.

## [0.2.0] - 2023-01-01

### Phase 2
- `PGExplainer`, `SubgraphX`, `Integrated Gradients`.

## [0.1.0] - 2023-01-01

### Phase 1
- Core (`Explainer`, `Explanation`, registry, backends).
- `GNNExplainer`, `Saliency`, static and interactive visualization.