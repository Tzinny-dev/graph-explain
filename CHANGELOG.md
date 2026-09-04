# Changelog

Todas las versiones notables de `graph-explain`.

## [0.6.0] - 2026-09-04

### Fase 8: más métodos
- `GraphLIME` (`graph_lime`/`glime`/`gl`): regresión lineal local (ridge) sobre
  features de los vecinos k-hop ponderada por similaridad; importancia de
  features y nodos sin entrenamiento.
- `NodeMask` (`node_mask`/`nodemask`/`nm`): máscara de nodos aprendida por
  optimización sobre el subgrafo k-hop (CE + entropía + supresión top-k).
- `GuidedBackprop` (`guided_backprop`/`guided-backprop`/`gbp`): gradientes
  guiados por la máscara ReLU con fallback a gradiente estándar.
- `RandomBaseline` (`random`/`random_baseline`/`rand`): línea base uniforme
  seed-able para benchmarks.
- Empaquetado: metadata de proyecto ampliada (classifiers, extras `dev`).

### Fase 7: benchmark comparativo
- `compare(...)` y `report_html(...)` para comparar todos los métodos sobre un
  nodo con battery de métricas (fid±, GEA, sparsity, stability).
- Subcomando CLI `bench` (tabla en terminal, informes JSON/HTML).

## [0.5.0] - 2026-04-04

### Fase 6: más métodos
- `DeepLift` (regla rescale aditiva, conservativa, baseline zero).
- `AttentionExplainer` (pesos de atención de `GATConv`).
- `GradXInput` (gradiente × activación).

## [0.4.0] - 2025-01-01

### Fase 5: CLI completa
- Subcomando `explain` para todos los métodos (aliases incluidos), métricas,
  narración y export JSON.
- Versión unificada en `pyproject.toml` y `graph_explain.__version__`.

## [0.3.0] - 2024-01-01

### Fases 3 y 4
- Métricas completas (fidelity±, stability, GEA) y backend DGL.
- `GNNGatedLRP`, explicaciones contrafactuales y narración con LLM.
- Benchmark sintético BA-Shapes con ground truth.

## [0.2.0] - 2023-01-01

### Fase 2
- `PGExplainer`, `SubgraphX`, `Integrated Gradients`.

## [0.1.0] - 2023-01-01

### Fase 1
- Núcleo (`Explainer`, `Explanation`, registry, backends).
- `GNNExplainer`, `Saliency`, visualización estática e interactiva.