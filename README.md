# graph-explain

Librería de explicabilidad para modelos basados en grafos (Graph Neural Networks).
Explica las predicciones de una GNN en términos de **nodos, aristas y subgrafos
importantes**, con métricas de evaluación y visualización integradas.

## Características

- **API unificada**: un solo objeto `Explainer` para todos los métodos.
- **Métodos de explicación**:
- `GNNExplainer` — máscaras suaves sobre nodos/aristas (perturbación).
  - `PGExplainer` — MLP que genera máscaras de aristas (inductive, rápido en inferencia).
  - `SubgraphX` — búsqueda MCTS de subgrafos que maximizan la predicción (alta fidelidad).
- `Saliency` — importancia basada en gradientes.
- `Integrated Gradients` — acumulación de gradientes con baseline (rutas de importancia).
- `GNNGatedLRP` — propagación de relevancia por capas (LRP-0/z+) sobre GCNs;
  reparte la relevancia entre nodos y aristas según las contribuciones positivas
  de cada capa conv/lineal; soporta `GCNConv` + `ReLU` + `Linear`.
- `Counterfactual` — perturbación mínima (aristas o features) que cambia la
  predicción de un nodo (búsqueda greedy determinista); devuelve los elementos
  modificados como importancia y los logits tras el cambio.
- **Métricas**:
  - `evaluate_sparsity` — esparcidad global o local (`local=True`, sobre el subgrafo k-hop).
  - `evaluate_fidelity_plus` — **necesidad**: caída de `P(c)` al eliminar los top-k elementos.
  - `evaluate_fidelity_minus` — **suficiencia**: `P(c)` conservada al quedarse SOLO con los top-k.
  - `evaluate_stability` — similitud media entre explicaciones ante perturbaciones de features/aristas.
  - `evaluate_gea` — **Graph Explanation Accuracy**: solape de los top-k con el subgrafo de ground truth (BA-Shapes).
- **Benchmarks incluidos**: generador sintético BA-Shapes con ground truth y
  helpers `ground_truth_nodes` / `ground_truth_edge_ids`.
- **Visualización** estática (matplotlib + networkx) e interactiva (pyvis → HTML).
- **Backends**: PyTorch Geometric y DGL (vía adaptador; DGL requiere una versión
  de PyTorch con librerías precompiladas de graphbolt).
- **CLI** para explicar modelos guardados sin escribir código.

## Instalación

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .[all]
```

Dependencias opcionales: `pyg` (PyTorch Geometric), `dgl` (backend DGL),
`interactive` (plotly/pyvis).

## Uso rápido

```python
from graph_explain import Explainer, GNNExplainer, Saliency
from graph_explain.benchmarks.synthetic import build_data
from graph_explain.visualization import show

data = build_data(base_nodes=300, num_houses=80)   # BA-Shapes con ground truth
model = GCN(in_channels=data.x.size(1))            # tu GNN entrenada
model.eval()

explainer = Explainer(algorithm=GNNExplainer(epochs=150))
expl = explainer.explain_node(data, model, node_idx=42)

print(expl.evaluate(metrics=["fidelity", "sparsity"]))
print(expl.evaluate(metrics=["sparsity"], local=True))  # esparcidad en el subgrafo k-hop del nodo
show(expl, show_labels=True)                       # resalta el subgrafo explicativo
```

## Notas de tuning de esparcidad

- **Modelos estructurales**: los explicadores por perturbación (GNNExplainer,
  PGExplainer, SubgraphX) asumen que la predicción depende de la estructura
  vecinal. Un `GCNConv` con `add_self_loops=True` y `bias=True` puede predecir la
  clase solo con sesgos/self-loops; en ese caso las máscaras de aristas colapsan a
  cero porque las aristas no importan. Para demostraciones con sentido usa
  `GCNConv(..., add_self_loops=False, bias=False)` (ver `examples/model.py`).
- **Split del benchmark**: `build_data` reparte train/test sobre **todos** los
  nodos (incluidos los motivos). Si el modelo solo se entrena con clase 0 aprende
  a ignorar la estructura.
- **`PGExplainer(temp=...)`**: con `temp=5` el gradiente del muestreo
  Gumbel-sigmoid se aplana (~0.05) y la máscara colapsa a cero. El valor por
  defecto es `temp=1.0`.
- **Esparcidad local**: `evaluate_sparsity(expl, local=True)` mide la esparcidad
  sobre el subgrafo `k-hop` del nodo explicado en lugar de todo el grafo; si la
  máscara se cuenta sobre el grafo completo, las explicaciones locales se diluyen
  (aparentan sparsity casi 1).

## CLI

```bash
# Guarda modelo y datos antes:
torch.save(model, "model.pt"); torch.save(data, "data.pt")

graph-explain explain \
  --model model.pt --data data.pt \
  --method gnn_explainer --node 42 \
  --plot explicacion.png
```

## Objeto `Explanation`

- `node_importance`: importancia por nodo `(num_nodes,)`.
- `edge_importance`: importancia por arista.
- `feature_importance`: importancia por característica (según método).
- `prediction_original` / `prediction_explanation`: logits para evaluar fidelidad.
- Métodos: `evaluate(metrics=[...])`, `to_networkx(threshold=...)`, `visualize_static(...)`.

## Estructura

```
src/graph_explain/
├── core/         # Explainer, Explanation, registry, evaluation
├── methods/      # gnn_explainer, subgraphx, pg_explainer, saliency, integrated_gradients
├── backends/     # Backend API + PyGAdapter + DGLAdapter
├── benchmarks/   # generador sintético BA-Shapes + helpers de ground truth
└── visualization/ # plots estáticos

```

`get_backend(name)` devuelve `PyGAdapter` o `DGLAdapter`. Para DGL, las
features van en `ndata['feat']`, las etiquetas en `ndata['label']` y los pesos de
arista en `edata['w']`; el modelo debe leer `g.ndata['feat']` y `g.edata['w']`.

## Métricas (fase 3)

````python
from graph_explain.core.evaluation import (
    evaluate_fidelity_plus, evaluate_fidelity_minus,
    evaluate_stability, evaluate_gea,
)

fp = evaluate_fidelity_plus(model, expl)    # necesidad: quitar top-k elementos → baja P(c)
fm = evaluate_fidelity_minus(model, expl)   # suficiencia: solo top-k → conserva P(c)
stab = evaluate_stability(
    lambda d: Explainer(algorithm=GNNExplainer(epochs=40)).explain_node(d, model, node_idx=42),
    data, num_perturbations=5, noise_std=0.02,
)
gea = evaluate_gea(expl, data=data)         # solape con el motivo BA-Shapes
```

Ejemplo en `examples/example.py`, benchmark con `num_houses=30`: GNNExplainer →
`fid+ 0.74 / fid- 0.99 / GEA 0.92 / stab 0.85`.
````

## Roadmap

- [x] Fase 2: PGExplainer, SubgraphX, Integrated Gradients
- [x] Fase 2: visualización interactiva (pyvis → HTML)
- [x] Fase 3: métricas completas (fidelity±, stability, GEA)
- [x] Fase 3: backend DGL (adaptador; validado con mock de la API dgl)
- [~/] Fase 4: GNN-LRP (relevancia por capas para GCNs; valida el motivo house en BA-Shapes)
- [~/] Fase 4: explicaciones contrafactuales (mínima eliminación de aristas/features que cambia la clase)
- [ ] Fase 4: narración con LLM

## Licencia

MIT