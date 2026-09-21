---
name: Bug report
about: Something in graph-explain is broken or behaves unexpectedly
title: '[bug] '
labels: bug, triage
assignees: ''

---

**Describe el bug**

Una descripción clara y concisa de qué pasó.

**Comando / código mínimo que reproduce el issue**

```python
# Código mínimo que reproduce el problema.
# Si es por CLI, pega el comando exacto:
# graph-explain explain --model model.pt --data data.pt --method gnn_explainer --node 42
```

**Salida completa (incluye traceback)**

```
# Pega aquí la salida completa, el traceback o el JSON reportado
```

**Entorno (por favor completa):**

- `graph-explain` version: `python -c "import graph_explain; print(graph_explain.__version__)"`
- Python version: `python --version`
- OS: (e.g. Ubuntu 22.04, macOS 14, Windows 11)
- Backend: (PyTorch Geometric / DGL / ambos)
- torch / torch_geometric / dgl versions: `pip list | grep -E "torch|dgl"`
- Cómo lo instalaste: (`pip install graph-explain`, `pip install graph-explain[all]`, desde fuente, etc.)

**Comportamiento esperado**

Qué esperabas que pasara en lugar de lo que ocurrió.

**Contexto adicional**

Capturas, datasets sintéticos que fallan, o cualquier cosa que ayude a
reproducirlo. Si la explicación es incorrecta (no un crash), describe por qué
crees que la atribución es errónea (qué nodos/aristas esperabas como relevantes).
