---
name: Feature request
about: Proponer una idea o mejora para graph-explain
title: '[feature] '
labels: enhancement, triage
assignees: ''

---

**¿Problema a resolver?**

Describe el problema o necesidad. Ej: "Siempre me frustro cuando [...]"
o "No hay forma de [...]" — qué caso de uso no está cubierto hoy.

**Solución propuesta**

Descripción clara de lo que te gustaría que pasara.

**Alternativas consideradas**

Otras aproximaciones o librerías que hayas evaluado
(p.ej. Captum, PyTorch Geometric ExplainerBenchmark, DGL explainers) y
por qué no cubren el caso.

**Área afectada**

Marca las que apliquen:

- [ ] Nuevo método de explicación (`methods/`)
- [ ] Métricas (`core/evaluation`)
- [ ] Backend (PyG / DGL)
- [ ] Narración / descripción en lenguaje natural
- [ ] Visualización (estática / interactiva)
- [ ] CLI (`graph-explain ...`)
- [ ] Benchmarks sintéticos
- [ ] Documentación
- [ ] Otro

**Contexto adicional**

Referencias a papers, enlaces, mockups o ejemplos de la API que imaginas:

```python
# API ideal que imaginas (pseudocódigo está bien)
expl = Explainer(algorithm=MiMetodo(...)).explain_node(data, model, node_idx=42)
```
