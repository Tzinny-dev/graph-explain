<!--
Gracias por contribuir a graph-explain!
Antes de abrir el PR, revisa CONTRIBUTING.md. Los checks de CI deben pasar.
-->

## Descripción

<!-- Qué cambia este PR y por qué. Referencia issues con "Closes #123". -->

## Tipo de cambio

- [ ] 🐛 Bug fix (cambio que corrige un issue)
- [ ] ✨ Nueva feature (método de explicación, métrica, backend, etc.)
- [ ] 🔧 Refactor (no cambia comportamiento)
- [ ] 📝 Documentación
- [ ] 🧪 Tests
- [ ] 🏗️ CI / build

## Checklist

- [ ] `ruff check src tests examples` y `ruff format --check src tests examples` pasan
- [ ] `python -m pytest -q` pasa localmente
- [ ] Tests nuevos/actualizados cubren el cambio
- [ ] Si agrega un método: sigue la guía de CONTRIBUTING.md
      (registro con `@register`, aliases en CLI, export en `__init__`, docs en `docs/api.rst` y `README.md`)
- [ ] Si agrega strings de narración: están en ambos idiomas (`_TEMPLATES["es"]` y `_TEMPLATES["en"]` en `narration/narrator.py`)
- [ ] Cambios de user-facing documentados en `CHANGELOG.md` (sección `[Unreleased]`)
- [ ] Si cambia la API pública: actualizó docstrings Google-style y el README

## ¿Cómo se probó?

<!-- Describe los pasos/scenarios probados. Incluye node-level y graph-level si aplica. -->

```python
# Snippet de verificación (opcional pero ayuda mucho)
```
