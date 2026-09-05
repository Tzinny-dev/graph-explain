# Contributing

Thanks for contributing to `graph-explain`!

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,pyg]"
```

Optional extras: `dgl` (DGL backend), `interactive` (plotly/pyvis),
`docs` (Sphinx), `publish` (twine).

## Commands

```bash
ruff check src tests examples        # lint
ruff format --check src tests examples  # formatting (or `ruff format` to apply)
python -m pytest -q                  # test suite
python -m sphinx -b html docs docs/_build/html  # build the docs
```

The `dgl` extra validates against the real DGL library, which only ships
pre-built graphbolt binaries for `torch<=2.2.1`. To run that validation
locally, create an isolated environment with `torch==2.2.1` + `dgl==2.1.0`
(see `README.md` for the exact command). CI runs it for you under `main`.

## Project structure

```
src/graph_explain/
├── core/         # Explainer, Explanation, registry, evaluation, benchmark
├── methods/      # one file per algorithm (gnn_explainer, saliency, ...)
├── backends/     # Backend API + PyGAdapter + DGLAdapter
├── benchmarks/   # synthetic generators + ground-truth helpers
├── narration/    # bilingual (es/en) template narration + LLM hook
├── visualization/# static and interactive rendering
└── cli.py        # the `graph-explain` command-line interface
```

## Adding a new explanation method

1. Create `src/graph_explain/methods/<family>/<name>.py` subclassing
   `ExplanationAlgorithm`.
2. Implement `_explain_node` and, when the method is graph-capable,
   `_explain_graph`. Add the `@register("name", "alias", ...)` decorator and
   declare `graph_level = True` if it supports graph-level explanations.
3. Add its aliases to `_METHODS` in `src/graph_explain/cli.py`.
4. Export it in `src/graph_explain/__init__.py` and document it in
   `docs/api.rst` and `README.md`.
5. Add tests under `tests/` covering node-level and (if applicable)
   graph-level behaviour, CLI invocation, and the comparative benchmark.

Methods must work through the `Backend` abstraction (PyG and DGL): reach the
graph structure and predictions via backend methods instead of importing
PyG/DGL data types directly in the core paths.

## Releasing

Releases are driven by git tags. To publish `x.y.z`:

```bash
# bump version in pyproject.toml and src/graph_explain/__init__.py
git tag v0.7.2
git push origin main --follow-tags
```

The `publish` workflow builds the sdist/wheel, uploads to PyPI (needs the
`PYPI_API_TOKEN` repository secret) and creates the GitHub Release.

## Code style

- Lint and format are enforced by `ruff` (`line-length = 88`). Run
  `ruff format` before committing.
- Docstrings use Google-style conventions (see existing methods).
- Default narration language is `"es"`; new narration strings must be added
  to both `_TEMPLATES["es"]` and `_TEMPLATES["en"]` in `narration/narrator.py`.
- Keep the `Explanation` object as the single output contract.