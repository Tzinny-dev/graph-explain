# Changelog

All notable versions of `graph-explain`.

## [0.8.0] - 2026-09-22

### Static analysis with CodeQL
- New `codeql` job in the `ci` workflow: runs on every push to `main` and
  every pull request, analyzing `src/` with the `security-extended` query
  suite. Results are uploaded to the **Security** tab as SARIF and fail the
  build on any high-severity finding in our own code.
- Standalone `.github/workflows/codeql.yml` runs the same analysis on a
  weekly Sunday schedule (cron `0 6 * * 0`) to catch drift.
- Complements the already-enabled Dependabot security updates and secret
  scanning.

### Static typing (mypy)
- `mypy` config in `pyproject.toml` (`[tool.mypy]`): `src` layout,
  `ignore_missing_imports` for untyped ML libs, explicit excludes for
  `build/` and untyped viz narrators.
- Typing fixes across the codebase: typed `ExplanationAlgorithm.explain`
  protocol (`index: Any = None`), `type[ExplanationAlgorithm]` in the
  registry, `dict[str, Any]` benchmark entries, `NDArray[np.integer]`
  separation from torch tensors in `synthetic.py`, `RemovableHandle` hook
  lists, empty-list returns in evaluation BFS helpers and defensive casts
  for `nn.Module` attribute access in LRP/DeepLift.
- `mypy` runs on every CI push/PR (Python 3.12) as part of the lint job.

### Repository hygiene
- `pre-commit` hooks (`ruff-check --fix`, `ruff-format`, hygiene checks:
  trailing whitespace, EOF newline, YAML/TOML validity, merge-conflict
  markers, large-file guard) with pinned hook revisions.
- `dependabot.yml`: weekly grouped updates for pip dependencies (`build: ...`)
  and GitHub Actions (`ci: ...`); security-only updates stay on Dependabot
  security alerts.
- Newline-at-EOF fixes across tracked files and import sorting in
  `docs/conf.py`; its version hook now catches only `ImportError`.

### Coverage in CI
- New dedicated `coverage` CI job: branch coverage on Python 3.12 with a
  `fail_under = 80` threshold enforced from `pyproject.toml`
  (`[tool.coverage.report]`). Locally, `python -m pytest --cov=graph_explain`
  picks up the same config.
- `pytest-cov` added to the `dev` extra; standard excludes for
  `TYPE_CHECKING`, `__main__`, import fallbacks and abstract stubs.
- Coverage artifacts (`coverage.xml`/`coverage.json`) uploaded on every CI run
  and a self-hosted `coverage.svg` badge published to the docs site
  (`/_static/badges/coverage.svg`), rendered on the docs landing page and the
  README.
- Test results (`pytest-results.xml`) are now uploaded as artifacts for all
  Python versions in the matrix.

## [Unreleased]

## [0.7.3] - 2026-09-21

### Documentation and CI
- Sphinx documentation is now published to GitHub Pages at
  https://tzinny-dev.github.io/graph-explain/ through a dedicated `docs`
  workflow (`build` + `deploy` jobs using `upload-pages-artifact` and
  `deploy-pages`), triggered by pushes to `main` that touch `docs/`, `src/`,
  `pyproject.toml` or the workflow itself.
- `docs/conf.py` mocks `torch`/`torch_geometric`/`dgl` with
  `autodoc_mock_imports`, so the API reference builds without the full ML stack
  and the pages show the real package version.
- `publish` workflow reduced to PyPI upload + GitHub Release (docs deployment
  moved out to its own workflow).
- GitHub Releases now use the matching `CHANGELOG.md` section as their body
  (falling back to a link to the changelog when the version has no entry yet).
- `Documentation` project URL now points to the published site.

## [0.7.2] - 2026-09-05

### Narration and CI
- Bilingual narration: `describe`/`narrate`/`Narrator` accept
  `lang="es"|"en"` (Spanish default); English templates, LLM prompt and fallback.
- `all` extra now includes `dgl`.
- CI: real DGL integration job (torch 2.2.1 + DGL 2.1.0).

## [0.7.1] - 2026-09-05

### Packaging
- `torch` moved to base dependencies so `pip install graph-explain` imports
  out of the box (PyG remains an optional extra).
- Added `LICENSE` (MIT) packaged via `license-files`, `py.typed` marker for
  type checkers, and `[project.urls]` (Homepage/Repository/Documentation).
- `description` metadata translated to English.

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
