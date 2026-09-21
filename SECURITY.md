# Security Policy

## Supported versions

Only the latest release line receives security fixes. Patch releases are
published to PyPI; check <https://pypi.org/project/graph-explain/> for the
current version.

| Version | Supported |
|---------|-----------|
| 0.7.x   | ✅        |
| < 0.7   | ❌        |

## Reporting a vulnerability

**Please do not report security issues through public GitHub Issues.**

Use one of these channels instead:

1. **GitHub Private Vulnerability Reporting** (preferred) —
   <https://github.com/Tzinny-dev/graph-explain/security/advisories/new>.
2. **Email** — [921charly@gmail.com](mailto:921charly@gmail.com)
   (please put `[security]` in the subject line).

Include as much of the following as you can:

- The affected version (`python -c "import graph_explain; print(graph_explain.__version__)"`).
- A minimal reproduction (script, model and data description).
- The impact you believe it has (e.g. arbitrary code execution via model
  loading, incorrect attribution leading to wrong conclusions).

## What to expect

- **Acknowledgement** within 7 days.
- An assessment and, when confirmed, a fix coordinated with you before any
  public disclosure.
- Credit in the release notes of the fix, if you wish.

## Scope notes

`graph-explain` is a research library. Two things are intentionally
out of scope for "vulnerabilities" but worth knowing:

- **Model/artifact loading**: the CLI loads models and data with
  `torch.load`/pickle. Loading untrusted checkpoints is unsafe by design;
  only explain models and graphs you trust. Hardening this is tracked, but
  treat `.pt` files as code.
- **Numerical output**: explanation attributions are heuristic estimates,
  not guarantees. Silent "wrong" attributions (e.g. a method producing
  misleading importance for an unsupported layer type) are bugs worth
  reporting, but are treated as correctness issues, not security ones.
