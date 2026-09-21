project = "graph-explain"
author = "graph-explain contributors"
copyright = "2026, graph-explain contributors"

# Add src to path so autodoc can find the package
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
]

# Mock heavy ML imports so autodoc can import graph_explain without torch/dgl
autodoc_mock_imports = [
    "torch",
    "torch_geometric",
    "torch_geometric.nn",
    "torch_geometric.data",
    "dgl",
    "dgl.nn",
    "dgl.data",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

autodoc_member_order = "bysource"
autodoc_default_options = {
    "show-inheritance": True,
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}


def setup(app):
    try:
        from graph_explain import __version__

        app.config.version = __version__
        app.config.release = __version__
    except ImportError:
        pass
