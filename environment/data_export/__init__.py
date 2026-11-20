"""Canonical MAESTRO paired-case and grouped-split dataset exporter.

Lazy public re-export avoids loading the private-context CLI before
`python -m environment.data_export.private_context` executes it.
"""

__all__ = ["export_dataset"]


def __getattr__(name: str):
    if name == "export_dataset":
        from .export import export_dataset
        return export_dataset
    raise AttributeError(name)
