"""Learnable MAESTRO Main-Agent components."""

from .schema import ORCH_TOKEN

__all__ = ["ORCH_TOKEN", "MaestroOrchestratorModel"]


def __getattr__(name):
    if name == "MaestroOrchestratorModel":
        from .modeling import MaestroOrchestratorModel
        return MaestroOrchestratorModel
    raise AttributeError(name)
