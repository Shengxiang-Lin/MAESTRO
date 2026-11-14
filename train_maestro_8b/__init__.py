"""MAESTRO 8B risk pretraining; GRPO-compatible PEFT policy checkpoint."""

__all__ = ["WindowRiskModel"]

def __getattr__(name):
    if name == "WindowRiskModel":
        from .risk_model import WindowRiskModel
        return WindowRiskModel
    raise AttributeError(name)
