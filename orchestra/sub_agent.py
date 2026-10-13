"""Backward-compatible SubAgent import proxy."""
from orchestra.subagents.react_agent import ReActAgent

OrchestraSubAgent = ReActAgent
__all__ = ["OrchestraSubAgent", "ReActAgent"]
