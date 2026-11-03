"""Shared error type for source-semantic synthesis invariants."""
class ScenarioError(ValueError):
    """A certified source-synthesis invariant was violated."""
__all__=["ScenarioError"]
