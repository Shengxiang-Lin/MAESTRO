"""Orchestra-o1 Main-Agent baseline.

All generic orchestration mechanics live in :mod:`orchestra.base_main_agent`.
The O1 baseline only selects the original O1 prompt at the benchmark entrypoint.
"""

from orchestra.base_main_agent import (
    BaseMainAgent,
    build_model_pricing_table,
)


class MainAgent(BaseMainAgent):
    """Original Orchestra-o1 decision policy on top of the shared runtime."""

    pass


__all__ = ["MainAgent", "build_model_pricing_table"]
