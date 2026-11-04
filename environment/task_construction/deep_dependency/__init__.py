"""Deep ProgramIR composition and fixed-pool executable-world generation."""
from .candidate_selection import generate_candidate_pool
from .composer import compose_candidate_stream,compose_deep_program
from .terminal_families import terminal_family,terminal_is_supported
from .world_generation import generate_world_batch
__all__=[
    "compose_candidate_stream","compose_deep_program",
    "generate_candidate_pool","generate_world_batch",
    "terminal_family","terminal_is_supported",
]
