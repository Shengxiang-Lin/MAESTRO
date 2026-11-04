"""Structural regression tests independent of YAGO, vLLM and source data."""
from __future__ import annotations

import random
import unittest

from environment.task_construction.ir import ProgramIR, ProgramNode
from environment.task_construction.binding.graph_grounding import validate_program_structure
from environment.task_construction.binding.program_binder import BindingError
from environment.task_construction.deep_dependency.composer import (
    _attach_chain, _fragment, compose_deep_program,
)


class ChooseLast:
    def choice(self, items):
        return items[-1]


class ProgramIRDependencyTests(unittest.TestCase):
    def test_fragment_keeps_complete_ancestor_closure(self):
        # The old code picked a deep endpoint, then truncated the ancestor
        # traversal at max_depth, silently dropping dependencies.
        p = ProgramIR([
            ProgramNode(0, "FIND"),
            ProgramNode(1, "RELATE", (0,), ("<REL>", "forward")),
            ProgramNode(2, "RELATE", (1,), ("<REL>", "forward")),
            ProgramNode(3, "RELATE", (2,), ("<REL>", "forward")),
        ])
        f = _fragment(p, min_depth=2, max_depth=3, rng=ChooseLast())
        self.assertEqual([n.op for n in f], ["FIND", "RELATE", "RELATE"])
        self.assertTrue(all(dep < node.node_id for node in f for dep in node.dependencies))
        validate_program_structure(ProgramIR(f))

    def test_fragment_never_truncates_multi_root_join(self):
        p = ProgramIR([
            ProgramNode(0, "FIND"),
            ProgramNode(1, "FIND"),
            ProgramNode(2, "INTERSECT", (0, 1)),
        ])
        self.assertEqual(
            _fragment(p, min_depth=2, max_depth=4, rng=ChooseLast()),
            [],
        )

    def test_chaining_replaces_find_root_instead_of_orphaning_relate(self):
        fragment = [
            ProgramNode(0, "FIND", (), ("<ENTITY_OR_TYPE>",)),
            ProgramNode(1, "RELATE", (0,), ("<REL>", "forward")),
        ]
        chain = _attach_chain(fragment, fragment)
        self.assertEqual([n.op for n in chain], ["FIND", "RELATE", "RELATE"])
        self.assertEqual(chain[-1].dependencies, (1,))
        validate_program_structure(ProgramIR(chain))

    def test_composed_program_has_no_orphaned_relation_nodes(self):
        source = ProgramIR([
            ProgramNode(0, "FIND"),
            ProgramNode(1, "RELATE", (0,), ("<REL>", "forward")),
            ProgramNode(2, "RELATE", (1,), ("<REL>", "forward")),
            ProgramNode(3, "QUERY_NAME", (2,)),
        ])
        for seed in range(40):
            composed = compose_deep_program(
                [source] * 5,
                min_fragment_depth=2, max_fragment_depth=3,
                min_orchestration_depth=3, max_orchestration_depth=5,
                min_parallel_width=1, seed=seed,
            )
            validate_program_structure(composed)
            self.assertEqual(sum(n.op == "FIND" for n in composed.nodes), 1)
            self.assertTrue(all(
                len(n.dependencies) == 1 for n in composed.nodes
                if n.op == "RELATE"
            ))

    def test_binder_reports_malformed_relation_instead_of_index_error(self):
        malformed = ProgramIR([
            ProgramNode(0, "RELATE", (), ("<REL>", "forward")),
        ])
        with self.assertRaisesRegex(BindingError, "malformed_program:op=RELATE"):
            validate_program_structure(malformed)

    def test_binder_rejects_invalid_join_arity(self):
        malformed = ProgramIR([
            ProgramNode(0, "FIND"),
            ProgramNode(1, "INTERSECT", (0,)),
        ])
        with self.assertRaisesRegex(BindingError, "malformed_program:op=INTERSECT"):
            validate_program_structure(malformed)


if __name__ == "__main__":
    unittest.main()
