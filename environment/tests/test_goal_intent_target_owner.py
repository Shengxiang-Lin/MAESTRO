"""Focused Stage2A intent checks based on real 50-task pilot regressions."""
import unittest

from environment.delegation_blueprint import (
    _fragment_goal_packet, _terminal_result_contract,
)
from environment.question_generation.goal_review import (
    deterministic_goal_issues,
)


def terminal(action, *, target_type=None, attribute=None):
    steps = [{
        "id": "S9", "operation": "INTERSECT",
        "inputs_from": ["S3", "S7"],
        "goal": "The same entities must satisfy all input branches.",
    }, {
        "id": "S10", "operation": action,
        "inputs_from": ["S9"],
        "goal": {
            "COUNT": "Count distinct matching entities.",
            "QUERY_NAME": "Return all matching entity NAMES.",
            "QUERY_ATTR": "Return the requested attribute values.",
        }[action],
    }]
    if attribute:
        steps[-1]["output_attribute"] = attribute
    task = {
        "fragment_id": "fragment:terminal", "depends_on": [],
        "output_kind": "value", "public_steps": steps,
    }
    # Public semantic query: the same source used by Stage2B, not a
    # private Gold result, user answer, or source observation.
    branch = {
        "definition": "INTERSECT", "conditions": [],
        "inputs": [{
            "definition": "FILTER_NUMBER",
            "conditions": [{"entity_type": target_type}] if target_type else [],
            "inputs": [],
        }, {
            "definition": "FILTER_DATE",
            "conditions": [{"entity_type": target_type}] if target_type else [],
            "inputs": [],
        }],
    }
    outline = {
        "tasks": [task],
        "task_semantics": {"definition": action, "inputs": [branch]},
        "public_modalities": [],
    }
    return outline, task


class IntentTargetOwnerTests(unittest.TestCase):
    def check(self, outline, task, goal):
        return deterministic_goal_issues(outline, task, goal)

    def test_count_entity_instead_of_number_is_rejected(self):
        outline, task = terminal("COUNT", target_type="Organization")
        self.assertIn(
            "terminal_count_intent_missing",
            self.check(
                outline, task,
                "Identify the single entity that simultaneously meets "
                "the organization criteria of the two branches.",
            ),
        )
        self.assertEqual(
            self.check(
                outline, task,
                "Count the distinct organizations satisfying the "
                "specified conditions in both branches.",
            ), [],
        )
        self.assertEqual(
            self.check(
                outline, task,
                "Find how many matching organizations the two branches share.",
            ), [],
        )

    def test_name_output_entity_type_cannot_be_swapped(self):
        outline, task = terminal("QUERY_NAME", target_type="Event")
        self.assertEqual(
            _terminal_result_contract(outline, task)["target_entity_type"],
            "Event",
        )
        wrong = (
            "Intersect the sport and emblem candidate groups, then "
            "return the names of any matching individuals."
        )
        self.assertIn(
            "terminal_wrong_result_entity_type:event",
            self.check(outline, task, wrong),
        )
        correct = (
            "Identify events appearing in both evidence branches, "
            "then return the names of those events."
        )
        self.assertEqual(self.check(outline, task, correct), [])

    def test_query_attr_output_must_be_value_not_just_names(self):
        outline, task = terminal(
            "QUERY_ATTR", target_type="Place", attribute="end date",
        )
        self.assertEqual(
            _terminal_result_contract(outline, task)["output_attribute"],
            "end date",
        )
        self.assertIn(
            "terminal_attribute_values_replaced_by_names",
            self.check(
                outline, task,
                "Find places meeting the specified conditions and return "
                "the names of matching places.",
            ),
        )
        self.assertEqual(
            self.check(
                outline, task,
                "Find the end date values for places matching the criteria.",
            ), [],
        )

    def test_population_attribute_must_not_become_person_sex(self):
        fragment = {
            "fragment_id": "fragment:branch:5",
            "depends_on": [], "output_kind": "entity_set",
            "public_steps": [
                {"id": "S1", "operation": "FILTER_NUMBER",
                 "attribute": "male population",
                 "comparison": "less than", "public_value": "727"},
            ],
        }
        outline = {"tasks": [fragment]}
        self.assertIn(
            "population_attribute_misread_as_gender:male",
            self.check(
                outline, fragment,
                "Identify entities from available evidence that are "
                "male, located below a given elevation.",
            ),
        )
        self.assertEqual(
            self.check(
                outline, fragment,
                "Find administrative areas meeting the specified "
                "male-population and elevation conditions.",
            ), [],
        )
        self.assertEqual(
            self.check(
                outline, fragment,
                "Find administrative areas matching the specified "
                "demographic criteria.",
            ), [],
        )

    def test_place_end_date_cannot_be_attached_to_later_event(self):
        fragment = {
            "fragment_id": "fragment:branch:8",
            "depends_on": [], "output_kind": "entity_set",
            "public_steps": [
                {"id": "S0", "operation": "FIND"},
                {"id": "S1", "operation": "RELATE",
                 "direction": "forward", "relation": "death place"},
                {"id": "S2", "operation": "FILTER_DATE",
                 "attribute": "end date",
                 "comparison": "equal to", "public_value": "2004-05-03"},
                {"id": "S3", "operation": "RELATE",
                 "direction": "backward", "relation": "death place"},
            ],
        }
        outline = {"tasks": [fragment]}
        self.assertIn(
            "place_end_date_attributed_to_event",
            self.check(
                outline, fragment,
                "Find events ending on May 3, 2004 at those death places "
                "and return matching entities.",
            ),
        )
        self.assertEqual(
            self.check(
                outline, fragment,
                "Find related entities using death places that satisfy "
                "the specified end-date criteria.",
            ), [],
        )

    def test_terminal_output_hint_is_local_and_public(self):
        outline, task = terminal("QUERY_NAME", target_type="Event")
        packet = _fragment_goal_packet(outline, task, "")
        self.assertEqual(packet["terminal_result_contract"], {
            "action": "QUERY_NAME",
            "output_kind": "value",
            "target_entity_type": "Event",
        })
        self.assertNotIn("task_semantics", packet)
        self.assertNotIn("reference_question", packet)
        self.assertNotIn("expected_answer", packet)
        self.assertNotIn("operation_results", packet)

    def test_nonterminal_fragment_has_no_global_entity_hint(self):
        outline, task = terminal("QUERY_NAME", target_type="Event")
        root = {
            "fragment_id": "fragment:0", "depends_on": [],
            "public_steps": [{"id": "S0", "operation": "FIND"}],
            "output_kind": "entity_set",
        }
        outline["tasks"] = [root, task]
        self.assertIsNone(_terminal_result_contract(outline, root))


if __name__ == "__main__":
    unittest.main()
