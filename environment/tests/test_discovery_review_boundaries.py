"""Source-type evidence, bounded question semantics and search/inspection separation."""
from __future__ import annotations

import unittest
from unittest.mock import Mock

from environment.trajectory_simulation.contract import validate_feedback
from environment.trajectory_simulation.eligibility import check_question
from environment.trajectory_simulation.natural_language import (
    feedback_writer_prompt, feedback_repair_prompt,
    feedback_reviewer_prompt, realize_feedback_natural,
)
from environment.trajectory_simulation.search_simulator import (
    SimulatedSearchIndex,
)
from environment.trajectory_simulation.tool_provenance import (
    source_evidence_guide, unsupported_tool_actions,
)
from environment.tests.test_review_reliability import ReviewReliabilityTests


def record(sid, name, kind, claims):
    return {
        "resource_id":sid,
        "title":name,
        "source_type":"text",
        "visibility":"provided",
        "entity_types":dict(kind),
        "claims":list(claims),
    }


def payload():
    first=record(
        "source_01","Reference 1",
        {"Dunborough":"Administrative Area","Dunport":"Administrative Area"},
        [{"subject":"Dunborough","predicate":"male population","object":"490"},
         {"subject":"Dunport","predicate":"start date","object":"1915"}],
    )
    second=record(
        "source_02","Reference 2",
        {"Winterwood":"Administrative Area","Faircrest":"Person"},
        [{"subject":"Winterwood","predicate":"male population","object":"490"},
         {"subject":"Faircrest","predicate":"start date","object":"1915"}],
    )
    fragment={
        "scene":"omnigaia",
        "operations":[{"op":"FIND_ALL","inputs":[]},
                      {"op":"FILTER_TYPE","inputs":["Administrative Area"]}],
        "operation_results":[
            {"answer":["Dunborough","Dunport","Winterwood"],"input_values":[]},
            {"answer":["Dunborough","Dunport","Winterwood"],
             "input_values":["Dunborough","Dunport","Winterwood","Faircrest"]},
        ],
        "expected_answer":["Dunborough","Dunport","Winterwood"],
        "observations":[first,second],
        "known_observations":[],
    }
    return {"fragment":fragment,"task_instruction":"Discover relevant Administrative Areas.",
            "tool_provenance":{"steps":[]}}


class ReviewAndSearchBoundaryTests(unittest.TestCase):
    def test_type_evidence_is_explicit_in_grouped_packet_and_critic(self):
        p=payload()
        guide=source_evidence_guide(p["fragment"],p["tool_provenance"])
        self.assertEqual(guide[0]["entity_types"]["Dunborough"],"Administrative Area")
        self.assertEqual(guide[1]["entity_types"]["Faircrest"],"Person")
        writer=feedback_writer_prompt(p,{})
        repair=feedback_repair_prompt(p,{})
        reviewer=feedback_reviewer_prompt(p,{
            "summary":"Dunborough, Dunport and Winterwood are the results.",
            "trace_summary":"Checked source types.",
        })
        for text in (writer,repair,reviewer):
            self.assertIn("entity_types",text)
        self.assertIn("type predicate",reviewer)

    def test_single_source_wrong_fact_still_rejected(self):
        p=payload()
        candidate={
            "summary":"The relevant entities are Dunborough, Dunport and Winterwood.",
            "trace_summary":"Reference 1 recorded Winterwood male population as 490.",
        }
        okay, issues=validate_feedback(candidate,p["fragment"])
        self.assertFalse(okay)
        self.assertTrue(any("source_fact_mismatch" in x for x in issues))

    def test_complex_sentence_no_cross_product_of_independent_source_facts(self):
        p=payload()
        candidate={
            "summary":"Dunborough, Dunport and Winterwood are the relevant entities.",
            "trace_summary":(
                "Reference 1 recorded Dunborough male population, while "
                "Winterwood male population was discussed in the subsequent result."
            ),
        }
        # Single named reference but two separate subjects and same
        # predicate: the deterministic substring checker cannot assign
        # the second claim to that reference. The LLM critic remains strict.
        okay, issues=validate_feedback(candidate,p["fragment"])
        self.assertFalse(any("source_fact_mismatch" in x for x in issues),issues)
        self.assertTrue(okay,issues)

    def test_search_teaser_never_copies_full_numeric_or_date_facts(self):
        src=record(
            "source_01","Reference 1",
            {"Dunborough":"Administrative Area"},
            [{"subject":"Dunborough","predicate":"male population","object":"490"},
             {"subject":"Dunborough","predicate":"start date","object":"1915-05-02"}],
        )
        src["visibility"]="discoverable"
        src["content"]="Dunborough — male population: 490. Dunborough — start date: 1915-05-02."
        idx=SimulatedSearchIndex([src])
        result=idx.search("administrative area male population")
        self.assertEqual(len(result),1)
        teaser=result[0]["snippet"]
        self.assertIn("Dunborough",teaser)
        self.assertIn("male population",teaser)
        self.assertNotIn("490",teaser)
        self.assertNotIn("1915",teaser)
        self.assertNotIn("1915-05-02",teaser)
        self.assertNotIn("claims",result[0])
        self.assertLessEqual(len(teaser),150)

    def test_no_search_authority_rejects_searched_evidence_and_internal_fields(self):
        p=payload()
        good={
            "summary":"Dunborough, Dunport and Winterwood match.",
            "trace_summary":"- Examined Reference 1. - Confirmed the relevant types.",
        }
        accepted=realize_feedback_natural(p,good)
        self.assertIn("\n- Confirmed",accepted["trace_summary"])
        self.assertEqual(good["trace_summary"],
                         "- Examined Reference 1. - Confirmed the relevant types.")
        invalid={
            **good,"trace_summary":"Searched the available evidence to locate Dunborough, "
                                    "Dunport and Winterwood.",
        }
        with self.assertRaisesRegex(ValueError,"unverified_tool_action_claim"):
            realize_feedback_natural(p,invalid)
        self.assertIn("GoogleSearchAction",unsupported_tool_actions(
            "Searched the available evidence.",p["tool_provenance"],
        ))
        with self.assertRaisesRegex(ValueError,"leaks_internal_evidence_handle"):
            realize_feedback_natural(p,{
                **good,"trace_summary":"The source evidence guide lists Dunborough.",
            })

    def test_stage3_question_reviewer_contract_matches_stage2_discovery(self):
        case=ReviewReliabilityTests()
        gold,record,role=case.review_fixture()
        role.check_question.return_value={"decision":"eligible","issues":[]}
        check_question(gold,record,role)
        arguments=[c.args[0] for c in role.check_question.call_args_list]
        text="\n".join(arguments)
        self.assertIn("finite task evidence domain",text)
        self.assertIn("When the terminal is QUERY_NAME",text)
        self.assertIn("missing relation traversal",text)


if __name__=="__main__":
    unittest.main()
