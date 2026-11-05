"""Discovery-root tasks must not present oracle candidate sets as initial inputs."""
from __future__ import annotations

from types import SimpleNamespace
import unittest

from environment.task_construction.rendering.questions import QuestionRenderer
from environment.question_generation.validators import preflight
from environment.trajectory_simulation.realization import (
    task_sentence, task_variants, report_sentence,
)
from environment.trajectory_simulation.natural_language import (
    feedback_writer_prompt, feedback_reviewer_prompt,
    realize_feedback_natural, main_single_task_prompt,
)


class CandidateDiscoveryTests(unittest.TestCase):
    def test_renderer_scopes_find_all_without_presupplied_list(self):
        q=QuestionRenderer(None)
        node=SimpleNamespace(node_id=0,op="FIND_ALL",inputs=[],dependencies=[])
        for i in range(3):
            q._variant=lambda *a,**kw:i
            phrase=q._describe(node,{},None)
            self.assertIn("task" if i!=0 else "relevant",phrase)
            self.assertNotIn("all candidate entities",phrase)
            self.assertNotIn("the entities in the records",phrase)
            self.assertNotIn("the available entities",phrase)

    def test_search_root_and_type_root_require_entity_discovery(self):
        find_all={"op":"FIND_ALL","inputs":[],"input_bindings":[]}
        find_type={"op":"FIND","find_kind":"type",
                   "inputs":["Administrative Area"],"input_bindings":[]}
        self.assertIn("Search task-relevant sources",task_sentence(find_all,[],can_search=True))
        self.assertIn("enumerate",task_sentence(find_all,[],can_search=False))
        self.assertIn("identify entities of type Administrative Area",
                      task_sentence(find_type,[],can_search=True))
        self.assertNotIn("all supplied records",task_sentence(find_all,[],can_search=True))
        self.assertNotIn("Identify all entities",task_sentence(find_type,[],can_search=False))
        frag={
            "fragment_id":"fragment:root","operations":[find_all, {
                "op":"FILTER_TYPE","inputs":["Administrative Area"],
                "input_bindings":[{"kind":"step","step":1}],
            }],
            "suggested_tools":["GoogleSearchAction","ExtractUrlContentAction"],
        }
        task=task_variants(frag,{},visible_resources=[
            {"resource_id":"source_01","title":"Reference 1","source_type":"text"},
        ])[0]
        self.assertIn("Discover candidates",task["task_instruction"])
        self.assertIn("Search task-relevant sources",task["task_instruction"])
        self.assertIn("Candidate entity names must be learned",task["context"])
        self.assertNotIn("Dunhaven",task["task_instruction"]+task["context"])
        self.assertNotIn("all supplied records",task["task_instruction"]+task["context"])

    def test_later_round_may_use_explicit_prior_results(self):
        op={"op":"COUNT","inputs":[],
            "input_bindings":[{"kind":"completed","fragment_id":"fragment:root"}]}
        frag={"operations":[op],"suggested_tools":["ExecuteCodeAction"]}
        completed={"fragment:root":{
            "public_ref":"Attempt 1, subtask 1","answer":'["Dunhaven", "Maplevale"]',
        }}
        task=task_variants(frag,completed)[0]
        self.assertIn("Dunhaven",task["task_instruction"])
        self.assertIn("Maplevale",task["context"])
        self.assertNotIn("Search task-relevant sources",task["task_instruction"])

    def test_canonical_feedback_discovery_result_is_not_an_initial_list(self):
        result=report_sentence(
            {"op":"FIND_ALL","inputs":[]},
            {"answer":["Dunhaven","Maplevale"],"input_values":[]},
        )
        self.assertIn("Enumerated candidate entities",result)
        self.assertNotIn("all supplied records",result)

    def test_stage2_preflight_rejects_presupposed_complete_supplied_roster(self):
        payload={
            "draft_question":"Among all candidate entities, which satisfy male population below 727?",
            "resources":[{"visibility":"attached"},{"visibility":"discoverable"}],
            "task_design":{"final_operation":"QUERY_NAME"},
            "terminal_contract":{},
        }
        bad="Which entities in the complete candidate list were supplied and match male population below 727?"
        result=preflight(payload,bad)
        self.assertIn("discovery_candidates_not_pre_supplied",result.issues)
        good="Which entities documented in relevant task evidence have male population below 727?"
        self.assertNotIn(
            "discovery_candidates_not_pre_supplied",
            preflight(payload,good).issues,
        )

    def test_subagent_report_names_can_be_findings_not_preset_inputs(self):
        fragment={
            "scene":"omnigaia","expected_answer":["Dunhaven","Maplevale"],
            "operations":[{"op":"FIND_ALL","inputs":[],"find_kind":"entity"}],
            "operation_results":[{"answer":["Dunhaven","Maplevale"],"input_values":[]}],
            "observations":[{
                "resource_id":"source_01","source_type":"text",
                "title":"Reference 1","visibility":"discoverable",
                "claims":[
                    {"subject":"Dunhaven","predicate":"male population","object":"254"},
                    {"subject":"Maplevale","predicate":"male population","object":"244"},
                ],
            }],
            "known_observations":[],
        }
        payload={
            "fragment":fragment,
            "task_instruction":"Discover entities represented in task evidence.",
            "tool_provenance":{"steps":[]},
        }
        good={
            "summary":"The source evidence identifies Dunhaven and Maplevale.",
            "trace_summary":(
                "Examined Reference 1 and identified Dunhaven and Maplevale "
                "as candidate entities documented in that source."
            ),
        }
        self.assertEqual(realize_feedback_natural(payload,good),good)
        bad={**good,"trace_summary":(
            "Started with the full candidate set of Dunhaven and Maplevale "
            "from the initial records."
        )}
        with self.assertRaisesRegex(ValueError,"presupplied_candidate_list"):
            realize_feedback_natural(payload,bad)
        self.assertIn("RESULTS OF DISCOVERY",feedback_writer_prompt(payload,good))
        self.assertIn("already provided initially",feedback_reviewer_prompt(payload,good))

    def test_main_rewrite_instruction_explicitly_forbids_initial_gold_roster(self):
        canonical={"action":"delegate_task","params":{"tasks":[{
            "task_instruction":"Search task sources and identify Administrative Areas.",
            "context":"Inspect accessible resources.",
            "tools":["GoogleSearchAction"],
        }]}}
        payload={
            "public_state":{"question":"Find the Administrative Areas matching a condition."},
            "visible_resources":[],"completed_subtask_contracts":[],
        }
        prompt=main_single_task_prompt(payload,canonical,0)
        self.assertIn("candidate names are initially UNKNOWN",prompt)
        self.assertIn("later rounds",prompt)


if __name__=="__main__":
    unittest.main()
