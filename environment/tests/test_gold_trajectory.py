"""Offline regressions: first-stage fragments -> O1 Main/Sub-Agent gold episodes."""
from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from environment.trajectory_simulation.backend import PassthroughPlanner, PassthroughExecutor
from environment.trajectory_simulation.compiler import (
    GoldCompilationError, compile_gold_solution, private_guidance, ready_fragments,
)
from environment.trajectory_simulation.contract import (
    feedback_schema, answers_equal, validate_feedback,
)
from environment.trajectory_simulation.pipeline import (
    GoldTrajectoryError, plan_trajectories_jsonl, synthesize_one,
    synthesize_trajectories_jsonl,
)


def fixture(*, dependent=False, terminal_evidence=False):
    """Composition links are the authoritative subtask boundaries."""
    nodes = [
        {"node_id": 0, "op": "FIND", "dependencies": [], "inputs": ["entity:A"]},
        {"node_id": 1, "op": "RELATE", "dependencies": [0],
         "inputs": ["r:first", "forward"]},
        {"node_id": 2, "op": "RELATE", "dependencies": [1] if dependent else [0],
         "inputs": ["r:second", "forward"]},
    ]
    if dependent:
        nodes.append({"node_id": 3, "op": "QUERY_NAME", "dependencies": [2], "inputs": []})
    else:
        nodes.extend([
            {"node_id": 3, "op": "INTERSECT", "dependencies": [1, 2], "inputs": []},
            {"node_id": 4, "op": "QUERY_NAME", "dependencies": [3], "inputs": []},
        ])
    if terminal_evidence:
        nodes.append({
            "node_id": 5, "op": "QUERY_ATTR", "dependencies": [4],
            "inputs": ["r:height"],
        })
    last = nodes[-1]["node_id"]
    record = {
        "program": {
            "nodes": nodes,
            "metadata": {"deep_dependency": {
                "composition_links": [
                    {"fragment": 0, "stage": 0, "producer_node_id": None,
                     "output_node_id": 1},
                    {"fragment": 1, "stage": 1 if dependent else 0,
                     "producer_node_id": 1 if dependent else None,
                     "output_node_id": 2},
                ],
                "orchestration_motif": "chain" if dependent else "fork_join",
            }},
        },
        "world": {
            "entities": {
                "A": "schema:Person", "B": "schema:Person",
                **({"C": "schema:Person"} if dependent else {}),
            },
            "facts": [
                {"fact_id": "f1", "subject": "A", "predicate": "r:first",
                 "object": "B", "object_kind": "entity", "source_id": "s1"},
                {"fact_id": "f2", "subject": "B" if dependent else "A",
                 "predicate": "r:second",
                 "object": "C" if dependent else "B",
                 "object_kind": "entity", "source_id": "s2"},
            ] + ([{
                "fact_id": "f3", "subject": "B", "predicate": "r:height",
                "object": 175, "object_kind": "number", "source_id": "s3",
            }] if terminal_evidence else []),
            "sources": {
                "s1": {"source_id": "s1", "source_type": "text",
                       "fact_ids": ["f1"], "metadata": {}},
                "s2": {"source_id": "s2", "source_type": "text",
                       "fact_ids": ["f2"], "metadata": {}},
                **({"s3": {"source_id": "s3", "source_type": "table",
                           "fact_ids": ["f3"], "metadata": {}}}
                   if terminal_evidence else {}),
            },
            "metadata": {},
        },
    }
    rendered_sources = [
        SimpleNamespace(source_id="s1", source_type="text",
                        title="First reference", content="Alice first Bob"),
        SimpleNamespace(source_id="s2", source_type="text",
                        title="Second reference",
                        content="Bob second Carol" if dependent else "Alice second Bob"),
    ]
    resources = [
        {"resource_id": "s1", "source_type": "text", "visibility": "provided",
         "anchor_dependency_resource_ids": []},
        {"resource_id": "s2", "source_type": "text",
         "visibility": "discoverable" if dependent else "provided",
         "anchor_dependency_resource_ids": ["s1"] if dependent else []},
    ]
    if terminal_evidence:
        resources.append({
            "resource_id": "s3", "source_type": "table", "visibility": "provided",
            "anchor_dependency_resource_ids": [],
        })
        rendered_sources.append(SimpleNamespace(
            source_id="s3", source_type="table",
            title="Height lookup", content="Bob has height 175",
        ))
    realized = {
        "task_id": "world-test",
        "question": "Find the result of the composed relations for Alice.",
        "capability_world": {"resources": resources, "binding_audit": {}},
    }
    rendered = SimpleNamespace(
        task_id="world-test",
        answer=[175] if terminal_evidence else ["Carol"] if dependent else ["Bob"],
        hidden_metadata={
            "entity_aliases": {"A": "Alice", "B": "Bob", "C": "Carol"},
            "literal_surfaces": {},
            "literal_surfaces_by_predicate": {},
        },
        sources=rendered_sources,
    )
    return realized, record, rendered


class FragmentCompilationTests(unittest.TestCase):
    def test_parallel_fragments_share_question_but_not_sources(self):
        question, world, rendered = fixture()
        gold = compile_gold_solution(question, world, rendered)
        self.assertEqual(len(gold["fragments"]), 3)
        self.assertEqual(gold["waves"], [["fragment:0", "fragment:1"], ["fragment:terminal"]])
        self.assertEqual(gold["fragments"][0]["resource_ids"], ["s1"])
        self.assertEqual(gold["fragments"][1]["resource_ids"], ["s2"])
        self.assertEqual(gold["fragments"][0]["expected_answer"], ["Bob"])
        self.assertEqual(gold["fragments"][1]["expected_answer"], ["Bob"])
        self.assertEqual(gold["final_answer"], ["Bob"])
        self.assertEqual(gold["policy"], "first_stage_composition_links_fragment_gold")

    def test_dependency_and_resource_anchor_are_respected(self):
        gold = compile_gold_solution(*fixture(dependent=True))
        self.assertEqual(gold["waves"], [["fragment:0"], ["fragment:1"], ["fragment:terminal"]])
        self.assertEqual(gold["fragments"][1]["depends_on"], ["fragment:0"])
        self.assertEqual(gold["fragments"][1]["resource_ids"], ["s2"])
        self.assertEqual(gold["fragments"][1]["support_resource_ids"], ["s1", "s2"])
        self.assertEqual(gold["fragments"][1]["expected_answer"], ["Carol"])

    def test_terminal_evidence_gets_its_own_fragment(self):
        gold = compile_gold_solution(*fixture(terminal_evidence=True))
        self.assertEqual(gold["fragments"][-1]["fragment_id"], "fragment:terminal")
        self.assertEqual(gold["fragments"][-1]["expected_answer"], [175])
        self.assertEqual(gold["fragments"][-1]["resource_ids"], ["s3"])
        self.assertEqual(gold["fragments"][-1]["support_resource_ids"], ["s1", "s2", "s3"])
        self.assertEqual(gold["waves"][-1], ["fragment:terminal"])
        self.assertEqual([op["op"] for op in gold["fragments"][-1]["operations"]],
                         ["INTERSECT", "QUERY_NAME", "QUERY_ATTR"])

    def test_terminal_computation_is_kept_without_new_sources(self):
        gold = compile_gold_solution(*fixture())
        terminal = gold["fragments"][-1]
        self.assertEqual(terminal["resource_ids"], [])
        self.assertEqual([op["op"] for op in terminal["operations"]],
                         ["INTERSECT", "QUERY_NAME"])
        self.assertEqual({obs["resource_id"] for obs in terminal["known_observations"]},
                         {"s1", "s2"})

    def test_comparison_guidance_and_rejected_candidate_evidence(self):
        question, record, rendered = fixture()
        record["program"]["nodes"][1] = {
            "node_id": 1, "op": "FILTER_YEAR", "dependencies": [0],
            "inputs": ["r:birth", "<CMP:=>", "1674"],
        }
        record["world"]["facts"][0].update(
            predicate="r:birth", object="1687", object_kind="year")
        rendered.answer = []
        rendered.sources[0].content = "Alice birth year 1687"
        gold = compile_gold_solution(question, record, rendered)
        first = gold["fragments"][0]
        self.assertEqual(first["expected_answer"], [])
        self.assertEqual(first["resource_ids"], ["s1"])
        self.assertEqual(first["operations"][-1]["inputs"][1], "=")
        self.assertNotIn("<CMP:", json.dumps(private_guidance(gold, set())))

    def test_compact_schedule_merges_linear_decisions_without_losing_gold(self):
        baseline = compile_gold_solution(*fixture(dependent=True), schedule="baseline")
        compact = compile_gold_solution(*fixture(dependent=True), schedule="compact")
        self.assertGreater(len(baseline["waves"]), len(compact["waves"]))
        self.assertEqual(compact["final_answer"], baseline["final_answer"])
        self.assertEqual(compact["schedule"], "compact")

    def test_unsourced_distractor_entities_do_not_block_public_witness(self):
        # Stage-1 FIND_ALL includes 12 synthetic background entities, but
        # no public source mentions them. The downstream RELATE restricts
        # to a fully evidenced result; the public projection can replay it.
        question, record, rendered = fixture()
        record["program"]["nodes"][0]["op"] = "FIND_ALL"
        record["program"]["nodes"][0]["inputs"] = []
        record["world"]["entities"].update({
            f"d{i:04d}": "schema:Person" for i in range(1, 13)
        })
        gold = compile_gold_solution(question, record, rendered)
        self.assertEqual(gold["final_answer"], ["Bob"])
        self.assertEqual(gold["fragments"][0]["expected_answer"], ["Bob"])
        result = synthesize_one(
            question, record, rendered,
            PassthroughPlanner(), PassthroughExecutor(),
        )
        self.assertTrue(result["validation"]["public_evidence_replay_verified"])
        self.assertEqual(result["trajectory"][-1]["params"]["answer"], '["Bob"]')
        # Private world is untouched, and hidden distractors never appear
        # in the public Main state or its history.
        self.assertNotIn("d0001", json.dumps(result["trajectory"]))

    def test_public_intermediate_domain_may_differ_without_changing_final_answer(self):
        from environment.trajectory_simulation.public_preflight import audit_public_executability
        question, record, rendered = fixture()
        record["program"]["nodes"] = [
            {"node_id": 0, "op": "FIND", "dependencies": [],
             "inputs": ["type:schema:Person"]},
            {"node_id": 1, "op": "RELATE", "dependencies": [0],
             "inputs": ["r:first", "forward"]},
            {"node_id": 2, "op": "QUERY_NAME", "dependencies": [1], "inputs": []},
        ]
        record["program"]["metadata"]["deep_dependency"]["composition_links"] = [
            {"fragment": 0, "stage": 0, "producer_node_id": None,
             "output_node_id": 0},
            {"fragment": 1, "stage": 1, "producer_node_id": 0,
             "output_node_id": 1},
        ]
        # This unsourced person appears in the private FIND(type) output
        # but is invisible to a public Sub-Agent. The downstream relation
        # never uses it, so the full public answer remains Bob.
        record["world"]["entities"]["ghost"] = "schema:Person"
        gold = compile_gold_solution(question, record, rendered)
        root = next(f for f in gold["fragments"] if f["output_node_id"] == 0)
        self.assertEqual(root["expected_answer"], ["Alice", "Bob"])
        self.assertTrue(any(
            item["node_id"] == 0 for item in gold["public_intermediate_projections"]
        ))
        self.assertTrue(audit_public_executability(gold)["publicly_executable"])

    def test_final_source_domain_mismatch_is_never_projected_away(self):
        question, record, rendered = fixture()
        record["program"]["nodes"] = [
            {"node_id": 0, "op": "FIND", "dependencies": [],
             "inputs": ["type:schema:Person"]},
            {"node_id": 1, "op": "COUNT", "dependencies": [0], "inputs": []},
        ]
        record["program"]["metadata"]["deep_dependency"]["composition_links"] = [
            {"fragment": 0, "stage": 0, "producer_node_id": None,
             "output_node_id": 0},
        ]
        record["world"]["entities"]["ghost"] = "schema:Person"
        rendered.answer = 3
        with self.assertRaisesRegex(
            GoldCompilationError, "public_full_world_answer_disagrees_with_gold"
        ):
            compile_gold_solution(question, record, rendered)

    def test_unique_public_intersection_can_be_certified_without_stage1_tag(self):
        from environment.trajectory_simulation.public_preflight import audit_public_executability
        question, record, rendered = fixture()
        record["program"]["nodes"] = [
            {"node_id": 0, "op": "FIND", "dependencies": [],
             "inputs": ["entity:A"]},
            {"node_id": 1, "op": "FIND", "dependencies": [],
             "inputs": ["type:schema:Person"]},
            {"node_id": 2, "op": "FILTER_YEAR", "dependencies": [1],
             "inputs": ["r:birth", "<CMP:=>", "1912"]},
            {"node_id": 3, "op": "INTERSECT", "dependencies": [0, 2], "inputs": []},
            {"node_id": 4, "op": "QUERY_NAME", "dependencies": [3], "inputs": []},
        ]
        record["program"]["metadata"]["deep_dependency"]["composition_links"] = [
            {"fragment": 0, "stage": 0, "producer_node_id": None,
             "output_node_id": 2},
        ]
        record["world"]["facts"][0].update(
            predicate="r:birth", object="1912", object_kind="year")
        record["world"]["facts"][1].update(
            subject="B", predicate="r:birth", object="1980", object_kind="year")
        question["question"] = "Which Person has recorded birth year 1912?"
        rendered.answer = ["Alice"]
        rendered.hidden_metadata.pop("public_query_plan", None)
        gold = compile_gold_solution(question, record, rendered)
        self.assertEqual(len(gold["public_projections"]), 1)
        self.assertEqual(gold["public_projections"][0]["rule"],
                         "witness_verified_intersection")
        self.assertTrue(audit_public_executability(gold)["publicly_executable"])
        # A second matching person destroys the singleton proof: fail closed.
        record["world"]["facts"][1]["object"] = "1912"
        not_unique = compile_gold_solution(question, record, rendered)
        self.assertEqual(not_unique["public_projections"], [])
        self.assertFalse(audit_public_executability(not_unique)["publicly_executable"])

    def test_missing_entities_that_change_answer_still_rejected(self):
        from environment.trajectory_simulation.compiler import _minimal_witness
        from environment.task_construction.rendering.pipeline import program_from_dict, world_from_dict
        _, record, _ = fixture()
        record["program"]["nodes"] = [
            {"node_id": 0, "op": "FIND_ALL", "dependencies": [], "inputs": []}
        ]
        record["world"]["entities"]["d0001"] = "schema:Person"
        program = program_from_dict(record["program"])
        world = world_from_dict(record["world"])
        with self.assertRaises(GoldCompilationError) as caught:
            _minimal_witness(
                program, world, set(world.entities),
                {"s1": [], "s2": []}, max_sources=8,
            )
        self.assertEqual(
            str(caught.exception), "witness_answer_disagrees_with_gold"
        )
        self.assertFalse(caught.exception.diagnostics["all_sources_answer_matches"])

    def test_witness_diagnostics_report_unobservable_entity(self):
        from environment.trajectory_simulation.compiler import _minimal_witness
        from environment.task_construction.rendering.pipeline import program_from_dict, world_from_dict
        q, record, rendered = fixture()
        program = program_from_dict(record["program"])
        world = world_from_dict(record["world"])
        with self.assertRaises(GoldCompilationError) as caught:
            _minimal_witness(
                program, world, {"B"}, {"s1": [], "s2": []},
                max_sources=8, required_entity_ids=("unknown_ghost",),
            )
        self.assertEqual(str(caught.exception),
                         "witness_required_entities_not_observable")
        self.assertIn("unknown_ghost",
                      caught.exception.diagnostics["unobservable_required_entity_ids"])

    def test_witness_diagnostics_report_unsourced_fact(self):
        from environment.trajectory_simulation.compiler import _minimal_witness
        from environment.task_construction.rendering.pipeline import program_from_dict, world_from_dict
        _, record, _ = fixture()
        program = program_from_dict(record["program"])
        world = world_from_dict(record["world"])
        with self.assertRaises(GoldCompilationError) as caught:
            _minimal_witness(
                program, world, {"B"}, {"s1": [], "s2": []},
                max_sources=8, required_fact_ids=("missing-fact",),
            )
        self.assertEqual(str(caught.exception),
                         "witness_required_facts_not_in_sources")
        self.assertIn("missing-fact",
                      caught.exception.diagnostics["unsourced_required_fact_ids"])

    def test_public_projection_removes_certified_concealed_intersection(self):
        from environment.trajectory_simulation.public_preflight import audit_public_executability
        question,record,rendered=fixture()
        # Stage 1 concealed a singleton because an independently evidenced
        # type+attribute branch uniquely identifies the very same entity.
        record['program']['nodes']=[
            {'node_id':0,'op':'FIND','dependencies':[],'inputs':['entity:A']},
            {'node_id':1,'op':'FIND','dependencies':[],'inputs':['type:schema:Person']},
            {'node_id':2,'op':'FILTER_YEAR','dependencies':[1],
             'inputs':['r:birth','<CMP:=>','1912']},
            {'node_id':3,'op':'INTERSECT','dependencies':[0,2],'inputs':[]},
            {'node_id':4,'op':'QUERY_NAME','dependencies':[3],'inputs':[]},
        ]
        record['program']['metadata']['deep_dependency']['composition_links']=[
            {'fragment':0,'stage':0,'producer_node_id':None,'output_node_id':2}
        ]
        record['world']['facts'][0].update(
            predicate='r:birth',object='1912',object_kind='year')
        record['world']['facts'][1].update(
            subject='B',predicate='r:birth',object='1980',object_kind='year')
        question['question']='Which Person has recorded birth year 1912?'
        rendered.answer=['Alice']
        rendered.hidden_metadata['public_query_plan']={
            'latent_entities':[{
                'entity_id':'A','entity_type':'schema:Person',
                'find_node_ids':[0],
                'rule':'unique_intersection_clue',
                'identifying_node_ids':[2],
                'support_fact_ids':['f1'],
            }]
        }
        baseline=compile_gold_solution(question,record,rendered,schedule='baseline')
        self.assertEqual(len(baseline['public_projections']),1)
        self.assertTrue(all(
            fragment['output_node_id'] not in (0,3)
            for fragment in baseline['fragments']))
        self.assertTrue(audit_public_executability(baseline)['publicly_executable'])
        compact=compile_gold_solution(question,record,rendered,schedule='compact')
        self.assertTrue(audit_public_executability(compact)['publicly_executable'])
        episode=synthesize_one(
            question,record,rendered,
            PassthroughPlanner(),PassthroughExecutor(),schedule='compact')
        self.assertTrue(episode['validation']['accepted'])
        self.assertEqual(episode['trajectory'][-1]['params']['answer'],'["Alice"]')
        self.assertNotIn('fragment:public_join:',json.dumps(episode['trajectory']))

    def test_unexpected_synthesis_error_retains_type_and_traceback(self):
        question, record, rendered = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            with patch('environment.trajectory_simulation.pipeline._rows',
                       return_value=[question]), patch(
                'environment.trajectory_simulation.pipeline._world_map',
                return_value={question['task_id']: (record, rendered)},
            ), patch(
                'environment.trajectory_simulation.pipeline.synthesize_one',
                side_effect=KeyError(15),
            ):
                summary = synthesize_trajectories_jsonl(
                    base / 'worlds.jsonl', base / 'realized.jsonl',
                    base / 'good.jsonl',
                    rejected_output_path=base / 'bad.jsonl',
                    quarantine_output_path=base / 'quarantine.jsonl',
                    planner=PassthroughPlanner(), executor=PassthroughExecutor(),
                    progress=False,
                )
            self.assertEqual(summary['rejected'], 1)
            item = json.loads((base / 'bad.jsonl').read_text().strip())
            self.assertEqual(item['reason'], 'unexpected_error')
            self.assertEqual(item['issues'], ['KeyError: 15'])
            self.assertEqual(item['audit']['exception_type'], 'KeyError')
            self.assertIn('KeyError: 15', item['audit']['traceback'])

    def test_semantic_reviewer_rewires_only_certified_redundant_nodes(self):
        from environment.trajectory_simulation.eligibility import check_question
        question, record, rendered = fixture()
        record['program']['nodes'] = [
            {'node_id': 0, 'op': 'FIND', 'dependencies': [], 'inputs': ['entity:A']},
            {'node_id': 1, 'op': 'FIND', 'dependencies': [], 'inputs': ['type:schema:Person']},
            {'node_id': 2, 'op': 'FILTER_YEAR', 'dependencies': [1],
             'inputs': ['r:birth', '<CMP:=>', '1912']},
            {'node_id': 3, 'op': 'INTERSECT', 'dependencies': [0, 2], 'inputs': []},
            {'node_id': 4, 'op': 'QUERY_NAME', 'dependencies': [3], 'inputs': []},
        ]
        record['program']['metadata']['deep_dependency']['composition_links'] = [
            {'fragment': 0, 'stage': 0, 'producer_node_id': None,
             'output_node_id': 2},
        ]
        record['world']['facts'][0].update(
            predicate='r:birth', object='1912', object_kind='year')
        record['world']['facts'][1].update(
            subject='B', predicate='r:birth', object='1980', object_kind='year')
        question['question'] = 'Which Person has recorded birth year 1912?'
        rendered.answer = ['Alice']
        rendered.hidden_metadata['public_query_plan'] = {
            'latent_entities': [{
                'entity_id': 'A', 'entity_type': 'schema:Person',
                'find_node_ids': [0], 'rule': 'unique_intersection_clue',
                'identifying_node_ids': [2], 'support_fact_ids': ['f1'],
            }],
        }
        gold = compile_gold_solution(question, record, rendered)
        self.assertEqual(len(gold['public_projections']), 1)
        self.assertNotIn(0, {
            op['node_id'] for f in gold['fragments'] for op in f['operations']
        })
        self.assertNotIn(3, {
            op['node_id'] for f in gold['fragments'] for op in f['operations']
        })

        class Reviewer:
            offline = False
            backend = SimpleNamespace(model='regression-reviewer', base_url='test://local')
            payload = None

            def check_question(self, prompt):
                self.payload = json.loads(prompt.split('\n', 1)[1])
                return {'decision': 'eligible', 'issues': []}

        role = Reviewer()
        verdict = check_question(gold, record, role)
        self.assertTrue(verdict['reviewed'])
        self.assertTrue(verdict['eligible'])
        steps = {item['step']: item for item in role.payload['canonical_steps']}
        self.assertEqual(set(steps), {1, 2, 4})
        self.assertEqual(steps[4]['input_steps'], [2])
        self.assertEqual(record['program']['nodes'][-1]['dependencies'], [3])
        self.assertEqual(len(role.payload['certified_public_projections']), 1)
        # A pruned delegation view alone is NOT permission to erase semantics.
        from environment.trajectory_simulation.eligibility import semantic_review_nodes
        full=semantic_review_nodes({**gold,'public_projections':[]},record)
        self.assertEqual(set(full), {0, 1, 2, 3, 4})
        self.assertEqual(full[4]['dependencies'], [3])

    def test_missing_composition_metadata_rejected_without_fallback(self):
        question, world, rendered = fixture()
        world["program"]["metadata"] = {}
        with self.assertRaisesRegex(GoldCompilationError, "composition_links"):
            compile_gold_solution(question, world, rendered)

    def test_private_guidance_has_operations_not_future_answers(self):
        gold = compile_gold_solution(*fixture(dependent=True))
        prompt_state = private_guidance(gold, set())
        self.assertEqual(len(prompt_state["ready"]), 1)
        self.assertNotIn("expected_answer", json.dumps(prompt_state))
        self.assertNotIn("Carol", json.dumps(prompt_state))
        self.assertNotIn("observations", json.dumps(prompt_state))
        self.assertFalse(prompt_state["may_complete"])
        self.assertEqual(prompt_state["ready"][0]["operations"][0]["inputs"], ["Alice"])


class FragmentTrajectoryTests(unittest.TestCase):
    def test_unrelated_task_is_rejected_by_fixed_contract_review(self):
        class Main(PassthroughPlanner):
            offline = False
            def decide(self, payload, prompt):
                value = super().decide(payload, prompt)
                value["params"]["tasks"][0]["task_instruction"] = "Summarize the weather."
                return value
            def review(self, prompt):
                return {"accept": False, "issues": [{
                    "quote": "Summarize the weather.",
                    "reason": "The assigned subgoal is a relation query, not weather.",
                }]}
        with self.assertRaisesRegex(GoldTrajectoryError, "language_inconsistency"):
            synthesize_one(*fixture(), Main(), PassthroughExecutor(), max_attempts=1)

    def test_contradictory_summary_is_rejected_despite_fixed_answer(self):
        class Sub(PassthroughExecutor):
            offline = False
            def respond(self, payload, prompt):
                return {"summary": "The answer is Nobody, not Bob.",
                        "trace_summary": "Used the assigned relation evidence."}
            def review(self, prompt):
                return {"accept": False, "issues": [{
                    "quote": "The answer is Nobody, not Bob.",
                    "reason": "The authoritative subtask answer is Bob.",
                }]}
        with self.assertRaisesRegex(GoldTrajectoryError, "language_inconsistency"):
            synthesize_one(*fixture(), PassthroughPlanner(), Sub(), max_attempts=1)

    def test_review_requires_a_quote_from_generated_text(self):
        from environment.trajectory_simulation.review import review_language
        role = SimpleNamespace(offline=False, review=lambda prompt: {
            "accept": False, "issues": [{"quote": "not in the summary", "reason": "Wrong."}],
        })
        self.assertEqual(review_language(role, "test", ["The result is Bob."]),
                         (False, ["invalid_language_review"]))

    def test_unknown_reference_title_has_specific_rejection(self):
        gold = compile_gold_solution(*fixture())
        self.assertIn("trace_summary_unknown_resource_title:Reference 99",
                      validate_feedback({
                          "summary": "The result is Bob.",
                          "trace_summary": "Consulted Reference 99.",
                      }, gold["fragments"][0])[1])

    def test_public_feedback_cannot_describe_the_synthesis_controller(self):
        gold = compile_gold_solution(*fixture())
        for label in ("ProgramIR", "SymbolicExecutor", "symbolic executor"):
            with self.subTest(label=label):
                self.assertFalse(validate_feedback({
                    "summary": "The result is Bob.",
                    "trace_summary": label + " supplied the answer.",
                }, gold["fragments"][0])[0])

    def test_single_source_fact_attribution_must_match_its_claims(self):
        gold = compile_gold_solution(*fixture(dependent=True))
        fragment = gold["fragments"][0]
        fragment["known_observations"] = gold["fragments"][1]["observations"]
        fragment["observations"][0]["title"] = "Reference 1"
        fragment["known_observations"][0]["title"] = "Reference 2"
        bad = {"summary": "The result is Bob.",
               "trace_summary": "Reference 1 confirmed Bob's second relation to Carol."}
        self.assertTrue(any("source_fact_mismatch:s1" in issue
                            for issue in validate_feedback(bad, fragment)[1]))
        good = {**bad, "trace_summary": "Reference 2 confirmed Bob's second relation to Carol."}
        self.assertTrue(validate_feedback(good, fragment)[0])

    def test_feedback_can_reuse_prior_evidence_without_new_sources(self):
        from environment.trajectory_simulation.contract import feedback_prompt
        gold = compile_gold_solution(*fixture(terminal_evidence=True))
        terminal = gold["fragments"][-1]
        terminal["known_observations"][0]["claims"].append({
            "subject": "Alice", "predicate": "birth date",
            "object": "1687", "object_kind": "year"})
        value = {
            "summary": "The previously read s1 record lists Alice's birth date as 1687.",
            "trace_summary": "Reused the prior evidence without another retrieval.",
        }
        self.assertTrue(validate_feedback(value, terminal)[0])
        prompt = feedback_prompt({
            "question": "Find Alice.", "task_instruction": "Finish the attribute query.",
            "dependency_results": [], "fragment": terminal,
        })
        self.assertIn("Reused the earlier observations", prompt)
        self.assertIn("First reference", prompt)
        self.assertNotIn("1687", prompt)  # Unused facts are not language choices.

    def test_main_prompt_explicitly_requires_native_params_tasks(self):
        from environment.trajectory_simulation.contract import main_prompt
        gold = compile_gold_solution(*fixture())
        prompt = main_prompt({
            "public_state": {"question": gold["question"]},
            "gold_guidance": private_guidance(gold, set()),
        })
        self.assertIn("params.tasks", prompt)
        self.assertIn("required_output_schema", prompt)
        self.assertNotIn("fragment:", prompt)
        self.assertNotIn('"s1"', prompt)

    def test_relative_step_labels_are_not_numeric_fact_claims(self):
        gold = compile_gold_solution(*fixture())
        self.assertTrue(validate_feedback({
            "summary": "The result is Bob.",
            "trace_summary": "Executed Step 1 and Step 2 using the assigned evidence.",
        }, gold["fragments"][0])[0])

    def test_all_source_modality_tools_are_required(self):
        from environment.trajectory_simulation.contract import validate_action
        gold = compile_gold_solution(*fixture())
        guidance = private_guidance(gold, set())
        guidance["ready"][0]["suggested_tools"] = ["ExtractUrlContentAction", "ImageAnalysisAction"]
        action = PassthroughPlanner().decide({
            "public_state": {"question": gold["question"]},
            "gold_guidance": guidance,
        }, "")
        action["params"]["tasks"][0]["tools"] = ["ExtractUrlContentAction"]
        self.assertIn("task_0:tools_not_grounded",
                      validate_action(action, guidance, ["Bob"], {})[1])

    def test_complete_answer_must_use_o1_string_transport(self):
        from environment.trajectory_simulation.contract import validate_action
        action = {"action": "complete", "reasoning": "",
                  "params": {"answer": 2}}
        self.assertIn("complete_answer_must_be_nonempty_string",
                      validate_action(action, {"may_complete": True}, 2, {})[1])

    def test_public_o1_state_and_private_supervision_are_separate(self):
        result = synthesize_one(*fixture(dependent=True),
                                PassthroughPlanner(), PassthroughExecutor())
        self.assertTrue(result["validation"]["accepted"])
        self.assertEqual([step["action"] for step in result["trajectory"]],
                         ["delegate_task", "delegate_task", "delegate_task", "complete"])
        first = result["trajectory"][0]
        self.assertEqual(set(first["state_before"]),
                         {"question", "subtask_history", "available_tools", "evidence_store"})
        self.assertEqual(first["state_before"]["subtask_history"],
                         "No subtasks completed yet.")
        second = result["trajectory"][1]
        second_history = second["state_before"]["subtask_history"]
        self.assertIn("[Attempt 1] 🔄 Parallel Batch (1 subtasks)", second_history)
        self.assertIn('["Bob"]', second_history)
        self.assertIn("Trace summary:", second_history)
        final_history = result["trajectory"][-1]["state_before"]["subtask_history"]
        self.assertIn("[Attempt 1]", final_history)
        self.assertIn("[Attempt 2]", final_history)
        self.assertIn("Summary: 3/3 subtasks done", final_history)
        self.assertIn('["Carol"]', final_history)
        public = json.dumps(result["trajectory"])
        self.assertNotIn("fragment:0", public)
        self.assertNotIn("symbolic_trace", public)
        self.assertNotIn("private_gold_guidance", public)
        self.assertEqual(result["gold_supervision"]["parallel_waves"],
                         [["fragment:0"], ["fragment:1"], ["fragment:terminal"]])
        self.assertEqual(result["trajectory"][-1]["params"]["answer"], '["Carol"]')

    def test_subagent_result_is_symbolically_fixed_not_model_copied_facts(self):
        result = synthesize_one(*fixture(),
                                PassthroughPlanner(), PassthroughExecutor())
        replies = result["trajectory"][0]["result"]["subtask_results"]
        self.assertEqual([reply["finish_result"]["result"] for reply in replies],
                         ['["Bob"]', '["Bob"]'])
        self.assertNotIn("grounded_claims", json.dumps(replies))
        self.assertNotIn("fact_id", json.dumps(replies))
        self.assertIn("trace_summary", replies[0])
        schema = feedback_schema()
        self.assertEqual(set(schema["properties"]), {"summary", "trace_summary"})

    def test_o1_history_contains_all_parallel_results_across_rounds(self):
        from environment.trajectory_simulation.history import format_o1_subtask_history
        episode = synthesize_one(*fixture(), PassthroughPlanner(), PassthroughExecutor())
        before_finish = episode["trajectory"][-1]["state_before"]
        history = before_finish["subtask_history"]
        self.assertIn("Parallel Batch (2 subtasks)", history)
        self.assertIn("[1] ✅ done", history)
        self.assertIn("[2] ✅ done", history)
        self.assertEqual(history.count("Trace summary:"), 3)
        self.assertEqual(history.count("Result:"), 3)
        self.assertIn("Summary: 3/3 subtasks done", history)
        self.assertEqual(
            history,
            format_o1_subtask_history(episode["trajectory"][:-1]),
        )

    def test_main_prompt_uses_formatted_history_but_keeps_private_gold_separate(self):
        from environment.trajectory_simulation.contract import main_prompt
        from environment.trajectory_simulation.compiler import private_guidance
        episode = synthesize_one(*fixture(dependent=True),
                                 PassthroughPlanner(), PassthroughExecutor())
        state = episode["trajectory"][-1]["state_before"]
        self.assertNotIn("private_gold_guidance", json.dumps(state))
        self.assertNotIn("fragment:", json.dumps(state))
        compiled = compile_gold_solution(*fixture(dependent=True))
        guidance = private_guidance(compiled, set())
        text = main_prompt({"public_state": state, "gold_guidance": guidance})
        self.assertIn("public_o1_state", text)
        self.assertIn("private_gold_guidance", text)
        self.assertIn("Trace summary:", text)
        self.assertIn("Summary: 3/3 subtasks done", text)

    def test_parallel_subagents_are_independent_requests(self):
        class SlowExecutor(PassthroughExecutor):
            def __init__(self):
                self.lock = threading.Lock()
                self.active = self.peak = 0
            def respond(self, payload, prompt):
                with self.lock:
                    self.active += 1
                    self.peak = max(self.peak, self.active)
                time.sleep(.03)
                value = super().respond(payload, prompt)
                with self.lock:
                    self.active -= 1
                return value
        executor = SlowExecutor()
        result = synthesize_one(*fixture(), PassthroughPlanner(), executor,
                                subagent_workers=2)
        self.assertTrue(result["success"])
        self.assertGreaterEqual(executor.peak, 2)

    def test_invalid_number_or_private_handle_rejected(self):
        class BadExecutor(PassthroughExecutor):
            def respond(self, payload, prompt):
                return {"summary": "Found 99999 unobserved records.",
                        "trace_summary": "Read fragment:0."}
        with self.assertRaisesRegex(GoldTrajectoryError, "unsupported_numeric_claim"):
            synthesize_one(*fixture(), PassthroughPlanner(), BadExecutor(),
                           max_attempts=1)

    def test_main_cannot_leak_future_resource_handles(self):
        class LeakyPlanner(PassthroughPlanner):
            def decide(self, payload, prompt):
                value = super().decide(payload, prompt)
                if value["action"] == "delegate_task":
                    value["params"]["tasks"][0]["task_instruction"] += " source_99"
                return value
        with self.assertRaisesRegex(GoldTrajectoryError,
                                    "future_resource_id_in_public_action"):
            synthesize_one(*fixture(), LeakyPlanner(), PassthroughExecutor(),
                           max_attempts=1)

    def test_subagent_cannot_override_programmed_answer(self):
        class UntrustedExecutor(PassthroughExecutor):
            def respond(self, payload, prompt):
                value = super().respond(payload, prompt)
                value["answer"] = "Wrong answer"
                return value
        with self.assertRaisesRegex(GoldTrajectoryError, "unexpected_feedback_fields"):
            synthesize_one(*fixture(), PassthroughPlanner(), UntrustedExecutor(),
                           max_attempts=1)

    def test_main_cannot_drop_gold_parallel_fragment(self):
        class IncompletePlanner(PassthroughPlanner):
            def decide(self, payload, prompt):
                value = super().decide(payload, prompt)
                if value["action"] == "delegate_task":
                    value["params"]["tasks"].pop()
                return value
        with self.assertRaisesRegex(GoldTrajectoryError,
                                    "parallel_gold_fragment_count_mismatch"):
            synthesize_one(*fixture(), IncompletePlanner(),
                           PassthroughExecutor(), max_attempts=1)

    def test_final_answer_checked_after_all_fragments(self):
        class WrongPlanner(PassthroughPlanner):
            def decide(self, payload, prompt):
                value = super().decide(payload, prompt)
                if value["action"] == "complete":
                    value["params"]["answer"] = '["Incorrect"]'
                return value
        with self.assertRaisesRegex(GoldTrajectoryError, "final_answer_mismatch"):
            synthesize_one(*fixture(), WrongPlanner(),
                           PassthroughExecutor(), max_attempts=1)

    def test_answer_contract_supports_set_list_numeric_and_boolean(self):
        self.assertTrue(answers_equal('["B","A"]', ["A", "B"], {"collection_mode": "set"}))
        self.assertFalse(answers_equal('["B","B"]', ["A", "B"], {"collection_mode": "set"}))
        self.assertTrue(answers_equal("1,000", 1000, {}))
        self.assertFalse(answers_equal("true", 1, {}))
        self.assertTrue(answers_equal("true", True, {}))

    def test_jsonl_plan_and_synthesis(self):
        question, world, rendered = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            qs = root/"questions.jsonl"
            qs.write_text(json.dumps(question)+"\n", encoding="utf-8")
            mapping = {"world-test": (world, rendered)}
            with patch("environment.trajectory_simulation.pipeline._world_map",
                       return_value=mapping):
                inspection = plan_trajectories_jsonl(
                    root/"worlds.jsonl", qs, root/"plan.jsonl")
                self.assertEqual(inspection["ready"], 1)
                production = synthesize_trajectories_jsonl(
                    root/"worlds.jsonl", qs, root/"episodes.jsonl",
                    planner=PassthroughPlanner(), executor=PassthroughExecutor(),
                    progress=False)
                self.assertEqual(production["accepted"], 1)
                output = json.loads((root/"episodes.jsonl").read_text().strip())
                self.assertTrue(output["validation"]["final_answer_verified"])
                self.assertEqual(output["gold_supervision"]["policy"],
                                 "first_stage_composition_links_fragment_gold")


if __name__ == "__main__":
    unittest.main()
