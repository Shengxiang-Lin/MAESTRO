"""Reject real omissions, retry format failures, and keep audits teacher-private."""
import json
import copy
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

from environment.question_generation.backend import BackendError, OpenAICompatibleBackend
from environment.tests.test_gold_trajectory import fixture
from environment.tests.test_round_synthesis import reviewed
from environment.trajectory_simulation.backend import LLMExecutor, LLMPlanner, PassthroughPlanner
from environment.trajectory_simulation.compiler import compile_gold_solution
from environment.trajectory_simulation.eligibility import (
    check_question, controlled_question_reference,
    semantic_review_steps, semantic_review_expression,
)
from environment.trajectory_simulation.rounds import compose_main_rounds
from environment.trajectory_simulation.subtask_bank import _report_natural, build_subtask_bank


class ReviewReliabilityTests(unittest.TestCase):
    def rendered_fixture(self):
        from environment.task_construction.rendering.pipeline import render_task_record
        q,r,_=fixture()
        t=render_task_record(r,seed=17)
        q.update(task_id=t.task_id,question=t.question)
        return compile_gold_solution(q,r,t,schedule="baseline"),r,t

    def test_v7_eligible_with_issues_requires_independent_confirmation(self):
        gold,record,rendered=self.rendered_fixture()
        gold["question"] = rendered.question + " Return the exact entity names."
        role=LLMPlanner(OpenAICompatibleBackend("test://local","test"))
        contradictory={
            "decision":"eligible","relation_coverage":[],
            "issues":[{
                "kind":"output","program_steps":[0],
                "question_quote":"Return the exact entity names.",
                "program_claim":"Return matching names.",
                "question_claim":"Return entity names.",
                "reason":"The question and program request the same output.",
            }],
        }
        role.check_question=Mock(side_effect=[contradictory,contradictory])
        role.confirm_question_consistency=Mock(return_value={
            "equivalent":True,
            "question_quote":"Return the exact entity names.",
            "explanation":"The complete constraints and output agree.",
        })
        result=check_question(gold,record,role,rendered=rendered)
        self.assertTrue(result["eligible"],str(result)[:1000])
        self.assertTrue(result["reviewed"])
        self.assertEqual(role.check_question.call_count,2)
        role.confirm_question_consistency.assert_called_once()
        self.assertTrue(any(
            a.get("stage")=="consistency_resolution"
            and a.get("result")=="independent_confirmed"
            for a in result["review_audit"]
        ))

    def test_v7_conflicting_or_unquoted_consistency_does_not_approve(self):
        for confirmation in (
            {"equivalent":False, "question_quote":"Return the exact entity names.",
             "explanation":"The scope is not the same."},
            {"equivalent":True,"question_quote":"invented span",
             "explanation":"No actual question witness."},
        ):
            with self.subTest(confirmation=confirmation):
                gold,record,rendered=self.rendered_fixture()
                gold["question"]=rendered.question+" Return the exact entity names."
                role=LLMPlanner(OpenAICompatibleBackend("test://local","test"))
                contradictory={"decision":"eligible","issues":[{
                    "kind":"output","program_steps":[0],
                    "question_quote":"Return the exact entity names.",
                    "program_claim":"Return names.","question_claim":"Return names.",
                    "reason":"These agree.",
                }],"relation_coverage":[]}
                role.check_question=Mock(side_effect=[contradictory,contradictory])
                role.confirm_question_consistency=Mock(return_value=confirmation)
                result=check_question(gold,record,role,rendered=rendered)
                self.assertFalse(result["eligible"])
                self.assertEqual(result["review_failure_kind"],"eligible_with_issues")

    def test_paired_question_audit_direct_cli_imports_project_package(self):
        """Reproduce the user's direct-file invocation without PYTHONPATH."""
        import os
        from pathlib import Path
        import subprocess
        import sys
        import tempfile
        from environment.task_construction.rendering.pipeline import render_task_record

        _,world,_=fixture()
        rendered=render_task_record(world,seed=17)
        root=Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)
            bank_file=directory/"bank.jsonl"
            quarantine_file=directory/"quarantine.jsonl"
            stage2_dir=directory/"questions_sample"
            stage2_dir.mkdir()
            realized_file=stage2_dir/"realized_capability_worlds.jsonl"
            critic_file=stage2_dir/"question_critic_audit.jsonl"
            worlds_file=directory/"typed_worlds.jsonl"
            result_file=directory/"comparison.jsonl"
            def put(path, items):
                path.write_text(
                    "".join(json.dumps(item,ensure_ascii=False)+"\n" for item in items),
                    encoding="utf-8",
                )
            put(bank_file,[{"bank":{
                "task_id":rendered.task_id,
                "eligibility":{"eligible":True,"reviewed":True},
            }}])
            put(quarantine_file,[])
            put(worlds_file,[world])
            put(realized_file,[{
                "task_id":rendered.task_id,"question":rendered.question,
                "draft_question":rendered.question,"authoring":{"audit":[]},
            }])
            put(critic_file,[{
                "task_id":rendered.task_id,"accepted":True,"audit":[],
            }])
            env=os.environ.copy()
            env.pop("PYTHONPATH",None)
            invocation=[
                sys.executable,str(root/"environment/audit_stage2_questions.py"),
                "--bank",str(bank_file),
                "--quarantine",str(quarantine_file),
                "--realized",str(realized_file),
                "--worlds",str(worlds_file),
                "--max-records","1",
                "--output",str(result_file),
            ]
            run=subprocess.run(
                invocation,cwd=root,env=env,text=True,
                capture_output=True,timeout=30,check=False,
            )
            self.assertEqual(run.returncode,0,run.stderr)
            report=json.loads(run.stdout)
            self.assertEqual(report["audited"],1)
            self.assertEqual(report["accepted"],1)
            result=json.loads(result_file.read_text(encoding="utf-8").strip())
            self.assertEqual(result["task_id"],rendered.task_id)
            self.assertEqual(result["programir"],world["program"])
            self.assertTrue(result["same_as_renderer"])

    def test_reconstructed_supported_question_does_not_need_llm_vote(self):
        gold,record,rendered=self.rendered_fixture()
        role=Mock(offline=False,backend=Mock())
        result=check_question(gold,record,role,rendered=rendered)
        self.assertTrue(result["eligible"])
        self.assertTrue(result["reviewed"])
        self.assertEqual(result["method"],"verified_controlled_renderer")
        self.assertEqual(result["renderer_certificate"]["verified_node_ids"],
                         [0,1,2,3,4])
        role.check_question.assert_not_called()

    def test_rewrite_is_reviewed_even_with_forged_canonical_question(self):
        gold,record,rendered=self.rendered_fixture()
        gold["question"]="Find entities associated with Alice."
        gold["canonical_question"]=gold["question"]
        role=Mock(offline=False,backend=Mock())
        role.check_question.return_value=self.mismatch(gold,record)
        result=check_question(gold,record,role,rendered=rendered)
        self.assertFalse(result["eligible"])
        self.assertEqual(result["method"],"llm_semantic_screen")
        prompt=role.check_question.call_args.args[0]
        self.assertIn("changed_spans",prompt)
        self.assertIn(rendered.question,prompt)

    def test_changed_program_cannot_reuse_old_renderer_text_certificate(self):
        gold,record,rendered=self.rendered_fixture()
        changed=copy.deepcopy(record)
        changed["program"]["nodes"][1]["inputs"][0]="r:second"
        reference=controlled_question_reference(gold,changed,rendered)
        # Signature-based task identity changes too; either absence of a
        # reference or a different question precludes the exact-match route.
        self.assertTrue(reference is None or reference["question"]!=gold["question"])

    def test_renderer_certificate_refuses_unconsumed_branch_and_unsupported_op(self):
        gold,record,rendered=self.rendered_fixture()
        changed=copy.deepcopy(record)
        changed["program"]["nodes"].append(
            {"node_id":5,"op":"FIND","dependencies":[],"inputs":["entity:A"]})
        self.assertIsNone(controlled_question_reference(gold,changed,rendered))
        changed=copy.deepcopy(record)
        changed["program"]["nodes"][-1]["op"]="QUERY_ATTR_QUALIFIER"
        self.assertIsNone(controlled_question_reference(gold,changed,rendered))

    def test_renderer_certificate_checks_every_use_of_hidden_identity(self):
        from environment.task_construction.rendering.query_grounding import (
            LatentEntityPlan,PublicQueryPlan,
        )
        gold,record,rendered=self.rendered_fixture()
        unsafe=PublicQueryPlan((LatentEntityPlan(
            "A","schema:Person",(0,),"unique_attribute_clue",(1,),()),))
        with patch("environment.task_construction.rendering.query_grounding.build_public_query_plan",
                   return_value=unsafe):
            self.assertIsNone(controlled_question_reference(gold,record,rendered))

    def test_identical_mismatch_claims_request_review_instead_of_approval(self):
        gold,record,role=self.review_fixture()
        verdict=self.mismatch(gold,record)
        verdict["issues"][0]["question_claim"]=verdict["issues"][0]["program_claim"]
        role.check_question.return_value=verdict
        result=check_question(gold,record,role)
        self.assertFalse(result["eligible"])
        self.assertEqual(result["review_failure_kind"],"contradictory_mismatch")
        self.assertEqual(role.check_question.call_count,2)

    def test_expansion_keeps_inherited_filters_and_separate_branch_scopes(self):
        nodes={
            0:{"op":"FIND_ALL","inputs":[],"dependencies":[]},
            1:{"op":"FILTER_YEAR","inputs":["birth date","<","1840"],"dependencies":[0]},
            2:{"op":"RELATE","inputs":["winner","backward"],"dependencies":[1]},
            3:{"op":"FILTER_YEAR","inputs":["start date",">","1915"],"dependencies":[1]},
            4:{"op":"INTERSECT","inputs":[],"dependencies":[2,3]},
        }
        steps=semantic_review_steps(nodes)
        text=semantic_review_expression(steps)
        self.assertEqual(text.count("whose own birth date < 1840"),2)
        self.assertIn("subjects x",text)
        self.assertIn("Intersection",text)
        self.assertNotRegex(text,r"\bS\d+\b")
        self.assertIsNone(semantic_review_expression(steps,max_chars=20))

    def test_relation_endpoints_filter_owners_and_selection_union_are_explicit(self):
        nodes={
            0:{"op":"FIND_ALL","inputs":[],"dependencies":[]},
            1:{"op":"FILTER_YEAR","inputs":["birth date","<","1840"],"dependencies":[0]},
            2:{"op":"RELATE","inputs":["winner","backward"],"dependencies":[1]},
            3:{"op":"RELATE","inputs":["neighbor","forward"],"dependencies":[1]},
            4:{"op":"SELECT_BETWEEN","inputs":["female population",">"],"dependencies":[2,3]},
            5:{"op":"QUERY_ATTR","inputs":["start date"],"dependencies":[4]},
            6:{"op":"VERIFY_YEAR","inputs":[">","1915"],"dependencies":[5]},
        }
        steps={s["step"]:s for s in semantic_review_steps(nodes)}
        self.assertIn("whose own birth date",steps[1]["meaning"])
        self.assertIn("subjects x",steps[2]["meaning"])
        self.assertIn("y in S1",steps[2]["meaning"])
        self.assertIn("objects y",steps[3]["meaning"])
        self.assertIn("x in S1",steps[3]["meaning"])
        self.assertIn("union of S2, S3",steps[4]["meaning"])
        self.assertIn("maximum",steps[4]["meaning"])
        self.assertIn("not a threshold filter",steps[4]["meaning"])
        self.assertIn("recorded start date values",steps[5]["meaning"])
        self.assertIn("any value in S5 > 1915",steps[6]["meaning"])

    def review_fixture(self):
        q,r,t=fixture()
        gold=compile_gold_solution(q,r,t)
        gold["question"]+=" Check first and second."
        role=Mock(offline=False,backend=Mock())
        return gold,r,role

    def mismatch(self,gold,record):
        return {"decision":"mismatch","issues":[{
            "kind":"missing_constraint",
            "program_steps":[record["program"]["nodes"][0]["node_id"]],
            "question_quote":gold["question"][:80],
            "program_claim":"A required relation is traversed.",
            "question_claim":"The wording omits that relation.",
            "reason":"Required hop omitted.",
        }]}

    def test_negative_requires_recheck_of_full_input(self):
        gold,record,role=self.review_fixture()
        role.check_question.side_effect=[
            self.mismatch(gold,record),{"decision":"eligible","issues":[]},
        ]
        result=check_question(gold,record,role)
        self.assertTrue(result["eligible"])
        self.assertEqual(len(result["review_audit"]),2)
        prompt=role.check_question.call_args.args[0]
        self.assertIn(gold["question"],prompt)
        self.assertIn("Previous review:",prompt)
        self.assertIn("FIND_ALL",prompt)

    def test_real_omission_is_retained_even_with_identical_canonical_wording(self):
        gold,record,role=self.review_fixture()
        gold["canonical_question"]=gold["question"]
        role.check_question.return_value=self.mismatch(gold,record)
        result=check_question(gold,record,role)
        self.assertFalse(result["eligible"])
        self.assertTrue(result["reviewed"])
        self.assertEqual(result["decision"],"mismatch")
        self.assertEqual(role.check_question.call_count,2)

    def test_invalid_evidence_never_becomes_a_semantic_mismatch(self):
        gold,record,role=self.review_fixture()
        invalid=self.mismatch(gold,record)
        invalid["issues"][0]["program_steps"]=[999999]
        role.check_question.return_value=invalid
        result=check_question(gold,record,role)
        self.assertFalse(result["eligible"])
        self.assertFalse(result["reviewed"])
        self.assertEqual(result["decision"],"review_error")
        self.assertEqual(role.check_question.call_count,2)

    def test_uncertain_does_not_get_voted_into_approval(self):
        gold,record,role=self.review_fixture()
        role.check_question.return_value={"decision":"uncertain","issues":[]}
        result=check_question(gold,record,role)
        self.assertEqual(result["decision"],"uncertain")
        self.assertFalse(result["eligible"])
        self.assertEqual(role.check_question.call_count,1)

    def test_approval_without_evidence_for_unstated_hops_is_quarantined(self):
        gold,record,role=self.review_fixture()
        gold["question"]="Find the result for Alice."
        role.check_question.return_value={"decision":"eligible","issues":[]}
        result=check_question(gold,record,role)
        self.assertFalse(result["eligible"])
        self.assertEqual(result["decision"],"review_error")
        self.assertEqual(role.check_question.call_count,3)
        self.assertIn("REQUIRED-HOP",role.check_question.call_args.args[0])

    def test_focused_coverage_audit_retains_real_missing_hop(self):
        gold,record,role=self.review_fixture()
        gold["question"]="Find the result for Alice."
        role.check_question.side_effect=[
            {"decision":"eligible","issues":[]},self.mismatch(gold,record),
        ]
        result=check_question(gold,record,role)
        self.assertFalse(result["eligible"])
        self.assertEqual(result["decision"],"mismatch")

    def test_relation_paraphrases_can_pass_with_exact_coverage_evidence(self):
        gold,record,role=self.review_fixture()
        gold["question"]="Follow the primary link and the secondary link for Alice."
        role.check_question.side_effect=[
            {"decision":"eligible","issues":[]},
            {"decision":"eligible","issues":[],"relation_coverage":[
                {"step":1,"question_quote":"primary link","interpretation":"primary expresses first"},
                {"step":2,"question_quote":"secondary link","interpretation":"secondary expresses second"},
            ]},
        ]
        self.assertTrue(check_question(gold,record,role)["eligible"])

    def test_question_json_retry_respects_configured_output_budget(self):
        role=LLMPlanner(OpenAICompatibleBackend("test://local","test",max_tokens=3072))
        calls=[]
        def complete(client,*args,**kwargs):
            calls.append((client.max_tokens,kwargs.get("response_schema")))
            if len(calls)==1:raise BackendError("model did not return a JSON object")
            return {"decision":"eligible","issues":[]}
        audit=[]
        with patch.object(OpenAICompatibleBackend,"complete_json",complete):
            self.assertEqual(role.check_question("test",audit=audit)["decision"],"eligible")
        self.assertEqual([budget for budget,_ in calls],[3072,3072])
        self.assertTrue(all(schema and schema["additionalProperties"] is False
                            for _,schema in calls))
        self.assertEqual(role.request_counts["question_review_retry"],1)
        self.assertEqual(len(audit),1)

    def test_network_error_is_not_retried_as_bad_json(self):
        role=LLMPlanner(OpenAICompatibleBackend("test://local","test"))
        with patch.object(OpenAICompatibleBackend,"complete_json",
                          side_effect=BackendError("OpenAI-compatible call failed: timeout")) as complete:
            with self.assertRaises(BackendError):role.check_question("test")
        self.assertEqual(complete.call_count,1)

    def payload(self,name):
        return {"task_instruction":"Find "+name+".","tool_provenance":{"steps":[]},
                "fragment":{"expected_answer":[name],
                            "operations":[{"op":"FIND","inputs":[name]}],
                            "operation_results":[{"answer":[name],"input_values":[]}],
                            "observations":[],"known_observations":[]}}


    def test_contradictory_mismatch_witness_fails_closed(self):
        """v3 de510e6...: 'No discrepancy found' cannot certify mismatch."""
        gold,record,role=self.review_fixture()
        contradictory=self.mismatch(gold,record)
        contradictory["issues"][0]["reason"]=(
            "The wording matches the computation exactly. No discrepancy found."
        )
        role.check_question.return_value=contradictory
        result=check_question(gold,record,role)
        self.assertFalse(result["eligible"])
        self.assertFalse(result["reviewed"])
        self.assertEqual(result["decision"],"review_error")
        self.assertEqual(result["review_failure_kind"],"contradictory_mismatch")
        self.assertEqual(role.check_question.call_count,2)

    def test_fabricated_ellipsis_is_not_an_exact_question_witness(self):
        """v3 6f422...: a shortened quote with '...' is not in the input."""
        gold,record,role=self.review_fixture()
        wrong=self.mismatch(gold,record)
        wrong["issues"][0]["question_quote"]="entities connected ... by neighbor"
        role.check_question.return_value=wrong
        result=check_question(gold,record,role)
        self.assertFalse(result["eligible"])
        self.assertEqual(result["decision"],"review_error")
        self.assertEqual(result["review_failure_kind"],"invalid_mismatch_quote")

    def test_relation_coverage_is_chunked_and_all_hops_are_checked(self):
        """More than two missing literal relations require separate reviews."""
        gold,record,role=self.review_fixture()
        record["program"]["nodes"].append({
            "node_id":99,"op":"RELATE","dependencies":[2],
            "inputs":["r:third","forward"],
        })
        gold["question"]="Follow the primary, middle and tertiary link for Alice."
        role.check_question.side_effect=[
            {"decision":"eligible","issues":[]},
            {"decision":"eligible","issues":[],"relation_coverage":[
                {"step":1,"question_quote":"primary","interpretation":"first hop"},
                {"step":2,"question_quote":"middle","interpretation":"second hop"},
            ]},
            {"decision":"eligible","issues":[],"relation_coverage":[
                {"step":99,"question_quote":"tertiary","interpretation":"third hop"},
            ]},
        ]
        result=check_question(gold,record,role)
        self.assertTrue(result["eligible"], json.dumps(result,ensure_ascii=False,default=str)[:3500])
        self.assertEqual(role.check_question.call_count,3)
        self.assertEqual(
            len([a for a in result["review_audit"] if a["stage"]=="focused_hop_validation"]),
            2,
        )

    def test_reused_or_missing_relation_witness_stays_quarantined(self):
        gold,record,role=self.review_fixture()
        gold["question"]="Follow a link for Alice."
        incomplete={"decision":"eligible","issues":[],"relation_coverage":[
            {"step":1,"question_quote":"link","interpretation":"first hop"},
            {"step":2,"question_quote":"link","interpretation":"second hop"},
        ]}
        role.check_question.side_effect=[
            {"decision":"eligible","issues":[]},incomplete,incomplete,
        ]
        result=check_question(gold,record,role)
        self.assertFalse(result["eligible"])
        self.assertFalse(result["reviewed"])
        self.assertEqual(result["decision"],"review_error")
        self.assertEqual(result["review_failure_kind"],"reused_relation_coverage")
        self.assertEqual(role.check_question.call_count,3)

    def test_truncated_review_is_not_classified_as_semantic_mismatch(self):
        gold,record,role=self.review_fixture()
        error=BackendError("model returned invalid JSON")
        error.diagnostic={"finish_reason":"length","usage":{"completion_tokens":2048}}
        role.check_question.side_effect=error
        result=check_question(gold,record,role)
        self.assertEqual(result["decision"],"review_error")
        self.assertEqual(result["review_failure_kind"],"review_output_truncated")
        self.assertFalse(result["eligible"])
        self.assertFalse(result["reviewed"])

    def test_feedback_json_writer_and_reviewer_get_bounded_retries(self):
        backend=Mock()
        candidate={"summary":"Alice is the result.","trace_summary":"Returned Alice."}
        backend.complete_json.side_effect=[
            BackendError("model returned invalid JSON"),candidate,
            {"grounded":"yes","issues":[]},{"grounded":True,"issues":[]},
        ]
        executor=LLMExecutor(backend)
        audit=[]
        self.assertEqual(executor.naturalize_feedback(self.payload("Alice"),{},audit=audit),candidate)
        self.assertEqual(backend.complete_json.call_count,4)
        self.assertEqual(executor.request_counts["natural_feedback_retry"],1)
        self.assertEqual(executor.request_counts["natural_feedback_review_retry"],1)
        self.assertIn(candidate,[item.get("candidate") for item in audit])
        self.assertIn({"grounded":"yes","issues":[]},[item.get("verdict") for item in audit])

    def test_feedback_reviewer_issues_trigger_one_grounded_writer_repair(self):
        """Stage 2 style critic feedback is a hypothesis, not a new fact."""
        first={"summary":"Alice is the result.",
               "trace_summary":"Returned Alice from the assigned record."}
        repaired={"summary":"Alice is the result.",
                  "trace_summary":"Verified the assigned result Alice without adding claims."}
        backend=Mock()
        backend.complete_json.side_effect=[
            first,
            {"grounded":False,"issues":["Source attribution should be checked."]},
            repaired,
            {"grounded":True,"issues":[]},
        ]
        executor=LLMExecutor(backend)
        audit=[]
        result=executor.naturalize_feedback(
            self.payload("Alice"),{},audit=audit,
        )
        self.assertEqual(result,repaired)
        self.assertEqual(backend.complete_json.call_count,4)
        self.assertEqual(executor.request_counts["natural_feedback_write"],1)
        self.assertEqual(executor.request_counts["natural_feedback_content_repair"],1)
        self.assertEqual(executor.request_counts["natural_feedback_content_repair_success"],1)
        self.assertEqual(executor.request_counts["natural_feedback_retry"],0)
        repair_prompt=backend.complete_json.call_args_list[2].args[1]
        self.assertIn("UNTRUSTED DIAGNOSTICS",repair_prompt)
        self.assertIn("Source attribution should be checked.",repair_prompt)
        self.assertIn(first["trace_summary"],repair_prompt)
        self.assertIn("fixed_result",repair_prompt)
        self.assertIn("assigned_subtask",repair_prompt)
        self.assertIn("execution_reference",repair_prompt)
        self.assertIn("content_repair_review",[e["stage"] for e in audit])

    def test_false_critic_issue_does_not_override_fixed_answer(self):
        backend=Mock()
        supported={"summary":"Alice is the result.",
                   "trace_summary":"The verified result is Alice."}
        backend.complete_json.side_effect=[
            supported,
            {"grounded":False,"issues":["Alice is unsupported; replace with Bob."]},
            supported,
            {"grounded":True,"issues":[]},
        ]
        executor=LLMExecutor(backend)
        result=executor.naturalize_feedback(self.payload("Alice"),{})
        self.assertEqual(result,supported)
        self.assertEqual(executor.request_counts["natural_feedback_content_repair_success"],1)

    def test_critic_repair_hallucination_fails_closed(self):
        backend=Mock()
        backend.complete_json.side_effect=[
            {"summary":"Alice is the result.","trace_summary":"Returned Alice."},
            {"grounded":False,"issues":["Check factual attribution."]},
            {"summary":"Bob is the result.","trace_summary":"Returned Bob."},
        ]
        executor=LLMExecutor(backend)
        with self.assertRaisesRegex(ValueError,"omits_result_entity"):
            executor.naturalize_feedback(self.payload("Alice"),{})
        self.assertEqual(backend.complete_json.call_count,3)
        self.assertEqual(executor.request_counts["natural_feedback_content_repair_failed"],1)

    def test_failed_critic_repair_returns_canonical_report(self):
        backend=Mock()
        backend.complete_json.side_effect=[
            {"summary":"Alice is the result.","trace_summary":"Returned Alice."},
            {"grounded":False,"issues":["Check factual attribution."]},
            {"summary":"Alice is the result.","trace_summary":"Returned Alice."},
            {"grounded":False,"issues":["Still unsupported."]},
        ]
        executor=LLMExecutor(backend)
        executor.style_provider="natural"
        payload=self.payload("Alice")
        template={"summary":"Verified Alice.","trace_summary":"Canonical Alice."}
        report,meta=_report_natural(executor,payload,template,payload["fragment"])
        self.assertEqual(report,template)
        self.assertFalse(meta["natural"])
        self.assertIn("natural_semantic_review_failed",meta["reason"])
        self.assertEqual(backend.complete_json.call_count,4)
        self.assertEqual(executor.request_counts["natural_feedback_content_repair_failed"],1)

    def test_critic_without_specific_issues_does_not_trigger_repair(self):
        backend=Mock()
        backend.complete_json.side_effect=[
            {"summary":"Alice is the result.","trace_summary":"Returned Alice."},
            {"grounded":False,"issues":[]},
        ]
        executor=LLMExecutor(backend)
        with self.assertRaisesRegex(ValueError,"natural_semantic_review_failed"):
            executor.naturalize_feedback(self.payload("Alice"),{})
        self.assertEqual(backend.complete_json.call_count,2)
        self.assertEqual(executor.request_counts["natural_feedback_content_repair"],0)

    def test_parallel_failed_candidates_have_separate_complete_audits(self):
        class Backend:
            def complete_json(self,system,prompt):
                if "Strictly verify" in system:
                    return {"grounded":False,"issues":["PRIVATE_CANARY"]}
                name="Alice" if "Alice" in prompt else "Bob"
                return {"summary":name+" is the result.","trace_summary":"Returned "+name+"."}
        executor=LLMExecutor(Backend())
        executor.style_provider="natural"
        def report(name):
            payload=self.payload(name)
            return _report_natural(executor,payload,{"summary":"template"},payload["fragment"])[1]
        with ThreadPoolExecutor(max_workers=2) as pool:
            alice,bob=list(pool.map(report,["Alice","Bob"]))
        for name,result in (("Alice",alice),("Bob",bob)):
            self.assertFalse(result["natural"])
            self.assertEqual(result["audit"][0]["candidate"]["summary"],name+" is the result.")
            self.assertEqual(result["audit"][1]["verdict"]["issues"],["PRIVATE_CANARY"])
        self.assertIsNot(alice["audit"],bob["audit"])

    def test_bank_diagnostics_do_not_enter_main_training_or_public_history(self):
        q,r,t=fixture()
        with patch("environment.trajectory_simulation.subtask_bank.check_question",reviewed):
            bank=build_subtask_bank(q,r,t,planner=PassthroughPlanner(),
                                   executor=LLMExecutor(Mock()),schedule="baseline")
        for item in bank["fragments"].values():
            item["feedback_private_audit"]=[{"candidate":"PRIVATE_CANARY"}]
        bank["eligibility"]["review_audit"]=[{"verdict":"PRIVATE_CANARY"}]
        self.assertNotIn("PRIVATE_CANARY",json.dumps(bank["full_public_history"]))
        output=compose_main_rounds(bank,planner=PassthroughPlanner(),include_fallback_rounds=True)
        self.assertNotIn("PRIVATE_CANARY",json.dumps(output[0]))

    def test_backend_parse_failure_retains_finish_reason_and_content(self):
        response=Mock()
        response.read.return_value=json.dumps({
            "choices":[{"finish_reason":"length","message":{"content":"broken"}}],
            "usage":{"completion_tokens":2048},
        }).encode()
        context=Mock()
        context.__enter__=Mock(return_value=response)
        context.__exit__=Mock(return_value=False)
        with patch("urllib.request.urlopen",return_value=context):
            with self.assertRaises(BackendError) as failure:
                OpenAICompatibleBackend("http://test/v1","test").complete_json("system","prompt")
        self.assertEqual(failure.exception.diagnostic,{
            "finish_reason":"length","usage":{"completion_tokens":2048},"content":"broken",
        })
