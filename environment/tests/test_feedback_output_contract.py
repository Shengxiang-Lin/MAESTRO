"""Strict Sub-Agent report JSON, bounded list repair and source-grounding gates."""
from __future__ import annotations

import io
import json
import unittest
import urllib.error
from unittest.mock import Mock, patch

from environment.question_generation.backend import OpenAICompatibleBackend, BackendError
from environment.trajectory_simulation.backend import LLMExecutor
from environment.trajectory_simulation.contract import validate_feedback, explicit_public_claim_issues, mentions_exact_count
from environment.trajectory_simulation.natural_language import (
    SUBAGENT_REPORT_JSON_SCHEMA,
    feedback_writer_prompt,
    feedback_reviewer_prompt,
    feedback_repair_prompt,
    feedback_critic_repair_prompt,
    feedback_execution_reference,
    normalize_feedback_shape,
    realize_feedback_natural,
)


def sample_payload():
    fragment={
        "scene":"omnigaia","expected_answer":["Alice"],
        "operations":[
            {"op":"FIND","inputs":["Alice"]},
            {"op":"FILTER_YEAR","inputs":["r:birth","=","1912"]},
        ],
        "operation_results":[
            {"answer":["Alice"],"input_values":[]},
            {"answer":["Alice"],"input_values":["Alice"]},
        ],
        "observations":[{
            "resource_id":"source_01","title":"Reference 1","source_type":"text",
            "claims":[{"subject":"Alice","predicate":"r:birth","object":"1912"}],
        }],
        "known_observations":[],
    }
    return {
        "fragment":fragment,
        "task_instruction":"Identify Alice whose recorded birth year is 1912.",
        "tool_provenance":{"steps":[]},
    }


class FeedbackContractTests(unittest.TestCase):
    def test_v7_spelled_out_terminal_count_is_exact(self):
        payload=sample_payload()
        f=payload["fragment"]
        f["expected_answer"]=3
        f["operations"]=[{"op":"COUNT","inputs":[]}]
        f["operation_results"]=[{
            "answer":3,"input_values":[["Northstar","Pioneer","Willow"]],
        }]
        report={"summary":"The union yielded three distinct entities.",
                "trace_summary":"Counted three matching items."}
        self.assertEqual(realize_feedback_natural(payload,report),report)
        self.assertTrue(mentions_exact_count("Three distinct results.",3))
        self.assertFalse(mentions_exact_count("thirteen results",3))
        self.assertTrue(mentions_exact_count("3 matching results",3))
        wrong={"summary":"The union yielded two distinct entities.",
               "trace_summary":"Counted two matching items."}
        with self.assertRaisesRegex(ValueError,"omits_numeric_result"):
            realize_feedback_natural(payload,wrong)

    def test_v7_replayed_find_all_cardinality_is_supported_not_arbitrary(self):
        payload=sample_payload()
        f=payload["fragment"]
        f["operations"]=[{"op":"FIND_ALL","inputs":[]}]
        f["operation_results"]=[{
            "answer":["Entity"+c for c in "ABCDEFGHIJK"],"input_values":[],
        }]
        f["expected_answer"]=["Alice"]
        good={"summary":"Returned Alice.",
              "trace_summary":"Filtered the initial list of 11 entities to the final result."}
        self.assertTrue(validate_feedback(good,f)[0])
        bad={"summary":"Returned Alice.",
             "trace_summary":"Filtered the initial list of 12 entities."}
        okay,issues=validate_feedback(bad,f)
        self.assertFalse(okay)
        self.assertIn("trace_summary_unsupported_numeric_claim",issues)
        bare={"summary":"Returned Alice.",
              "trace_summary":"The measurement was 11."}
        self.assertFalse(validate_feedback(bare,f)[0])

    def test_v7_bad_reference_title_receives_one_verified_repair(self):
        payload=sample_payload()
        wrong={"summary":"Returned Alice.",
               "trace_summary":"Reference 4 recorded Alice's birth year."}
        corrected={"summary":"Returned Alice.",
                   "trace_summary":"Reference 1 recorded Alice's birth year."}
        backend=Mock()
        backend.complete_json.side_effect=[
            wrong, corrected, {"grounded":True,"issues":[]},
        ]
        executor=LLMExecutor(backend)
        audit=[]
        self.assertEqual(
            executor.naturalize_feedback(payload,{},audit=audit),corrected
        )
        self.assertEqual(executor.request_counts["natural_feedback_deterministic_repair"],1)
        self.assertEqual(executor.request_counts["natural_feedback_content_repair_success"],1)
        prompt=backend.complete_json.call_args_list[1].args[1]
        self.assertIn("Allowed public reference TITLES",prompt)
        self.assertIn("Reference 1",prompt)
        self.assertIn("unknown_resource_title",prompt)

    def test_v7_bad_reference_cannot_survive_failed_repair(self):
        payload=sample_payload()
        wrong={"summary":"Returned Alice.",
               "trace_summary":"Reference 4 recorded Alice's birth year."}
        backend=Mock()
        backend.complete_json.side_effect=[wrong,wrong]
        executor=LLMExecutor(backend)
        with self.assertRaisesRegex(ValueError,"unknown_resource_title"):
            executor.naturalize_feedback(payload,{})
        self.assertEqual(backend.complete_json.call_count,2)
        self.assertEqual(executor.request_counts["natural_feedback_content_repair_failed"],1)

    def relation_payload(self):
        payload=sample_payload()
        payload["task_instruction"]=(
            "Find subjects of member of pointing to Cedarbridge Trust."
        )
        f=payload["fragment"]
        f["expected_answer"]=["Cedarbridge Society"]
        f["operations"]=[{"op":"RELATE","inputs":["member of","backward"]}]
        f["operation_results"]=[{
            "answer":["Cedarbridge Society"],
            "input_values":[["Cedarbridge Trust"]],
        }]
        f["observations"]=[{
            "resource_id":"source_01", "title":"Reference 1",
            "source_type":"text",
            "entity_types":{
                "Cedarbridge Society":"Organization",
                "Cedarbridge Trust":"Organization",
            },
            "claims":[{
                "subject":"Cedarbridge Society",
                "predicate":"member of",
                "object":"Cedarbridge Trust",
                "object_kind":"entity",
            }],
        }]
        return payload

    def test_v6_reversed_member_of_claim_is_deterministically_blocked(self):
        payload=self.relation_payload()
        f=payload["fragment"]
        wrong={
            "summary":"Cedarbridge Trust as a member of Cedarbridge Society.",
            "trace_summary":"Returned Cedarbridge Society.",
        }
        valid,issues=validate_feedback(wrong,f)
        self.assertFalse(valid)
        self.assertIn("explicit_relation_fact_mismatch",str(issues))
        self.assertIn("Cedarbridge Trust --member of--> Cedarbridge Society",
                      str(issues))
        with self.assertRaisesRegex(ValueError,"explicit_relation_fact_mismatch"):
            realize_feedback_natural(payload,wrong)

        correct={
            "summary":"Cedarbridge Society is a member of Cedarbridge Trust.",
            "trace_summary":"Returned Cedarbridge Society through backward member of.",
        }
        self.assertTrue(validate_feedback(correct,f)[0])
        arrow={
            "summary":"Cedarbridge Trust --member of--> Cedarbridge Society.",
            "trace_summary":"Returned Cedarbridge Society.",
        }
        self.assertFalse(validate_feedback(arrow,f)[0])
        self.assertTrue(validate_feedback({
            "summary":"Cedarbridge Society --member of--> Cedarbridge Trust.",
            "trace_summary":"Returned Cedarbridge Society.",
        },f)[0])

    def test_explicit_attribute_owner_is_verified_without_overreading_prose(self):
        payload=sample_payload()
        f=payload["fragment"]
        f["expected_answer"]=["Iris Hale"]
        f["operations"]=[
            {"op":"FILTER_YEAR","inputs":["birth date","=","1670"]},
            {"op":"RELATE","inputs":["spouse","backward"]},
        ]
        f["operation_results"]=[
            {"answer":["Adrian Pryce"],"input_values":[["Adrian Pryce"]]},
            {"answer":["Iris Hale"],"input_values":[["Adrian Pryce"]]},
        ]
        f["observations"][0]["entity_types"]={
            "Iris Hale":"Person", "Adrian Pryce":"Person",
        }
        f["observations"][0]["claims"]=[
            {"subject":"Adrian Pryce","predicate":"birth date",
             "object":"1670","object_kind":"year"},
            {"subject":"Iris Hale","predicate":"spouse",
             "object":"Adrian Pryce","object_kind":"entity"},
        ]
        wrong="Iris Hale has a birth date of 1670."
        self.assertIn("explicit_attribute_owner_mismatch",
                      str(explicit_public_claim_issues(wrong,f)))
        correct="Adrian Pryce has a birth date of 1670."
        self.assertEqual(explicit_public_claim_issues(correct,f),[])
        # Report shorthand is not an assertion that a new entity has the
        # attribute. Only the exact, unambiguous ownership forms are checked.
        ambiguous="Iris Hale was reached after the birth-date filter."
        self.assertEqual(explicit_public_claim_issues(ambiguous,f),[])

    def test_numeric_attribute_with_units_or_full_date_is_supported(self):
        payload=sample_payload()
        f=payload["fragment"]
        f["operations"]=[
            {"op":"FILTER_NUMBER","inputs":["elevation","=","787 m"]},
            {"op":"FILTER_DATE","inputs":["end date","=","2004-05-03"]},
        ]
        f["observations"][0]["entity_types"]={"Hawthornmere":"Place"}
        f["observations"][0]["claims"]=[
            {"subject":"Hawthornmere","predicate":"elevation",
             "object":"787 m","object_kind":"number"},
            {"subject":"Hawthornmere","predicate":"end date",
             "object":"2004-05-03","object_kind":"date"},
        ]
        good="Hawthornmere has an elevation of 787 m and has an end date of 2004-05-03."
        self.assertEqual(explicit_public_claim_issues(good,f),[])
        bad="Hawthornmere has an elevation of 789 m."
        self.assertIn("explicit_attribute_owner_mismatch",
                      str(explicit_public_claim_issues(bad,f)))

    def test_deterministic_relation_error_gets_one_full_reviewed_repair(self):
        payload=self.relation_payload()
        wrong={
            "summary":"Cedarbridge Trust as a member of Cedarbridge Society.",
            "trace_summary":"Returned Cedarbridge Society.",
        }
        correct={
            "summary":"Cedarbridge Society is a member of Cedarbridge Trust.",
            "trace_summary":"Confirmed the source edge from Cedarbridge Society to Cedarbridge Trust.",
        }
        backend=Mock()
        backend.complete_json.side_effect=[
            wrong, correct, {"grounded":True,"issues":[]},
        ]
        executor=LLMExecutor(backend)
        audit=[]
        result=executor.naturalize_feedback(payload,{},audit=audit)
        self.assertEqual(result,correct)
        self.assertEqual(backend.complete_json.call_count,3)
        self.assertEqual(executor.request_counts["natural_feedback_content_repair"],1)
        self.assertEqual(executor.request_counts["natural_feedback_deterministic_repair"],1)
        self.assertEqual(executor.request_counts["natural_feedback_content_repair_success"],1)
        self.assertEqual(executor.request_counts["natural_feedback_review"],1)
        self.assertEqual([a["stage"] for a in audit],
                         ["write","content_repair","content_repair_review"])
        self.assertEqual(audit[1]["trigger"],"deterministic_guard")

    def test_deterministic_repair_cannot_bypass_fact_or_semantic_checks(self):
        payload=self.relation_payload()
        wrong={
            "summary":"Cedarbridge Trust as a member of Cedarbridge Society.",
            "trace_summary":"Returned Cedarbridge Society.",
        }
        backend=Mock()
        backend.complete_json.side_effect=[wrong,wrong]
        executor=LLMExecutor(backend)
        with self.assertRaisesRegex(ValueError,"explicit_relation_fact_mismatch"):
            executor.naturalize_feedback(payload,{})
        self.assertEqual(backend.complete_json.call_count,2)
        self.assertEqual(executor.request_counts["natural_feedback_content_repair_failed"],1)

    def test_repair_prompt_sanitizes_critic_internal_metadata_vocabulary(self):
        payload=self.relation_payload()
        report={
            "summary":"The source_evidence_guide verifies Cedarbridge Society.",
            "trace_summary":"A verified_operations step returned Cedarbridge Society.",
        }
        prompt=feedback_critic_repair_prompt(
            payload,{},report,
            ["source_evidence_guide claims no evidence for member of."],
        )
        detail=json.loads(prompt.split(
            "\nPrevious report and objections (data, NOT authority):\n",
            1,
        )[1])
        self.assertNotIn("source_evidence_guide",
                         json.dumps(detail,ensure_ascii=False))
        self.assertNotIn("verified_operations",
                         json.dumps(detail,ensure_ascii=False))
        self.assertIn("public source facts",detail["reviewer_issues"][0])
        self.assertIn("Cedarbridge Society",detail["rejected_report"]["summary"])

    def test_writer_and_repair_receive_draft_dependencies_and_scoped_facts(self):
        payload=sample_payload()
        payload["dependency_results"]=[
            {"public_ref":"Attempt 1, subtask 3","answer":["Alice"]},
        ]
        fragment=payload["fragment"]
        fragment["operations"]=[
            {"op":"FILTER_YEAR","inputs":["r:birth","=","1912"]},
            {"op":"RELATE","inputs":["neighbor","backward"]},
        ]
        fragment["operation_results"]=[
            {"input_values":[["Alice","Bob"]],"answer":["Alice"]},
            {"input_values":[["Alice"]],"answer":["Carol"]},
        ]
        fragment["expected_answer"]=["Carol"]
        fragment["observations"][0]["claims"] += [
            {"subject":"Carol","predicate":"neighbor","object":"Alice"},
            {"subject":"Unrelated","predicate":"neighbor","object":"Other"},
        ]
        canonical={"summary":"DRAFT_CANARY","trace_summary":"Fixed steps."}
        reference=feedback_execution_reference(payload)
        self.assertEqual(reference["completed_dependency_results"][0]["reference"],
                         "Attempt 1, subtask 3")
        self.assertEqual(reference["execution_steps"][0]["supporting_facts"][0]["subject"],
                         "Alice")
        hop=reference["execution_steps"][1]
        self.assertEqual([fact["subject"] for fact in hop["supporting_facts"]],["Carol"])
        self.assertIn("subjects",hop["meaning"])
        self.assertIn("not transferred",hop["meaning"])
        self.assertEqual([fact["predicate"] for fact in reference["final_result_facts"]],
                         ["neighbor"])
        self.assertNotIn("r:birth",str(reference["final_result_facts"]))
        self.assertIn("Carol",reference["result_statement"])
        for build in (feedback_writer_prompt,feedback_repair_prompt):
            prompt=build(payload,canonical)
            data=json.loads(prompt.split("\n",1)[1])
            self.assertEqual(data["canonical_report"],canonical)
            self.assertEqual(data["execution_reference"],reference)
        review=feedback_reviewer_prompt(payload,canonical)
        self.assertEqual(json.loads(review.split("\n",1)[1])["execution_reference"],
                         reference)

    def test_corroboration_requires_same_relevant_fact_in_distinct_sources(self):
        payload=sample_payload()
        fact={"subject":"Alice","predicate":"r:birth","object":"1912"}
        payload["fragment"]["observations"].append({
            "resource_id":"source_02","title":"Reference 2","source_type":"text",
            "claims":[fact,dict(fact)],
        })
        groups=feedback_execution_reference(payload)["corroborated_facts"]
        self.assertEqual(len(groups),1)
        self.assertEqual([r["title"] for r in groups[0]["references"]],
                         ["Reference 1","Reference 2"])
        self.assertIn("Cross-checked Reference 1, Reference 2",
                      feedback_execution_reference(payload)["corroboration_statements"][0])
        payload["fragment"]["observations"][1]["claims"]=[
            {**fact,"object":"1913"},
        ]
        self.assertEqual(feedback_execution_reference(payload)["corroborated_facts"],[])

    def test_attribute_writer_sources_do_not_offer_unrelated_source_values(self):
        payload=sample_payload()
        payload["fragment"]["operations"]=payload["fragment"]["operations"][1:]
        payload["fragment"]["operation_results"]=payload["fragment"]["operation_results"][1:]
        payload["fragment"]["observations"].append({
            "resource_id":"source_02","title":"Reference 2","source_type":"text",
            "claims":[{"subject":"Alice","predicate":"population","object":"1912"}],
        })
        data=json.loads(feedback_writer_prompt(payload,{}).split("\n",1)[1])
        sources=data["source_evidence_guide"]
        self.assertEqual(len(sources[0]["claims"]),1)
        self.assertEqual(sources[1]["claims"],[])

    def test_selection_reference_explains_extremum_and_candidate_fact_owners(self):
        payload=sample_payload()
        f=payload["fragment"]
        f["operations"]=[{"op":"SELECT_BETWEEN","inputs":["population","<"]}]
        f["operation_results"]=[{"input_values":[["Alice"],["Bob"]],"answer":["Bob"]}]
        f["observations"][0]["claims"]=[
            {"subject":"Alice","predicate":"population","object":"12"},
            {"subject":"Bob","predicate":"population","object":"8"},
        ]
        step=feedback_execution_reference(payload)["execution_steps"][0]
        self.assertIn("minimum",step["meaning"])
        self.assertIn("union",step["meaning"])
        self.assertIn("including ties",step["meaning"])
        self.assertEqual({fact["subject"] for fact in step["supporting_facts"]},
                         {"Alice","Bob"})

    def test_bank_publication_rechecks_with_the_same_dependency_references(self):
        from environment.trajectory_simulation.subtask_bank import _report_natural
        payload=sample_payload()
        payload["dependency_results"]=[
            {"public_ref":"Attempt 1, subtask 3","answer":["Alice"]},
        ]
        candidate={"summary":"Attempt 1 returned Alice.",
                   "trace_summary":"Attempt 1, subtask 3 returned Alice."}
        class Executor:
            style_provider="natural"
            def naturalize_feedback(self,payload,template):
                return candidate
        report,status=_report_natural(Executor(),payload,{},payload["fragment"])
        self.assertTrue(status["natural"],status)
        self.assertEqual(report,candidate)
        payload["dependency_results"]=[]
        _,status=_report_natural(Executor(),payload,{},payload["fragment"])
        self.assertFalse(status["natural"])

    def test_completed_task_numbers_are_metadata_but_new_numbers_still_fail(self):
        payload=sample_payload()
        payload["dependency_results"]=[
            {"public_ref":"Attempt 1, subtask 3","answer":["Alice"]},
            {"public_ref":"Attempt 1, subtask 2","answer":["Alice","Bob"]},
        ]
        good={
            "summary":"The intersection of Attempt 1 results is Alice.",
            "trace_summary":"Intersected Attempt 1, subtask 3 with Attempt 1, subtask 2 to obtain Alice.",
        }
        self.assertEqual(realize_feedback_natural(payload,good),good)
        for text in ("Returned Alice with population 99.",
                     "Attempt 99 returned Alice.",
                     "Subtask 99 returned Alice."):
            with self.subTest(text=text):
                with self.assertRaisesRegex(ValueError,"unsupported_numeric_claim"):
                    realize_feedback_natural(payload,{**good,"trace_summary":text})
        with self.assertRaisesRegex(ValueError,"unsupported_numeric_claim"):
            realize_feedback_natural({**payload,"dependency_results":[]},good)
        crossed={**payload,"dependency_results":[
            {"public_ref":"Attempt 1, subtask 2","answer":["Alice"]},
            {"public_ref":"Attempt 2, subtask 3","answer":["Bob"]},
        ]}
        with self.assertRaisesRegex(ValueError,"unobserved_task_reference"):
            realize_feedback_natural(crossed,good)

    def test_prompt_explicitly_requires_string_bullets(self):
        payload=sample_payload()
        prompt=feedback_writer_prompt(payload, {})
        self.assertIn("NOT A JSON ARRAY",prompt)
        self.assertIn("trace_summary MUST be one nonempty JSON STRING",prompt)
        self.assertIn("escaped newline",prompt)
        self.assertEqual(SUBAGENT_REPORT_JSON_SCHEMA["properties"]["trace_summary"]["type"],"string")
        self.assertFalse(SUBAGENT_REPORT_JSON_SCHEMA["additionalProperties"])

    def test_list_of_nonempty_bullets_normalized_without_changing_facts(self):
        payload=sample_payload()
        candidate={
            "summary":"Alice matches the recorded birth year.",
            "trace_summary":[
                "Checked Reference 1 and found Alice's birth year 1912.",
                "- The year matches the specified condition.",
            ],
        }
        normalized,fields=normalize_feedback_shape(candidate)
        self.assertEqual(fields,("trace_summary",))
        self.assertEqual(
            normalized["trace_summary"],
            "- Checked Reference 1 and found Alice's birth year 1912.\n"
            "- The year matches the specified condition.",
        )
        self.assertIsInstance(normalized["trace_summary"],str)
        self.assertEqual(realize_feedback_natural(payload,candidate),normalized)
        self.assertIsInstance(candidate["trace_summary"],list)
        for invalid in ([],[""],["Step",None],["Step",{"bad":"type"}]):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    realize_feedback_natural(payload,{
                        **candidate,"trace_summary":invalid,
                    })

    def test_single_element_summary_list_normalizes_without_creating_answer(self):
        proposal={"summary":["Alice matches."],"trace_summary":"The record confirms Alice."}
        normalized,fields=normalize_feedback_shape(proposal)
        self.assertEqual(fields,("summary",))
        self.assertEqual(normalized["summary"],"Alice matches.")
        self.assertEqual(normalize_feedback_shape({
            **proposal,"summary":["Alice","Bob"],
        })[1],())

    def test_internal_evidence_keys_and_invented_literal_query_are_rejected(self):
        payload=sample_payload()
        base={"summary":"Alice matches the condition.",
              "trace_summary":"Reference 1 records Alice's birth year 1912."}
        self.assertEqual(realize_feedback_natural(payload,base),base)
        for addition in (
            " I checked observation key 123.",
            " The claim ID was used to identify the person.",
            " I searched for 'Alice' and found the official record.",
        ):
            with self.subTest(addition=addition):
                with self.assertRaisesRegex(
                    ValueError,
                    "natural_report_leaks_internal_evidence_handle|unverified_literal_search_query",
                ):
                    realize_feedback_natural(payload,{
                        **base,"trace_summary":base["trace_summary"]+addition,
                    })

    def test_derived_relation_and_filter_outputs_allowed_by_reviewer_instructions(self):
        reviewer=feedback_reviewer_prompt(sample_payload(),{
            "summary":"Alice matches.",
            "trace_summary":"The record confirms Alice's birth year as 1912.",
        })
        self.assertIn("authoritative",reviewer)
        self.assertIn("backward relation traversal",reviewer)
        self.assertIn("correct reference",reviewer)
        self.assertIn("Reject actions",reviewer)

    def test_executor_normalizes_first_writer_list_without_format_retry(self):
        payload=sample_payload()
        class FakeBackend:
            model="fake"
            def __init__(self):
                self.calls=0
            def complete_json(self,system,prompt):
                self.calls+=1
                if self.calls==1:
                    return {
                        "summary":"Alice meets the birth-year requirement.",
                        "trace_summary":[
                            "Reference 1 records Alice's birth year as 1912.",
                            "This verifies Alice as the result.",
                        ],
                    }
                return {"grounded":True,"issues":[]}
        backend=FakeBackend()
        executor=LLMExecutor(backend)
        audit=[]
        report=executor.naturalize_feedback(payload,{},audit=audit)
        self.assertIsInstance(report["trace_summary"],str)
        self.assertEqual(executor.request_counts["natural_feedback_write"],1)
        self.assertEqual(executor.request_counts["natural_feedback_retry"],0)
        self.assertEqual(executor.request_counts["natural_feedback_normalized"],1)
        self.assertEqual(executor.request_counts["natural_feedback_review"],1)
        self.assertEqual(backend.calls,2)
        self.assertEqual(audit[0]["normalized_fields"],["trace_summary"])

    def test_backend_sends_json_schema_only_on_explicit_request(self):
        class Response:
            def __init__(self,content):
                self.content=content
            def __enter__(self):
                return self
            def __exit__(self,*exc):
                return False
            def read(self):
                return json.dumps({
                    "choices":[{"message":{"content":json.dumps(self.content)}}]
                }).encode("utf-8")
        requests=[]
        def urlopen(request,timeout):
            requests.append(json.loads(request.data.decode("utf-8")))
            return Response({"summary":"Alice.","trace_summary":"Confirmed the result."})
        backend=OpenAICompatibleBackend(
            "http://localhost:8000/v1","qwen-local",api_key=None,
        )
        with patch(
            "environment.question_generation.backend.urllib.request.urlopen",
            side_effect=urlopen,
        ):
            backend.complete_json("system","JSON please",response_schema=SUBAGENT_REPORT_JSON_SCHEMA)
            backend.complete_json("system","JSON please")
        self.assertEqual(requests[0]["response_format"]["type"],"json_schema")
        self.assertEqual(
            requests[0]["response_format"]["json_schema"]["schema"],
            SUBAGENT_REPORT_JSON_SCHEMA,
        )
        self.assertEqual(requests[1]["response_format"],{"type":"json_object"})

    def test_schema_unavailable_falls_back_once_only(self):
        class Response:
            def __enter__(self):return self
            def __exit__(self,*a):return False
            def read(self):
                return json.dumps({
                    "choices":[{"message":{"content":
                        '{"summary":"Alice.","trace_summary":"Confirmed."}'}}]
                }).encode("utf-8")
        requests=[]
        def request(request,timeout):
            body=json.loads(request.data)
            requests.append(body)
            if len(requests)==1:
                raise urllib.error.HTTPError(
                    request.full_url,400,"unsupported",{},
                    io.BytesIO(b'{"error":"response_format json_schema not supported"}'),
                )
            return Response()
        backend=OpenAICompatibleBackend(
            "http://localhost:8000/v1","qwen-local",
        )
        with patch(
            "environment.question_generation.backend.urllib.request.urlopen",
            side_effect=request,
        ):
            result=backend.complete_json(
                "system","JSON please",response_schema=SUBAGENT_REPORT_JSON_SCHEMA,
            )
        self.assertEqual(result["summary"],"Alice.")
        self.assertEqual(
            [r["response_format"]["type"] for r in requests],
            ["json_schema","json_object"],
        )

    def test_non_schema_http_errors_not_silently_retried(self):
        calls=[]
        def server(request,timeout):
            calls.append(True)
            raise urllib.error.HTTPError(
                request.full_url,401,"unauthorized",{},
                io.BytesIO(b'{"error":"authentication failed"}'),
            )
        backend=OpenAICompatibleBackend("http://localhost:8000/v1","qwen-local")
        with patch(
            "environment.question_generation.backend.urllib.request.urlopen",
            side_effect=server,
        ):
            with self.assertRaises(BackendError):
                backend.complete_json(
                    "system","JSON please",response_schema=SUBAGENT_REPORT_JSON_SCHEMA,
                )
        self.assertEqual(len(calls),1)


if __name__=="__main__":
    unittest.main()
