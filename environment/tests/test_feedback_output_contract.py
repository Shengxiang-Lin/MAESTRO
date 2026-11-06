"""Strict Sub-Agent report JSON, bounded list repair and source-grounding gates."""
from __future__ import annotations

import io
import json
import unittest
import urllib.error
from unittest.mock import patch

from environment.question_generation.backend import OpenAICompatibleBackend, BackendError
from environment.trajectory_simulation.backend import LLMExecutor
from environment.trajectory_simulation.natural_language import (
    SUBAGENT_REPORT_JSON_SCHEMA,
    feedback_writer_prompt,
    feedback_reviewer_prompt,
    feedback_repair_prompt,
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
