"""Current Stage-1 World and public projection, for task_plan only."""
from __future__ import annotations
from environment.generate_question.execution.program_ir import answer_values, evaluate_program
from environment.generate_question.planning.spec import spec_issues

WORLD_VERSION = "maestro.source_observation_world.v4"

def _rows(sid: str, modality: str, title: str, description: str,
          data: list[dict], *, original_handle: str | None = None) -> dict:
    if not data:
        raise ValueError("empty synthetic observation source")
    keys = list(data[0])
    if any(set(x) != set(keys) for x in data):
        raise ValueError("inconsistent_source_schema")
    return {
        "source_id": sid,
        "modality": modality,
        "title": title,
        "description": description,
        "representation": "synthetic_structured_observations_no_media_decoding",
        "original_mmqa_handle_used": False,
        "schema": keys,
        "observations": [{
            "observation_id": f"{sid}.OBS{i:03d}",
            "fact_refs": [f"{sid}:F{i:03d}"],
            "values": dict(entry),
        } for i, entry in enumerate(data, 1)],
    }

def initial_public_view(spec: dict, world: dict, question: str) -> dict:
    """Initial Main-Agent view; no original observations, answers, or program."""
    return {
        "task_id": spec["task_id"],
        "question": question,
        **({"task_input": spec["task_input"]}
           if "task_input" in spec else {}),
        "available_source_descriptions": [{
            "source_id": sid,
            "title": item["title"],
            "modality_hint": item["modality"],
            **({"resource_url": item["resource_url"]}
               if "resource_url" in item else {}),
            **({} if spec.get("reasoning_pattern") in {
                "macro_first_v2", "task_plan"
            } else {"representation": item["representation"]}),
            **({"access_mode": item["access_policy"]["mode"]}
               if "access_policy" in item else {}),
        } for sid, item in sorted(world["sources"].items())],
        "max_concurrent_subagents": 10,
        "status": ("ready_for_investigation"
                   if spec.get("reasoning_pattern") in {
                       "macro_first_v2", "task_plan"
                   } else "fictional_structured_multisource_world"),
    }


def build_scene(spec: dict, question: str) -> dict:
    """Full teacher scene; private gold execution and public projection."""
    program = compile_program(spec)
    world = generate_world(spec)
    execution = evaluate_program(program, sources=world["sources"])
    answer = answer_values(execution)
    return {
        "version": "maestro.generate_question.scene.v1",
        "task_id": spec["task_id"],
        "problem_spec": spec,
        "world": world,
        "gold_program": program,
        "gold_answer": answer,
        "gold_trace": execution["trace"],
        "question": question,
        "question_origin": "deterministic_frozen_problem_spec",
        "public": initial_public_view(spec, world, question),
        "execution_assumptions": {
            "all_subagents_reliable": True,
            "source_information_scoped": True,
            "real_media_perception_performed": False,
            "goal_program_hidden_from_main_agent": True,
        },
    }

def compile_program(spec: dict) -> dict:
    if spec_issues(spec):
        raise ValueError("invalid_spec:" + ",".join(spec_issues(spec)))
    from environment.generate_question.execution.task_plan_runtime import compile_macro_solvers
    return compile_macro_solvers(spec)


def generate_world(spec: dict) -> dict:
    if spec_issues(spec):
        raise ValueError("invalid_spec:" + ",".join(spec_issues(spec)))
    from environment.generate_question.execution.task_plan_runtime import generate_world as materialize
    return materialize(spec)
