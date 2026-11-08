"""Backward, typed Work-Product planner for new multi-domain Macro TaskGraphs.

Construction starts from the *target* WorkProduct, recursively chooses one
of multiple legal producer recipes, expands their typed inputs, and memoizes
intermediate products. No MMQA question/answer or MMQA operator tree drives
the Macro topology. MMQA gives a scene/source inventory only.

This is a clean *simulation* environment. A typed recipe must have plausible
information-flow semantics, a publicly addressable handoff, and transitive
source support; it is not one KoPL micro-step.
"""
from __future__ import annotations

import random
from collections import Counter

from envir.schema import digest
from .goal_grammar import select_goal, source_phrase
from .graph import (
    MacroPlanError, build_atomic_draft, coarsen, compile_contracts,
    compile_waves, complexity,
)
from .priors import choose_kqa
from .recipes import Recipe, RECIPES_BY_OUTPUT
from .scene_sources import (
    IneligibleScene, MODALITY_CAPABILITIES, partition_sources, requirement,
)

FINAL_PREFIX = "Deliver the following source-grounded work product: "
MIN_MACRO_TASKS = 3
MAX_MACRO_TASKS = 9
MAX_PLANNING_ATTEMPTS = 24


class UnsuitableSeed(ValueError):
    """A scene has insufficient usable evidence or no feasible WorkProduct DAG."""


def _topology_stats(tasks: list[dict], edges: list[dict],
                    waves: list[list[str]]) -> dict:
    outgoing = Counter(edge["from"] for edge in edges)
    return {
        "roots": sum(not task["depends_on"] for task in tasks),
        "joins": sum(len(task["depends_on"]) > 1 for task in tasks),
        "fanouts": sum(outgoing[task["task_id"]] > 1 for task in tasks),
        "width": max((len(w) for w in waves), default=0),
    }


def _topology_family(tasks: list[dict], metrics: dict,
                     depth: int) -> str:
    types = {task["work_type"] for task in tasks}
    if "partial_source_cross_check" in types:
        return "staged_merge"
    if "reference_index_construction" in types:
        return "indexed_reconciliation"
    if "evidence_consistency_review" in types and metrics["fanouts"]:
        return "independent_cross_check"
    if metrics["joins"] and metrics["fanouts"]:
        return "fork_join"
    if metrics["roots"] > 1 and metrics["joins"]:
        return "parallel_join"
    return "sequential_chain" if depth >= 3 else "direct"


def _choose_recipe(product: str, rng: random.Random,
                   root_count: int) -> Recipe:
    alternatives = RECIPES_BY_OUTPUT.get(product)
    if not alternatives:
        raise MacroPlanError(f"no recipe produces {product}")
    eligible = [
        recipe for recipe in alternatives
        if not (recipe.id == "evidence_staged" and root_count < 3)
    ]
    return rng.choice(eligible)


class _BackwardPlanner:
    """Expands typed target outputs to source work packages; memoizes handoffs."""

    def __init__(self, *, scene: dict, goal_id: str, goal: dict,
                 source_groups: list[list[dict]], rng: random.Random):
        self.scene = scene
        self.goal_id = goal_id
        self.goal = goal
        self.groups = source_groups
        self.rng = rng
        self.tasks: list[dict] = []
        self.edges: list[dict] = []
        self.recipe_routes: dict[str, str] = {}
        self.memo: dict[str, tuple[str, str, str]] = {}

    def _emit(self, *, recipe_id: str, work_type: str, artifact: str,
              goal: str, deliverable: str,
              inputs: list[tuple[str, str, str]] | None = None,
              requirements: list[dict] | None = None,
              operation: str | None = None,
              capability_hints: list[str] | None = None) -> tuple[str, str, str]:
        inputs = inputs or []
        tid = f"S{len(self.tasks) + 1}"
        schema = {}
        for i, (upstream, output_slot, value_type) in enumerate(inputs, 1):
            slot = f"evidence_{i}"
            schema[slot] = value_type
            self.edges.append({
                "from": upstream, "to": tid,
                "input_slot": slot, "output_slot": output_slot,
                "type": value_type, "kind": "required_data",
                "activation": "producer_completed",
                "handoff_reason": (
                    f"{work_type} requires typed {value_type} from {upstream}"
                ),
            })
        slot = "source_findings" if not inputs else "result"
        self.tasks.append({
            "task_id": tid,
            "work_type": work_type,
            "recipe_id": recipe_id,
            "goal": goal,
            "deliverable": deliverable,
            "outputs": {slot: artifact},
            "input_schema": schema,
            "capability_hints": capability_hints or ["Research", "EvidenceAnalysis"],
            "source_requirements": requirements or [],
            "internal_actions": [
                "inspect required input contracts and scope",
                "extract or bind source-scoped candidate observations",
                f"execute typed {operation or recipe_id} reasoning",
                "deliver the typed work product with supporting references",
            ],
            "operation_spec": {
                "op": operation or recipe_id,
                "question_operation": self.goal_id,
                "work_goal": self.goal["goal"],
                "topic": self.scene["topic"],
            },
            "grounding_status": "pending_synthetic_evidence",
        })
        self.recipe_routes[tid] = recipe_id
        return tid, slot, artifact

    def _source_root(self, index: int) -> tuple[str, str, str]:
        key = f"source:{index}"
        if key in self.memo:
            return self.memo[key]
        group = self.groups[index]
        modality = group[0]["modality"]
        requirements = [
            requirement(item, topic=self.scene["topic"]) for item in group
        ]
        output = self._emit(
            recipe_id=f"source_evidence_{modality}",
            work_type=f"source_investigation_{modality}",
            artifact="SourceEvidence",
            goal=(
                f"Investigate {len(group)} {modality} reference(s) about "
                f"{self.scene['topic']} and extract supported candidate records."
            ),
            deliverable=(
                f"SourceEvidence with each original {modality} document ID "
                "and candidate observation fields."
            ),
            requirements=requirements,
            operation=f"source_extract_{modality}",
            capability_hints=[
                MODALITY_CAPABILITIES[modality], "Research"
            ],
        )
        self.memo[key] = output
        return output

    def _inputs(self, recipe: Recipe) -> list[tuple[str, str, str]]:
        dependencies = []
        for product in recipe.inputs:
            if product == "SourceRoots":
                dependencies.extend(self._source_root(i)
                                    for i in range(len(self.groups)))
            elif product == "FirstTwoRoots":
                dependencies.extend([self._source_root(0), self._source_root(1)])
            elif product == "RemainingRoots":
                dependencies.extend(self._source_root(i)
                                    for i in range(2, len(self.groups)))
            else:
                dependencies.append(self.expand(product))
        if not dependencies:
            raise MacroPlanError(f"recipe {recipe.id} has no evidence inputs")
        return dependencies

    def expand(self, product: str) -> tuple[str, str, str]:
        if product in self.memo:
            return self.memo[product]
        if len(self.tasks) > MAX_MACRO_TASKS:
            raise MacroPlanError("planning budget exceeded")
        recipe = _choose_recipe(product, self.rng, len(self.groups))
        inputs = self._inputs(recipe)
        artifact = self.goal["artifact"] if product == "GoalReport" else product
        work_type = (
            self.goal["assessment"] if recipe.work_type == "GOAL_ASSESSMENT"
            else recipe.work_type
        )
        operation = self.goal_id if recipe.operation == "GOAL_OP" else recipe.operation
        goal_text = (
            self.goal["goal"] if product == "GoalAssessment"
            else (FINAL_PREFIX + self.goal["goal"] +
                  " Scene: " + self.scene["topic"] + ".")
            if product == "GoalReport"
            else recipe.description
        )
        if product in {"GoalReport", "GoalAssessment"}:
            goal_text += (
                "" if product == "GoalReport" else
                " Scene: " + self.scene["topic"] + "."
            )
        deliverable = (
            self.goal["deliverable"] if product == "GoalAssessment"
            else recipe.deliverable
        )
        output = self._emit(
            recipe_id=recipe.id,
            work_type=work_type,
            artifact=artifact,
            goal=goal_text,
            deliverable=deliverable,
            inputs=inputs,
            operation=operation,
        )
        if product == "GoalReport":
            # Only the terminal typed WorkProduct determines the NEW Question.
            modalities = sorted({
                item["modality"] for group in self.groups for item in group
            })
            self.tasks[-1]["question_blueprint"] = {
                "pattern_id": self.goal_id,
                "topic": self.scene["topic"],
                "modalities": modalities,
                "source_phrase": source_phrase(modalities),
                "goal_artifact": artifact,
                "origin": "taskgraph_terminal_workproduct",
                "recipe_route": recipe.id,
            }
        self.memo[product] = output
        return output


def compose_seed_taskfirst(*, seed_record: dict, ordinal: int,
                           run_seed: int, kqa: list[dict]) -> dict:
    """Select a scene/goal pair, then recursively *backchain* from the goal.

    Explore independent recipe alternatives until a bounded DAG meets exact
    source closure and typed handoff requirements. Do not add dummy nodes to
    hit an artificial depth or task-count distribution.
    """
    topic = str(seed_record["topic"]).strip()
    if not topic:
        raise UnsuitableSeed("scene lacks a usable topic")
    original_source_count = len({
        (item["modality"], item.get("doc_id"))
        for item in seed_record.get("supporting_sources", [])
        if item.get("modality") in {"table", "image", "text"}
    })
    if original_source_count < 2:
        raise UnsuitableSeed("at least two distinct supporting sources required")
    goal_id, goal = select_goal(
        ordinal=ordinal, run_seed=run_seed,
        source_count=original_source_count,
        scene_key=seed_record["record_id"],
    )
    failures = Counter()
    chosen = None
    for attempt in range(MAX_PLANNING_ATTEMPTS):
        rng = random.Random(int(digest(
            run_seed, ordinal, seed_record["record_id"], goal_id,
            "recipe_trial", attempt
        )[:14], 16))
        try:
            groups, coverage = partition_sources(seed_record, rng)
            planner = _BackwardPlanner(
                scene=seed_record, goal_id=goal_id, goal=goal,
                source_groups=groups, rng=rng,
            )
            terminal = planner.expand("GoalReport")
            if not MIN_MACRO_TASKS <= len(planner.tasks) <= MAX_MACRO_TASKS:
                raise MacroPlanError("macro task count outside 3..9")
            draft = build_atomic_draft(planner.tasks, planner.edges)
            tasks, coarsening = coarsen(draft)
            compile_contracts(tasks, planner.edges)
            waves = compile_waves(tasks)
            complexity_stats = complexity(tasks, waves, coarsening)
            if complexity_stats["macro_dependency_depth"] < 2:
                raise MacroPlanError("insufficient genuine work dependencies")
            # Every source leaf MUST be a reachable ancestor of the final.
            by_id = {task["task_id"]: task for task in tasks}
            ancestors = set()
            def visit(tid: str) -> None:
                if tid in ancestors:
                    return
                ancestors.add(tid)
                for previous in by_id[tid]["depends_on"]:
                    visit(previous)
            visit(terminal[0])
            if len(ancestors) != len(tasks):
                raise MacroPlanError("orphaned source or recipe work package")
            chosen = (
                planner, tasks, terminal, waves, complexity_stats,
                coverage, coarsening, attempt,
            )
            break
        except (MacroPlanError, IneligibleScene) as exc:
            failures[str(exc)] += 1
    if chosen is None:
        raise UnsuitableSeed(
            f"no valid recursive 3-9-task WorkProduct DAG after "
            f"{MAX_PLANNING_ATTEMPTS} trials: {dict(failures)}"
        )
    (planner, tasks, terminal, waves, complexity_stats,
     coverage, coarsening, attempt) = chosen
    metrics = _topology_stats(tasks, planner.edges, waves)
    kqa_rng = random.Random(int(digest(run_seed, ordinal, "kopl")[:14], 16))
    kqa_prior, kqa_policy = choose_kqa(
        kqa, "mixed" if metrics["joins"] else "chain",
        kqa_rng, ordinal,
    )
    task_id = "macro-taskfirst-" + digest(
        run_seed, ordinal, seed_record["record_id"], goal_id
    )
    plan = {
        "schema": "maestro.taskfirst.v1",
        "task_id": task_id,
        "scenario": {
            "domain": seed_record["domain_hint"],
            "setting": "MultiModalQA reference collection",
            "issue": "source-grounded problem-solving scenario",
            "user_role": "researcher",
            "source_kind": "MultiModalQA",
            "topic": topic,
            "source_record_id": seed_record["record_id"],
        },
        "goal": {
            "intent": goal_id,
            "desired_artifact_type": goal["artifact"],
            "pattern_origin": "curated_goal_bank_independently_sampled",
            "scene_record_id": seed_record["record_id"],
        },
        "macro_graph": {
            "motif": _topology_family(
                tasks, metrics, complexity_stats["macro_dependency_depth"]),
            "tasks": tasks,
            "edges": planner.edges,
            "terminal_handoffs": [{
                "from_task": terminal[0],
                "output_slot": terminal[1],
                "type": terminal[2],
            }],
            "reference_waves": waves,
            "provenance_recipe_routes": planner.recipe_routes,
            "complexity": {**complexity_stats, **metrics},
            "coarsening": coarsening,
            "source_coverage": coverage,
        },
        "source_priors": {
            "KQA": {
                "policy": kqa_policy, "signature": kqa_prior["signature"],
                "program": kqa_prior["program"],
                "usage": "unexecuted_internal_reasoning_prior",
            },
            "MultiModalQA": {
                "policy": "scene_and_source_metadata_only",
                "record_id": seed_record["record_id"],
                "official_question_type": seed_record["question_type"],
                "composition_family": seed_record["logical_pattern"],
                "composition_tree": seed_record["composition"],
                "source_modalities": seed_record["source_modalities"],
                "supporting_sources": seed_record["supporting_sources"],
                "original_question": seed_record.get("question", ""),
                "usage": "teacher_side_scene_source_provenance_only",
            },
            "YAGO": {
                "matches": seed_record.get("yago_matches", []),
                "usage": "optional_exact_label_context_not_claimed_source_truth",
            },
        },
        "task_plan_provenance": {
            "parent": "typed_workproduct_backward_recipe_planner",
            "ordinal": ordinal, "seed": run_seed,
            "question_generated_after_macro_graph": True,
            "question_inputs": "terminal_task_question_blueprint_not_original_mmqa_question",
            "goal_bank_pattern": goal_id,
            "selection_policy": "scene_goal_independent_then_recursive_recipe_selection",
            "recipe_attempt": attempt,
            "v3_final_gold_answer_not_used": True,
        },
        "scheduler": {
            "max_concurrent_subagents": 10,
            "runtime_variants_generated": False,
        },
    }
    from .question import task_level_brief, author_question
    plan["question_brief"] = task_level_brief(plan)
    plan["question"] = author_question(
        plan["question_brief"], ordinal=ordinal, seed=run_seed)
    return plan
