"""Scenario-first source construction and teacher/execution tool isolation."""
from __future__ import annotations

import unittest

from environment.task_construction.evidence.layout import (
    _plan_acquisition_scenario, _plan_modality_layout, _MEDIA_SOURCE_TYPES,
)
from environment.task_construction.evidence.scenario_prior import SourceMotifPrior
from environment.task_construction.task_design.planner import (
    _source_scenario, _visibility_plan,
)
from environment.task_construction.task_design.quality import assess_task_design
from environment.task_construction.task_design.schema import (
    ResourceVisibility, TaskDesign,
)
from environment.task_construction.world import EvidenceSource, SyntheticWorld
from environment.trajectory_simulation.search_simulator import simulate_search
from environment.trajectory_simulation.tool_provenance import (
    reconstruct_tool_observations, permitted_tool_names,
)


def world_with_sources(types, *, scenario=None):
    world = SyntheticWorld()
    world.sources = {
        f"source_{i}": EvidenceSource(f"source_{i}", kind, (), {})
        for i, kind in enumerate(types, 1)
    }
    if scenario:
        world.metadata = {
            "evidence_allocation": {
                "layout": {
                    "final_layout_plan": {
                        "acquisition_scenario": scenario,
                        "scenario_planned_before_source_modalities": True,
                    }
                }
            }
        }
    return world


class ScenarioFirstTests(unittest.TestCase):
    def test_scenario_selected_before_modalities_and_reproducible(self):
        prior=SourceMotifPrior.fallback()
        for forced,weights in (
            ("web_search", (1,0,0)),
            ("hybrid", (0,1,0)),
            ("provided_media", (0,0,1)),
        ):
            for seed in (17, 91, 101, 239):
                self.assertEqual(_plan_acquisition_scenario(seed,weights), forced)
                modes,origins,order,meta=_plan_modality_layout(
                    5,motif_prior=prior,seed=seed,
                    synthetic_modality_probability=.35,
                    acquisition_scenario_weights=weights,
                )
                self.assertEqual(len(modes),5)
                self.assertEqual(meta["acquisition_scenario"], forced)
                self.assertTrue(meta["scenario_planned_before_source_modalities"])
                self.assertEqual(
                    (modes,origins,order,meta),
                    _plan_modality_layout(
                        5,motif_prior=prior,seed=seed,
                        synthetic_modality_probability=.35,
                        acquisition_scenario_weights=weights,
                    ),
                )
                if forced=="hybrid":
                    self.assertTrue(any(m in _MEDIA_SOURCE_TYPES for m in modes))
                    self.assertTrue(any(m not in _MEDIA_SOURCE_TYPES for m in modes))
                if forced=="provided_media":
                    self.assertTrue(all(m in _MEDIA_SOURCE_TYPES for m in modes))

    def test_bad_scenario_weights_fail_closed(self):
        for weights in ((0,0,0), (-1,1,1), (1,2), (1,2,3,4)):
            with self.subTest(weights=weights):
                with self.assertRaisesRegex(ValueError,"scenario_weights"):
                    _plan_acquisition_scenario(17,weights)

    def test_web_search_can_contain_video_and_image_but_no_initial_attachment(self):
        world=world_with_sources(
            ("text","video_report","image_report"), scenario="web_search",
        )
        self.assertEqual(_source_scenario(world),"web_search")
        visibility=_visibility_plan(world,())
        self.assertEqual({item.visibility for item in visibility},{"discoverable"})
        self.assertEqual({item.role for item in visibility},{"web_result"})
        self.assertFalse(any(item.visibility=="attached" for item in visibility))
        from environment.trajectory_simulation.tools import tool_profile
        profile=tool_profile("omnigaia")
        self.assertEqual(profile.for_resource({
            "source_type":"video_report","visibility":"discoverable",
        }),("GoogleSearchAction","VideoAnalysisAction"))
        self.assertEqual(profile.for_resource({
            "source_type":"image_report","visibility":"discoverable",
        }),("GoogleSearchAction","ImageAnalysisAction"))

    def test_hybrid_attaches_real_media_instead_of_unknown_candidate_list(self):
        world=world_with_sources(
            ("text","video_report","database"), scenario="hybrid",
        )
        visibility=_visibility_plan(world,())
        self.assertEqual(sum(v.visibility=="attached" for v in visibility),1)
        self.assertEqual(sum(v.visibility=="discoverable" for v in visibility),2)
        attached=next(v for v in visibility if v.visibility=="attached")
        self.assertEqual(attached.source_type,"video_report")

    def test_quality_gate_allows_search_query_entry_only_with_retrieval_policy(self):
        resources=(
            ResourceVisibility("s1","video_report","discoverable","web_result","search"),
            ResourceVisibility("s2","text","discoverable","web_result","search"),
        )
        def design(metadata):
            return TaskDesign(
                task_id="test",motif="multi_source",final_operation="QUERY_NAME",
                answer_type="entity",entry_strategy="web_search",
                latent_entity_ids=(),latent_find_node_ids=(),public_clues=(),
                required_reasoning_node_ids=(1,2,3),
                resource_visibility=resources,minimum_required_source_count=2,
                metadata=metadata,
            )
        valid=assess_task_design(design({
            "retrieval_policy":"task_local_ranked_public_index_v1",
        }))
        self.assertTrue(valid.accepted,valid.reasons)
        self.assertTrue(valid.metrics["search_query_entry"])
        legacy=assess_task_design(design({}))
        self.assertFalse(legacy.accepted)
        self.assertIn("no_public_entry_resource",legacy.reasons)
        invalid=design({"retrieval_policy":"task_local_ranked_public_index_v1"})
        from dataclasses import replace
        malformed=replace(invalid,resource_visibility=(
            ResourceVisibility("s1","video_report","attached","entry_anchor",""),
            resources[1],
        ))
        self.assertIn("web_search_has_presupplied_source",
                      assess_task_design(malformed).reasons)

    def test_legacy_world_without_scenario_still_uses_old_inference(self):
        world=world_with_sources(("text","video_report"))
        self.assertEqual(_source_scenario(world),"hybrid")
        world=world_with_sources(("text","database"))
        self.assertEqual(_source_scenario(world),"web_search")


class SearchPermissionTests(unittest.TestCase):
    def test_unassigned_search_cannot_enter_tool_provenance(self):
        first={
            "resource_id":"source_1",
            "source_type":"text",
            "title":"Reference 1",
            "visibility":"provided",
            "content":"Record contains a population of 412.",
            "claims":[],
            "entity_types":{"Dunhaven":"Administrative Area"},
        }
        second={
            **first,"resource_id":"source_2",
            "visibility":"discoverable",
            "content":"A record with male population of 412.",
        }
        teacher_audit=simulate_search(
            (first,second),
            public_question="Find Administrative Areas with male population 412.",
            public_task="Identify the areas by their recorded population.",
        )
        self.assertTrue(teacher_audit["calls"])
        no_search={
            "scene":"omnigaia",
            "suggested_tools":["ExtractUrlContentAction"],
            "observations":[first],
            "known_observations":[],
            "acquisition_waves":[["source_1"]],
        }
        report=reconstruct_tool_observations(no_search,teacher_audit)
        self.assertEqual(report["search_calls"],[])
        self.assertEqual(permitted_tool_names(report),{"ExtractUrlContentAction"})
        self.assertFalse(any(x["tool"]=="GoogleSearchAction" for x in report["steps"]))
        # A truly fresh discoverable source cannot be read by this
        # Sub-Agent unless the Main delegated search permissions.
        no_search["observations"]=[second]
        no_search["acquisition_waves"]=[["source_2"]]
        with self.assertRaisesRegex(ValueError,"search_tool_not_assigned"):
            reconstruct_tool_observations(no_search,teacher_audit)
        no_search["suggested_tools"]=[
            "GoogleSearchAction","ExtractUrlContentAction",
        ]
        permitted=reconstruct_tool_observations(no_search,teacher_audit)
        self.assertTrue(permitted["search_calls"])
        self.assertIn("GoogleSearchAction",permitted_tool_names(permitted))


if __name__=="__main__":
    unittest.main()
