"""Web-first source discovery and native-O1 tool alignment, without Gold leakage."""
from __future__ import annotations

import unittest

from environment.task_construction.world import SyntheticWorld, EvidenceSource
from environment.task_construction.task_design.planner import (
    _visibility_plan, _source_scenario,
)
from environment.task_construction.capabilities.discovery import bind_resource_discovery
from environment.task_construction.capabilities.schema import AcquisitionResource
from environment.trajectory_simulation.tools import tool_profile
from environment.trajectory_simulation.search_simulator import (
    SimulatedSearchIndex, simulate_search, verify_retrieved_witnesses,
)
from environment.trajectory_simulation.tool_provenance import (
    reconstruct_tool_observations,
)
from environment.question_generation.validators import preflight


def source(sid, text, visibility="discoverable", kind="text", entity_type="Administrative Area"):
    return {
        "resource_id":sid, "title":f"Reference {sid}",
        "source_type":kind,"visibility":visibility,
        "content":text,
        "claims":[],
        "entity_types":{sid:entity_type},
        "anchor_dependency_resource_ids":[],
    }


class WebSearchSimulationTests(unittest.TestCase):
    def test_web_only_sources_are_discoverable_not_fake_provided_files(self):
        world=SyntheticWorld()
        world.sources={
            "s1":EvidenceSource("s1","text",(),{}),
            "s2":EvidenceSource("s2","database",(),{}),
        }
        self.assertEqual(_source_scenario(world),"web_search")
        vis=_visibility_plan(world,())
        self.assertTrue(all(v.visibility=="discoverable" for v in vis))
        self.assertTrue(all(v.role=="web_result" for v in vis))
        world.sources["s3"]=EvidenceSource("s3","video_report",(),{})
        self.assertEqual(_source_scenario(world),"hybrid")
        hybrid=_visibility_plan(world,())
        self.assertEqual(sum(v.visibility=="attached" for v in hybrid),1)

    def test_search_bootstrap_does_not_invent_a_provided_reference(self):
        world=SyntheticWorld()
        world.sources["s1"]=EvidenceSource("s1","text",(),{})
        world.sources["s2"]=EvidenceSource("s2","text",(),{})
        r=AcquisitionResource(
            "s1","text","discoverable","web_result",
            ("document_reading",),("search","read_text"),{},
        )
        bound,audit=bind_resource_discovery(
            world,(r,),web_query_bootstrap=True,
        )
        self.assertEqual(bound[0].visibility,"discoverable")
        self.assertEqual(bound[0].metadata["search_entry"],"public_question")
        self.assertEqual(audit["progressive_discovery_depth"],2)
        self.assertEqual(audit["frontiers"][0]["resource_ids"],[])
        self.assertFalse(audit["retrieval_verified_here"])
        with self.assertRaisesRegex(ValueError,"requires_all_sources_discoverable"):
            bind_resource_discovery(
                world,(AcquisitionResource(
                    "s2","text","provided","entry_anchor",
                    ("document_reading",),("read_text",),{},
                ),),web_query_bootstrap=True,
            )

    def test_ranked_search_uses_only_query_and_complete_public_corpus(self):
        catalog=[
            source("s1","Dunhaven — male population: 502. Elevation 200."),
            source("s2","Maplevale — male population: 712. Elevation 240."),
            source("s3","A playwright received an award for music.",entity_type="Person"),
        ]
        index=SimulatedSearchIndex(catalog)
        hits=index.search("Find administrative areas with male population")
        self.assertEqual({h["resource_id"] for h in hits},{"s1","s2"})
        self.assertTrue(all("claims" not in hit for hit in hits))
        self.assertTrue(all("entity_types" not in hit for hit in hits))
        self.assertEqual(index.search("galaxy telescope spectroscopy"),[])
        self.assertEqual(index.search("male population",per_page=1,page=0)[0]["resource_id"],"s1")
        self.assertEqual(index.search("male population",per_page=1,page=1)[0]["resource_id"],"s2")
        with self.assertRaisesRegex(ValueError,"duplicate_search_source"):
            SimulatedSearchIndex([catalog[0],catalog[0]])

    def test_search_returns_distractors_without_gold_answers(self):
        catalog=[
            source("s1","Dunhaven — female population: 111."),
            source("s2","Maplevale — female population: 222."),
            source("s3","Winterstead — male population: 300."),
        ]
        audit=simulate_search(
            catalog,
            public_question="Which administrative area has the largest female population?",
            public_task="Search for administrative areas with female population values.",
        )
        self.assertEqual(audit["policy"],"task_local_ranked_public_index_v1")
        self.assertFalse(audit["ranking_uses_gold_witness"])
        self.assertEqual(set(audit["discovered_source_ids"]),{"s1","s2","s3"})
        self.assertTrue(audit["calls"])
        self.assertIn("query",audit["calls"][0])
        self.assertNotIn("oracle_answer",repr(audit))

    def test_witness_must_be_found_by_search_and_cannot_be_injected(self):
        catalog=[
            source("s1","Dunhaven — male population: 500."),
            source("s2","A musical performance won an award.",
                   entity_type="Creative Work"),
        ]
        audit=simulate_search(
            catalog,
            public_question="What areas have male population 500?",
            public_task="Search for administrative areas by male population.",
        )
        verify_retrieved_witnesses(audit,[catalog[0]])
        with self.assertRaisesRegex(
            ValueError,"public_search_could_not_discover_required_source:s2",
        ):
            verify_retrieved_witnesses(audit,[catalog[1]])
        fragment={
            "scene":"omnigaia",
            "suggested_tools":["GoogleSearchAction","ExtractUrlContentAction"],
            "observations":[{**catalog[0],"claims":[]}],
            "known_observations":[],
            "acquisition_waves":[["s1"]],
        }
        plan=reconstruct_tool_observations(fragment,audit)
        self.assertFalse(plan["ranking_uses_gold_witness"])
        self.assertEqual(plan["origin"],"independent_public_search_index_and_frozen_inspection")
        self.assertTrue(any(c["tool"]=="GoogleSearchAction" for c in plan["search_calls"]))
        fragment["observations"]=[{**catalog[1],"claims":[]}]
        fragment["acquisition_waves"]=[["s2"]]
        with self.assertRaisesRegex(ValueError,"public_search_could_not_discover_required_source"):
            reconstruct_tool_observations(fragment,audit)

    def test_o1_six_native_tools_remain_global_and_assignment_is_scoped(self):
        profile=tool_profile("omnigaia")
        self.assertEqual(set(profile.available),{
            "GoogleSearchAction","ExtractUrlContentAction",
            "ImageAnalysisAction","VideoAnalysisAction",
            "ParseAudioAction","ExecuteCodeAction",
        })
        self.assertEqual(
            profile.for_resource(source("s1","some content")),
            ("GoogleSearchAction","ExtractUrlContentAction"),
        )
        self.assertEqual(
            profile.for_resource(source("s2","clip",kind="video_report",visibility="attached")),
            ("VideoAnalysisAction",),
        )

    def test_stage2_web_question_rejects_fake_attached_file(self):
        payload={
            "draft_question":"Which Administrative Areas have a population of 500?",
            "task_design":{"entry_strategy":"web_search","final_operation":"QUERY_NAME"},
            "terminal_contract":{},
            "resources":[{"source_type":"text","visibility":"discoverable"}],
        }
        bad="Which Administrative Areas in the supplied document have a population of 500?"
        self.assertIn("web_search_not_presupplied_file",preflight(payload,bad).issues)
        good="Which Administrative Areas can be identified with a population of 500?"
        self.assertNotIn("web_search_not_presupplied_file",preflight(payload,good).issues)


if __name__=="__main__":
    unittest.main()
