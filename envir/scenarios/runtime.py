"""Task-local simulated tool API: Main Agent sees only results of tool calls.

Only the teacher should access complete scene JSON or its 'world' field.
This small environment is not a cloud browser or a genuine media interpreter.
"""
from __future__ import annotations

import copy


class SimulatedRuntime:
    def __init__(self, scene: dict):
        self._scene = scene
        self._sources = {s["source_id"]: s for s in scene["sources"]}
        self.history = []
        self.total_cost = 0

    def public_task(self) -> dict:
        return {
            "task_id": self._scene["scene_id"],
            "question": self._scene["question"],
            "provided_source_ids": [f"{self._scene['scene_id']}:media",
                                    f"{self._scene['scene_id']}:schedule"]
                if self._scene["graph_pattern"] != "chain"
                else [f"{self._scene['scene_id']}:media"],
            "available_tools": [
                "InspectSimulatedMedia", "ReadSimulatedDocument",
                "SimulatedSearch", "OpenSimulatedPage",
            ],
            "simulation_disclosure": "The media and records are simulated observations.",
        }

    def call(self, action: str, *, source_id: str | None = None,
             query: str = "") -> dict:
        cost = {"InspectSimulatedMedia": 2, "ReadSimulatedDocument": 1,
                "SimulatedSearch": 1, "OpenSimulatedPage": 1}.get(action)
        if cost is None:
            raise ValueError("unknown action: " + action)
        if action == "SimulatedSearch":
            if not query or not query.strip():
                raise ValueError("query is required")
            words = query.lower().split()
            results = []
            for src in self._sources.values():
                if not src["searchable"]:
                    continue
                searchable = (src["title"] + " " + src["description"] + " "
                              + " ".join(str(o) for o in src["observations"])).lower()
                score = sum(word in searchable for word in words)
                if score:
                    results.append((-score, src["source_id"], src["title"]))
            results.sort()
            result = {"results": [
                {"source_id": srcid, "title": title}
                for _, srcid, title in results[:10]
            ]}
        else:
            src = self._sources.get(source_id)
            if src is None:
                raise KeyError("unknown source")
            expected_modality = {
                "InspectSimulatedMedia": {"image", "video", "audio"},
                "ReadSimulatedDocument": {"text", "table"},
                "OpenSimulatedPage": {"search"},
            }[action]
            if src["modality"] not in expected_modality:
                raise ValueError("wrong tool for source modality")
            result = {
                "source_id": src["source_id"], "modality": src["modality"],
                "title": src["title"], "description": src["description"],
                "observations": copy.deepcopy(src["observations"]),
                "simulation_disclosure": "No real media file was processed.",
            }
        self.total_cost += cost
        self.history.append({"action": action, "source_id": source_id,
                             "query": query, "cost_units": cost})
        return result

    def subagent_feedback(self, *, variant: str = "clean") -> dict:
        """Deterministic simulator responses, including uncertainty/attacks.

        Variant outputs are NOT gold-truth labels for a Main Agent.
        """
        item = self._scene.get("feedback_variants", {}).get(variant)
        if item is None:
            raise ValueError("unknown feedback variant")
        return copy.deepcopy({
            "subtask_id": item["affected_subtask"],
            "variant": variant,
            "status": item["status"],
            "answer": item.get("reported_reference"),
            "candidates": item.get("reported_candidates", []),
            "reason": item.get("reason"),
            "is_simulated_feedback": True,
        })
