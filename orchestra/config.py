"""Shared runtime and MAESTRO-specific configuration."""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass
class GAIAOrchestraConfig:
    """Benchmark/runtime config shared by MAESTRO and Orchestra-o1."""

    main_model: str
    sub_models: List[str]
    dataset_path: Path
    attachments_dir: Path
    level_filter: List[int] | None = None
    max_tasks: int | None = None
    max_attempts: int = 5
    max_concurrency: int = 1
    result_folder: Path = field(default_factory=lambda: Path("workspace/logs"))
    trajectory_folder: Path = field(
        default_factory=lambda: Path("workspace/logs/trajectories")
    )
    timestamp: str | None = None

    @classmethod
    def load(cls, config_path: Path | str) -> "GAIAOrchestraConfig":
        config_path = Path(config_path)
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        main_model = raw.get("main_model")
        if not main_model:
            raise ValueError("main_model is required")
        sub_models = raw.get("sub_models")
        if not sub_models or not isinstance(sub_models, list):
            raise ValueError("sub_models must be a non-empty list")
        dataset_path = raw.get("dataset_path")
        if not dataset_path:
            raise ValueError("dataset_path is required")
        attachments_dir = raw.get("attachments_dir")
        if not attachments_dir:
            raise ValueError("attachments_dir is required")
        return cls(
            main_model=str(main_model),
            sub_models=[str(model) for model in sub_models],
            dataset_path=cls._resolve_path(dataset_path, config_path),
            attachments_dir=cls._resolve_path(attachments_dir, config_path),
            level_filter=raw.get("level_filter"),
            max_tasks=raw.get("max_tasks"),
            max_attempts=int(raw.get("max_attempts", 5)),
            max_concurrency=int(raw.get("max_concurrency", 1)),
            result_folder=cls._resolve_path(
                raw.get("result_folder", "workspace/logs"),
                config_path,
            ),
            trajectory_folder=cls._resolve_path(
                raw.get("trajectory_folder", "workspace/logs/trajectories"),
                config_path,
            ),
        )

    @staticmethod
    def _resolve_path(path_str: str, config_path: Path) -> Path:
        path = Path(path_str)
        if path.is_absolute():
            return path
        rel_to_config = config_path.parent / path
        if rel_to_config.exists():
            return rel_to_config.resolve()
        repo_root = Path(__file__).resolve().parents[1]
        return (repo_root / path).resolve()


@dataclass
class ProspectiveConfig:
    """MAESTRO-only prospective/epistemic settings."""

    enabled: bool = True
    max_prompt_results: int = 12
    max_prompt_state_chars: int = 9000
    max_source_ids_per_result: int = 12
    max_epistemic_bindings: int = 8
    max_belief_history: int = 4
    max_evidence_attempts_per_target: int = 6
    max_future_requirements: int = 4
    max_critical_dependencies: int = 3
    max_tasks_per_action: int = 5
    max_evidence_refs: int = 6
    max_independent_refs: int = 4
    completion_consistency_check: bool = True
    execution_projection_enabled: bool = True
    strict_provider_diversity: bool = True
    blind_requery: bool = True
    route_constraints_enabled: bool = True
    provider_disjoint_routes: bool = True
    blind_verification: bool = True

    @classmethod
    def from_dict(cls, raw: Dict[str, Any] | None) -> "ProspectiveConfig":
        raw = raw or {}
        allowed = {item.name for item in fields(cls)}
        cfg = cls(**{key: value for key, value in raw.items() if key in allowed})
        if (
            "execution_projection_enabled" not in raw
            and "route_constraints_enabled" in raw
        ):
            cfg.execution_projection_enabled = bool(cfg.route_constraints_enabled)
        if (
            "strict_provider_diversity" not in raw
            and "provider_disjoint_routes" in raw
        ):
            cfg.strict_provider_diversity = bool(cfg.provider_disjoint_routes)
        if "blind_requery" not in raw and "blind_verification" in raw:
            cfg.blind_requery = bool(cfg.blind_verification)
        if "max_evidence_refs" not in raw and "max_independent_refs" in raw:
            cfg.max_evidence_refs = max(1, int(cfg.max_independent_refs))
        cfg.validate()
        return cfg

    @classmethod
    def load(cls, config_path: Path | str) -> "ProspectiveConfig":
        with Path(config_path).open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        return cls.from_dict(raw.get("prospective"))

    def validate(self) -> None:
        if int(self.max_prompt_results) < 1:
            raise ValueError("prospective.max_prompt_results must be >= 1")
        if int(self.max_prompt_state_chars) < 1000:
            raise ValueError("prospective.max_prompt_state_chars must be >= 1000")
        if int(self.max_source_ids_per_result) < 1:
            raise ValueError("prospective.max_source_ids_per_result must be >= 1")
        if not 1 <= int(self.max_epistemic_bindings) <= 20:
            raise ValueError("prospective.max_epistemic_bindings must be in [1, 20]")
        if not 1 <= int(self.max_belief_history) <= 10:
            raise ValueError("prospective.max_belief_history must be in [1, 10]")
        if not 1 <= int(self.max_evidence_attempts_per_target) <= 20:
            raise ValueError(
                "prospective.max_evidence_attempts_per_target must be in [1, 20]"
            )
        if not 1 <= int(self.max_future_requirements) <= 8:
            raise ValueError("prospective.max_future_requirements must be in [1, 8]")
        if not 0 <= int(self.max_critical_dependencies) <= 6:
            raise ValueError(
                "prospective.max_critical_dependencies must be in [0, 6]"
            )
        if not 1 <= int(self.max_tasks_per_action) <= 10:
            raise ValueError("prospective.max_tasks_per_action must be in [1, 10]")
        if not 1 <= int(self.max_evidence_refs) <= 12:
            raise ValueError("prospective.max_evidence_refs must be in [1, 12]")
        if not 1 <= int(self.max_independent_refs) <= 8:
            raise ValueError("prospective.max_independent_refs must be in [1, 8]")
