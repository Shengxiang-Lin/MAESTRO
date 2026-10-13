"""Shared runtime and MAESTRO-specific configuration."""
from __future__ import annotations

from dataclasses import dataclass, field, fields
import os
from pathlib import Path
from typing import Any, Dict, List

import yaml

from benchmark.dataset_preparation import normalize_dataset_type, prepare_dataset

from base.engine.async_llm import LLMsConfig
from base.engine.vllm_service import VLLMServiceManager


@dataclass
class GAIAOrchestraConfig:
    """Benchmark/runtime config shared by MAESTRO and Orchestra-o1."""

    main_model: str
    sub_models: List[str]
    dataset_path: Path
    attachments_dir: Path
    tool_models: Dict[str, str] = field(default_factory=dict)
    evaluation_model: str = "qwen3.7-plus"
    models: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    level_filter: List[int] | None = None
    max_tasks: int | None = None
    max_attempts: int = 5
    max_concurrency: int = 1
    result_folder: Path = field(default_factory=lambda: Path("workspace/logs"))
    trajectory_folder: Path = field(
        default_factory=lambda: Path("workspace/logs/trajectories")
    )
    timestamp: str | None = None
    dataset_type: str = "omnigaia"
    dataset_jsonl_path: Path | None = None
    dataset_source_path: Path | None = None

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
        tool_models = raw.get("tool_models") or {}
        if not isinstance(tool_models, dict):
            raise ValueError("tool_models must be a mapping from tool role to model name")
        models = raw.get("models") or {}
        if not isinstance(models, dict):
            raise ValueError("models must be a mapping from model name to configuration")
        return cls(
            dataset_type=normalize_dataset_type(raw.get("dataset_type", "omnigaia")),
            dataset_jsonl_path=(
                cls._resolve_path(raw["dataset_jsonl_path"], config_path)
                if raw.get("dataset_jsonl_path") else None
            ),
            main_model=str(main_model),
            sub_models=[str(model) for model in sub_models],
            dataset_path=cls._resolve_path(dataset_path, config_path),
            attachments_dir=cls._resolve_path(attachments_dir, config_path),
            tool_models={str(role): str(model) for role, model in tool_models.items()},
            evaluation_model=str(raw.get("evaluation_model") or "qwen3.7-plus"),
            models=cls._normalize_models(models, config_path),
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

    def prepare_dataset(self) -> Path:
        """Resolve the original source to the JSONL consumed by both runners."""
        source = self.dataset_source_path or self.dataset_path
        prepared = prepare_dataset(
            source, self.attachments_dir, self.dataset_type, self.dataset_jsonl_path,
        )
        self.dataset_source_path = source
        self.dataset_path = prepared
        return prepared

    def install_model_overrides(self) -> None:
        """Apply experiment-local model overrides on top of the shared registry.

        ``config/model_config.yaml`` is the canonical model/deployment registry.
        A benchmark may still define ``models:`` for one-off experiments, but
        those entries are overlays rather than full replacements. Nested
        mappings such as ``extra_body`` are merged recursively as well.
        """
        registry = LLMsConfig.default()
        for name, override in self.models.items():
            base = registry.configs.get(name, {})
            if not isinstance(base, dict):
                base = {}
            registry.add_config(
                name,
                self._merge_model_config(dict(base), dict(override)),
            )

    @classmethod
    def _merge_model_config(
        cls,
        base: Dict[str, Any],
        override: Dict[str, Any],
    ) -> Dict[str, Any]:
        merged = dict(base)
        for key, value in override.items():
            current = merged.get(key)
            if isinstance(current, dict) and isinstance(value, dict):
                merged[key] = cls._merge_model_config(current, value)
            else:
                merged[key] = value
        return merged

    def required_model_names(self) -> List[str]:
        names = [self.main_model, *self.sub_models, self.evaluation_model]
        names.extend(self.tool_models.values())
        return list(dict.fromkeys(str(name) for name in names if name))

    def validate_model_registry(self) -> None:
        """Validate model roles without accidentally launching local services."""
        registry = LLMsConfig.default()
        roles: Dict[str, str] = {
            "main_model": self.main_model,
            "evaluation_model": self.evaluation_model,
        }
        roles.update(
            {f"sub_models[{index}]": model for index, model in enumerate(self.sub_models)}
        )
        roles.update(
            {f"tool_models.{role}": model for role, model in self.tool_models.items()}
        )
        errors: List[str] = []
        for role, model in roles.items():
            if model not in registry.configs:
                errors.append(f"{role}={model!r}: configuration not found")
                continue
            raw = registry.configs[model]
            backend = str(raw.get("backend") or raw.get("provider") or "api").lower()
            if backend == "vllm":
                if not raw.get("model_path"):
                    errors.append(f"{role}={model!r}: vLLM model_path is required")
                continue
            try:
                registry.get(model)
            except Exception as exc:
                errors.append(f"{role}={model!r}: {exc}")
        if errors:
            raise ValueError(
                "Model configuration contains invalid registry entries:\n- "
                + "\n- ".join(errors)
            )

    def prepare_model_backends(self) -> Dict[str, Dict[str, Any]]:
        """Install overrides and prepare local Main/Sub/Evaluation backends.

        Remote API configs are left untouched. A config with ``backend: vllm``
        is started/reused locally, then rewritten in the registry as a normal
        OpenAI-compatible endpoint. Main-Agent and Sub-Agent code therefore use
        exactly the same LLM client for local and remote models.
        """
        self.install_model_overrides()
        self.validate_model_registry()
        registry = LLMsConfig.default()
        manager = VLLMServiceManager.default()
        runtime_names = list(
            dict.fromkeys([self.main_model, *self.sub_models, self.evaluation_model])
        )
        for name in runtime_names:
            raw = dict(registry.configs[name])
            backend = str(raw.get("backend") or raw.get("provider") or "api").lower()
            if backend == "vllm":
                registry.add_config(name, manager.prepare(name, raw))
        return manager.manifest(runtime_names)

    @classmethod
    def _normalize_models(
        cls,
        models: Dict[str, Any],
        config_path: Path,
    ) -> Dict[str, Dict[str, Any]]:
        normalized: Dict[str, Dict[str, Any]] = {}
        for name, value in models.items():
            if not isinstance(value, dict):
                raise ValueError(f"models.{name} must be a mapping")
            cfg = dict(value)
            backend = str(cfg.get("backend") or cfg.get("provider") or "api").lower()
            if backend == "vllm" and cfg.get("model_path"):
                raw_path = str(cfg["model_path"])
                expanded = os.path.expanduser(os.path.expandvars(raw_path))
                if not expanded.startswith("$"):
                    candidate = Path(expanded)
                    if not candidate.is_absolute():
                        rel_to_config = config_path.parent / candidate
                        if rel_to_config.exists():
                            cfg["model_path"] = str(rel_to_config.resolve())
                        else:
                            repo_root = Path(__file__).resolve().parents[1]
                            rel_to_repo = repo_root / candidate
                            if rel_to_repo.exists():
                                cfg["model_path"] = str(rel_to_repo.resolve())
                    else:
                        cfg["model_path"] = str(candidate)
            normalized[str(name)] = cfg
        return normalized

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
