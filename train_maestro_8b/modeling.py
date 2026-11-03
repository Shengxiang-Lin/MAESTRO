"""Qwen causal-LM wrapper with MAESTRO macro-policy and critic heads."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from orchestra.modes import ORCHESTRATION_MODES
from train_maestro_8b.schema import ORCH_TOKEN


class MaestroOrchestratorModel(nn.Module):
    """Shared Qwen backbone + original LM head + two lightweight readout heads.

    The LM head is the original causal-LM head. ``macro_head`` predicts the
    seven semantic orchestration modes from the hidden state at ``<|orch|>``.
    ``critic_head`` predicts one action value per macro mode from the same
    representation.

    The semantic heads do not change the public Orchestra action space.
    """

    def __init__(
        self,
        base_model: nn.Module,
        *,
        orch_token_id: int,
        lambda_macro: float = 0.25,
        lambda_critic: float = 1.0,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.orch_token_id = int(orch_token_id)
        self.lambda_macro = float(lambda_macro)
        self.lambda_critic = float(lambda_critic)
        hidden_size = int(getattr(base_model.config, "hidden_size"))
        self.macro_head = nn.Linear(hidden_size, len(ORCHESTRATION_MODES))
        self.critic_head = nn.Linear(hidden_size, len(ORCHESTRATION_MODES))

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        torch_dtype: torch.dtype | str | None = None,
        lambda_macro: float = 0.25,
        lambda_critic: float = 1.0,
        trust_remote_code: bool = True,
    ) -> tuple["MaestroOrchestratorModel", Any]:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.add_special_tokens(
            {"additional_special_tokens": [ORCH_TOKEN]}
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )
        if len(tokenizer) != base_model.get_input_embeddings().num_embeddings:
            base_model.resize_token_embeddings(len(tokenizer))
        orch_token_id = tokenizer.convert_tokens_to_ids(ORCH_TOKEN)
        model = cls(
            base_model,
            orch_token_id=orch_token_id,
            lambda_macro=lambda_macro,
            lambda_critic=lambda_critic,
        )
        heads_path = Path(model_name_or_path) / "maestro_heads.pt"
        if heads_path.exists():
            state = torch.load(heads_path, map_location="cpu")
            model.macro_head.load_state_dict(state["macro_head"])
            model.critic_head.load_state_dict(state["critic_head"])
        return model, tokenizer

    def _orch_hidden(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        mask = input_ids.eq(self.orch_token_id)
        if not bool(mask.any(dim=1).all()):
            raise ValueError(
                f"every training example must contain {ORCH_TOKEN!r}"
            )
        positions = torch.arange(
            input_ids.size(1),
            device=input_ids.device,
        ).unsqueeze(0).expand_as(input_ids)
        orch_positions = positions.masked_fill(~mask, -1).max(dim=1).values
        batch = torch.arange(input_ids.size(0), device=input_ids.device)
        return hidden_states[batch, orch_positions]

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        macro_labels: torch.Tensor | None = None,
        critic_targets: torch.Tensor | None = None,
        critic_mask: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor | None]:
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
        )
        orch_hidden = self._orch_hidden(outputs.hidden_states[-1], input_ids)
        macro_logits = self.macro_head(orch_hidden)
        critic_values = self.critic_head(orch_hidden)

        total_loss = outputs.loss
        macro_loss = None
        critic_loss = None

        if macro_labels is not None:
            macro_loss = F.cross_entropy(macro_logits, macro_labels)
            weighted = self.lambda_macro * macro_loss
            total_loss = weighted if total_loss is None else total_loss + weighted

        if critic_targets is not None:
            if critic_mask is None:
                critic_mask = torch.ones_like(critic_targets)
            squared = (critic_values.float() - critic_targets.float()).pow(2)
            denom = critic_mask.float().sum().clamp_min(1.0)
            critic_loss = (squared * critic_mask.float()).sum() / denom
            weighted = self.lambda_critic * critic_loss
            total_loss = weighted if total_loss is None else total_loss + weighted

        return {
            "loss": total_loss,
            "lm_loss": outputs.loss,
            "macro_loss": macro_loss,
            "critic_loss": critic_loss,
            "logits": outputs.logits,
            "macro_logits": macro_logits,
            "critic_values": critic_values,
            "orch_hidden": orch_hidden,
        }

    def freeze_for_critic(self, *, unfreeze_last_n_layers: int = 0) -> None:
        """Freeze policy/LM parameters; optionally unfreeze last backbone layers."""
        for parameter in self.base_model.parameters():
            parameter.requires_grad = False
        for parameter in self.macro_head.parameters():
            parameter.requires_grad = False
        for parameter in self.critic_head.parameters():
            parameter.requires_grad = True

        if unfreeze_last_n_layers <= 0:
            return
        backbone = getattr(self.base_model, "model", None)
        layers = getattr(backbone, "layers", None)
        if layers is None:
            raise ValueError(
                "cannot locate backbone layers for partial critic fine-tuning"
            )
        for layer in layers[-int(unfreeze_last_n_layers):]:
            for parameter in layer.parameters():
                parameter.requires_grad = True

    def save_heads(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "macro_head": self.macro_head.state_dict(),
                "critic_head": self.critic_head.state_dict(),
            },
            output_dir / "maestro_heads.pt",
        )
        (output_dir / "maestro_head_config.json").write_text(
            json.dumps(
                {
                    "orch_token": ORCH_TOKEN,
                    "orch_token_id": self.orch_token_id,
                    "orchestration_modes": list(ORCHESTRATION_MODES),
                    "lambda_macro": self.lambda_macro,
                    "lambda_critic": self.lambda_critic,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def save_pretrained(self, output_dir: str | Path, tokenizer=None) -> None:
        """Save the fine-tuned LM plus MAESTRO heads in one directory."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.base_model.save_pretrained(output_dir)
        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir)
        self.save_heads(output_dir)


__all__ = ["MaestroOrchestratorModel"]
