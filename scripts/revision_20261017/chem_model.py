"""Frozen ChemBERTa-MTR engineering interface; no data acquisition or training.

Only the actual output of RobertaIntermediate (post-GELU) is exposed as a.
The 199 checkpoint head outputs are anonymous normalized coordinates.  Native
mean replacement is the only implemented intervention.  Rotations are exposed
only as feature transforms, never as an extra model-forward condition.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor, nn
from transformers import RobertaConfig, RobertaModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "assets/chemberta_mtr_model"
TOKENIZER_DIR = ROOT / "assets/chemberta_mtr_tokenizer"
REPOSITORY = "DeepChem/ChemBERTa-77M-MTR"
REVISION = "66b895cab8adebea0cb59a8effa66b2020f204ca"
CHECKPOINT_SHA256 = "7882205587d3985c07e425632d0e57ba0fc2fc778e3b47aab02aef9b7961407f"
CONFIG_SHA256 = "6c15d3fd119919afb92b762c4e55cd46a595251ab799435168baa75a6ebd07e6"
PRIMARY_LAYER = 1  # zero-based; penultimate of three layers
TOKENIZER_SPECIAL_IDS = (0, 11, 12, 13, 14, 591, 592)


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class TokenBatch:
    input_ids: Tensor
    attention_mask: Tensor
    special_tokens_mask: Tensor
    atom_mask: Tensor
    pad_id: int = 0
    cls_id: int = 12

    def validate(self):
        shape = self.input_ids.shape
        if self.input_ids.ndim != 2 or not shape[0] or not shape[1]:
            raise ValueError("input_ids must be a nonempty batch x tokens tensor")
        if self.input_ids.dtype != torch.long:
            raise ValueError("input_ids must have dtype torch.long")
        for name in ("attention_mask", "special_tokens_mask", "atom_mask"):
            value = getattr(self, name)
            if value.shape != shape or value.device != self.input_ids.device:
                raise ValueError(f"{name} shape/device differs from input_ids")
            if not torch.all((value == 0) | (value == 1)):
                raise ValueError(f"{name} must be binary")
        attended = self.attention_mask.bool()
        actual_specials = torch.isin(self.input_ids, self.input_ids.new_tensor(TOKENIZER_SPECIAL_IDS))
        if torch.any(actual_specials & ~self.special_tokens_mask.bool()):
            raise ValueError("actual official tokenizer special IDs must be marked special")
        if not torch.all(attended[:, 0]) or not torch.all(self.input_ids[:, 0] == self.cls_id):
            raise ValueError("first token must be the official tokenizer CLS")
        if torch.any(self.atom_mask.bool() & (~attended | self.special_tokens_mask.bool())):
            raise ValueError("atom masks may never contain special or PAD tokens")
        if torch.any((~attended) & (self.input_ids != self.pad_id)):
            raise ValueError("unattended positions must use the official tokenizer PAD ID")
        if torch.any(attended[:, 1:] & ~attended[:, :-1]):
            raise ValueError("only right padding is supported")
        return self

    def scope(self, name: str, selected: Tensor | None = None) -> Tensor:
        """atoms excludes all specials; all includes attended CLS/SEP explicitly.

        scoped is an explicit subset of attended positions, including specials
        only if the caller deliberately selects them. PAD is always forbidden.
        """
        self.validate()
        if name == "atoms":
            mask = self.atom_mask.bool()
        elif name == "cls":
            mask = torch.zeros_like(self.attention_mask, dtype=torch.bool)
            mask[:, 0] = True
        elif name == "all":
            mask = self.attention_mask.bool()
        elif name == "scoped":
            if selected is None or selected.shape != self.input_ids.shape:
                raise ValueError("scoped mode requires a same-shape mask")
            if selected.device != self.input_ids.device or not torch.all((selected == 0) | (selected == 1)):
                raise ValueError("scoped mask must be binary and on the same device")
            mask = selected.bool()
        else:
            raise ValueError("scope must be atoms, cls, all, or scoped")
        if torch.any(mask & ~self.attention_mask.bool()):
            raise ValueError("scope may not include PAD")
        return mask


def batch_from_audits(audits: Sequence[dict], *, pad_to: int | None = None,
                      device: str | torch.device = "cpu") -> TokenBatch:
    """Consume accepted mapper audits; do not choose a scientific population.

    The mapper owns graph attribution and molecule exclusion. No retokenization,
    canonicalization, truncation, or token-level identity repair occurs here.
    """
    if not audits:
        raise ValueError("empty audits")
    for audit in audits:
        identity = audit.get("tokenizer", {})
        if (identity.get("repository"), identity.get("revision")) != (REPOSITORY, REVISION):
            raise ValueError("wrong mapper tokenizer identity")
        if not audit.get("accepted") or audit.get("gate") != "PASS_POSITIONAL_ELEMENT_ANCHOR":
            raise ValueError("mapper audit must explicitly accept every molecule")
    width = max(len(a["input_ids"]) for a in audits)
    if pad_to is not None:
        if pad_to < width:
            raise ValueError("truncation is forbidden")
        width = pad_to
    ids = torch.zeros((len(audits), width), dtype=torch.long, device=device)
    attention = torch.zeros_like(ids)
    specials = torch.ones_like(ids, dtype=torch.bool)
    atoms = torch.zeros_like(ids, dtype=torch.bool)
    for row, audit in enumerate(audits):
        n = len(audit["input_ids"])
        if any(len(audit[key]) != n for key in ("attention_mask", "special_tokens_mask")):
            raise ValueError("audit encoding field lengths differ")
        ids[row, :n] = torch.tensor(audit["input_ids"], device=device)
        attention[row, :n] = torch.tensor(audit["attention_mask"], device=device)
        specials[row, :n] = torch.tensor(audit["special_tokens_mask"], dtype=torch.bool, device=device)
        positions = [a["anchor_token_index"] for a in audit["atoms"]]
        if len(positions) != audit["graph_atom_count"] or len(set(positions)) != len(positions):
            raise ValueError("atom-anchor bijection differs from graph atom count")
        if not all(a["anchor_offset_verified"] for a in audit["atoms"]):
            raise ValueError("unverified anchor")
        if any(not isinstance(p, int) or not 0 <= p < n for p in positions):
            raise ValueError("invalid atom-anchor position")
        atoms[row, positions] = True
    return TokenBatch(ids, attention, specials, atoms).validate()


class MTRRegressionHead(nn.Module):
    """Official fixed source: CLS -> dropout -> linear -> ReLU -> dropout -> linear."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features: Tensor) -> Tensor:
        x = self.dropout(features[:, 0, :])
        x = torch.relu(self.dense(x))
        return self.out_proj(self.dropout(x))


@dataclass(frozen=True)
class NativeMeanIntervention:
    """Discovery means must be supplied from a separately frozen D artifact.

    Candidate [0,464) is +a; [464,928) is -a. A negative candidate and its
    correctly signed D mean produce the same physical unit replacement as the
    corresponding positive candidate. Never select both signs of one unit.
    """
    candidate_indices: tuple[int, ...]
    discovery_means: Tensor
    token_mask: Tensor


@dataclass(frozen=True)
class ActivationReplacement:
    """Absolute same-point replacement for sham and structural engineering checks."""
    values: Tensor
    token_mask: Tensor


@dataclass
class ForwardOutput:
    normalized_predictions: Tensor
    activations: Tensor
    applied_activations: Tensor
    dose: Tensor | None
    last_hidden_state: Tensor
    layer_index: int


def signed_features(a: Tensor, rotation: Tensor | None = None) -> Tensor:
    """Return exactly 2F candidates [a,-a] or [aQ,-aQ]; no model forward change."""
    if rotation is not None:
        if rotation.shape != (a.shape[-1], a.shape[-1]):
            raise ValueError("rotation shape must equal FFN width x FFN width")
        rotation = rotation.to(device=a.device, dtype=a.dtype)
        identity = torch.eye(rotation.shape[0], device=a.device, dtype=a.dtype)
        if not torch.allclose(rotation.T @ rotation, identity, atol=2e-5, rtol=2e-5):
            raise ValueError("Q must be orthogonal")
        a = a @ rotation
    return torch.cat((a, -a), dim=-1)


class ChembertaMTR(RobertaPreTrainedModel):
    """Frozen original encoder/head with explicit normalized prediction semantics."""

    def __init__(self, config: RobertaConfig):
        if (config.num_hidden_layers, config.hidden_size, config.intermediate_size,
                config.num_labels, config.hidden_act) != (3, 384, 464, 199, "gelu"):
            raise ValueError("config is not the locked 3-layer/384/464/199/GELU model")
        config._attn_implementation = "eager"
        super().__init__(config)
        self.register_buffer("norm_mean", torch.tensor(config.norm_mean, dtype=torch.float32))
        self.register_buffer("norm_std", torch.tensor(config.norm_std, dtype=torch.float32))
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.regression = MTRRegressionHead(config)
        self.post_init()
        self.norm_std.masked_fill_(self.norm_std == 0, 1)
        self.requires_grad_(False)
        self.eval()
        self.load_receipt = {"initialization": "from_config", "output_names": "anonymous_0_to_198"}

    @classmethod
    def from_official_checkpoint(cls, model_dir: Path | str = MODEL_DIR):
        model_dir = Path(model_dir)
        if _hash(model_dir / "config.json") != CONFIG_SHA256:
            raise ValueError("official config SHA256 mismatch")
        if _hash(model_dir / "pytorch_model.bin") != CHECKPOINT_SHA256:
            raise ValueError("official checkpoint SHA256 mismatch")
        config = RobertaConfig.from_json_file(str(model_dir / "config.json"))
        model = cls(config)
        state = torch.load(model_dir / "pytorch_model.bin", map_location="cpu", weights_only=True)
        ignored = []
        key = "roberta.embeddings.position_ids"
        if key in state:
            expected = torch.arange(config.max_position_embeddings).expand((1, -1))
            if not torch.equal(state[key], expected):
                raise ValueError("legacy checkpoint position_ids are unexpected")
            state.pop(key)
            ignored.append(key)
        model.load_state_dict(state, strict=True)
        # Checkpoint normalization buffers take precedence over config values.
        zeros = int((model.norm_std == 0).sum())
        model.norm_std.masked_fill_(model.norm_std == 0, 1)
        if model.norm_mean.shape != (199,) or model.norm_std.shape != (199,):
            raise ValueError("normalization buffer shape differs from anonymous head")
        if not torch.isfinite(model.norm_mean).all() or not torch.isfinite(model.norm_std).all():
            raise ValueError("nonfinite normalization buffer")
        if not torch.all(model.norm_std > 0):
            raise ValueError("normalization std must be positive after zero replacement")
        model.load_receipt = {
            "repository": REPOSITORY, "revision": REVISION,
            "checkpoint_sha256": CHECKPOINT_SHA256, "config_sha256": CONFIG_SHA256,
            "strict_state_dict_match": True, "legacy_nonpersistent_buffers_omitted": ignored,
            "normalization_source": "checkpoint_buffers", "zero_std_replaced_by_one": zeros,
            "output_names": "anonymous_0_to_198", "default_output": "normalized",
            "official_config_pad_id": config.pad_token_id,
            "tokenizer_pad_id": 0, "tokenizer_cls_id": 12,
        }
        return model

    @classmethod
    def initialized_from_config(cls, seed: int, config_path: Path | str = MODEL_DIR / "config.json"):
        """Same encoder/head architecture; caller must freeze reference seeds first.

        An initialized reference is descriptive, not an exchangeable causal null.
        No checkpoint is read and no training or real-data forward is performed.
        """
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = cls(RobertaConfig.from_json_file(str(config_path)))
        model.load_receipt.update({"seed": seed, "reference_role": "descriptive_initialization_reference"})
        return model

    def unnormalize_predictions(self, normalized: Tensor) -> Tensor:
        """Separate explicit interface; output coordinate identities remain unknown."""
        return normalized * self.norm_std + self.norm_mean

    def output_column_norms(self, layer_index: int = PRIMARY_LAYER) -> Tensor:
        return self.roberta.encoder.layer[layer_index].output.dense.weight.norm(dim=0)

    @torch.inference_mode()
    def forward(self, batch: TokenBatch, *, layer_index: int = PRIMARY_LAYER,
                intervention: NativeMeanIntervention | ActivationReplacement | None = None) -> ForwardOutput:
        if self.training:
            raise RuntimeError("engineering wrapper requires eval mode")
        batch.validate()
        if layer_index not in range(self.config.num_hidden_layers):
            raise ValueError("invalid zero-based layer index")
        if batch.input_ids.shape[1] > 512:
            raise ValueError("locked token window is 512; truncation is forbidden")
        if torch.any(batch.input_ids < 0) or torch.any(batch.input_ids >= self.config.vocab_size):
            raise ValueError("input ID outside checkpoint vocabulary")
        if batch.input_ids.device != next(self.parameters()).device:
            raise ValueError("batch and model devices differ")
        captured = {}

        def at_post_gelu(module, inputs, a):
            captured["a"] = a.clone()
            changed = a
            dose = None
            if intervention is not None:
                mask = intervention.token_mask
                if mask.shape != a.shape[:2] or mask.device != a.device or mask.dtype != torch.bool:
                    raise ValueError("intervention scope must be same-shape boolean on model device")
                if torch.any(mask & ~batch.attention_mask.bool()):
                    raise ValueError("intervention scope includes PAD")
                if isinstance(intervention, ActivationReplacement):
                    replacement = intervention.values
                    if replacement.shape != a.shape or not torch.isfinite(replacement).all():
                        raise ValueError("invalid full activation replacement")
                    changed = torch.where(mask.unsqueeze(-1), replacement.to(a), a)
                else:
                    candidates = intervention.candidate_indices
                    width = a.shape[-1]
                    if not candidates or any(not isinstance(i, int) or not 0 <= i < 2 * width for i in candidates):
                        raise ValueError("native candidate index outside [0,928)")
                    units = [i % width for i in candidates]
                    if len(set(units)) != len(units):
                        raise ValueError("duplicate physical unit, including opposite signs")
                    means = intervention.discovery_means.to(a)
                    if means.shape != (len(units),) or not torch.isfinite(means).all():
                        raise ValueError("provide one finite frozen-D signed mean per candidate")
                    signs = a.new_tensor([1 if i < width else -1 for i in candidates])
                    native_means = means * signs
                    original = a[..., units]
                    replaced = torch.where(mask.unsqueeze(-1), native_means, original)
                    changed = a.clone()
                    changed[..., units] = replaced
                    counts = mask.sum(dim=1)
                    # Per molecule, per unit: RMS over selected tokens only, then ||W[:,j]||2.
                    rms = ((replaced - original).square().sum(dim=1) / counts.clamp_min(1).unsqueeze(-1)).sqrt()
                    dose = rms * self.output_column_norms(layer_index)[units]
                    dose = torch.where((counts > 0).unsqueeze(-1), dose, torch.full_like(dose, float("nan")))
            captured["applied"] = changed.clone()
            captured["dose"] = dose
            return changed

        handle = self.roberta.encoder.layer[layer_index].intermediate.register_forward_hook(at_post_gelu)
        try:
            result = self.roberta(input_ids=batch.input_ids, attention_mask=batch.attention_mask,
                                  return_dict=True)
            predictions = self.regression(result.last_hidden_state)
        finally:
            handle.remove()
        return ForwardOutput(predictions, captured["a"], captured["applied"],
                             captured["dose"], result.last_hidden_state, layer_index)


def engineering_identity() -> dict:
    """A small metadata record; it contains no scientific dataset identifiers."""
    return {"repository": REPOSITORY, "revision": REVISION, "primary_layer_zero_based": PRIMARY_LAYER,
            "a": "actual RobertaIntermediate output = GELU(z)", "native_candidates": 928,
            "rotated_feature_candidates": 928, "interventions": "native_only",
            "head": "CLS/dropout/Linear384/ReLU/dropout/Linear199",
            "output_coordinate_identity": "anonymous; exact official descriptor order not recovered",
            "default_predictions": "normalized", "normalization": "checkpoint buffers; std0 becomes1"}
