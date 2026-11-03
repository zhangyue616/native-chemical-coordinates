"""Single-coordinate, real post-GELU alpha hook; no cached activation replay."""
from __future__ import annotations
import math
import torch
from chem_model import PRIMARY_LAYER


class AlphaCoordinateHook:
    def __init__(self, unit, native_mean, alpha, mask, outgoing_norm):
        if not isinstance(unit, int) or not 0 <= unit < 464:
            raise ValueError("invalid physical native unit")
        if alpha not in (0.0, 0.5, 1.0) or not math.isfinite(native_mean):
            raise ValueError("unlocked alpha or nonfinite native D mean")
        self.unit, self.mean, self.alpha = unit, native_mean, alpha
        self.mask, self.norm = mask, outgoing_norm
        self.original = self.applied = self.dose = None

    def __call__(self, module, inputs, activation):
        if activation.shape[-1] != 464 or self.mask.shape != activation.shape[:2] or self.mask.dtype != torch.bool:
            raise ValueError("post-GELU/mask geometry differs")
        original = activation[..., self.unit]
        if self.alpha == 1.0:
            target = torch.full_like(original, self.mean)
        elif self.alpha == 0.0:
            target = original
        else:
            target = original + self.alpha * (self.mean - original)
        changed = activation.clone()
        changed[..., self.unit] = torch.where(self.mask, target, original)
        difference = changed[..., self.unit] - original
        counts = self.mask.sum(1)
        rms = (difference.square().sum(1) / counts.clamp_min(1)).sqrt()
        self.dose = torch.where(counts > 0, rms * self.norm.to(activation), torch.full_like(rms, float("nan")))
        self.original, self.applied = activation.detach().clone(), changed.detach().clone()
        return changed


def coordinate_forward(model, batch, *, unit, native_mean, alpha, scope):
    """The external hook runs before the unchanged wrapper's capture-only hook."""
    mask = batch.scope(scope)
    hook = AlphaCoordinateHook(unit, native_mean, alpha, mask,
                               model.output_column_norms(PRIMARY_LAYER)[unit])
    handle = model.roberta.encoder.layer[PRIMARY_LAYER].intermediate.register_forward_hook(hook)
    try:
        result = model(batch, layer_index=PRIMARY_LAYER)
    finally:
        handle.remove()
    if hook.dose is None:
        raise RuntimeError("real post-GELU hook was not called")
    return result, hook
