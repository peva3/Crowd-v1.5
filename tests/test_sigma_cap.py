"""The relative-sigma clamp: long runs must not drift out of their tuning regime."""

from __future__ import annotations

import pytest
import torch

from crowd_v15.dust import DustConfig, dust_gradients
from crowd_v15.model import GPT, ModelConfig


def _setup():
    torch.manual_seed(0)
    model = GPT(ModelConfig.preset("tiny")).eval()
    idx = torch.randint(0, model.cfg.vocab_size, (2, 16))
    return model, idx, torch.randint(0, model.cfg.vocab_size, (2, 16))


def _run(sigma_mode="relative", **kwargs):
    model, idx, tgt = _setup()
    cfg = DustConfig(population=4, draw_chunk=1, mode="all", sigma_mode=sigma_mode,
                     sigma_rel=0.05, **kwargs)
    _, stats = dust_gradients(model, cfg, idx, tgt, base_seed=0, step=0)
    return stats


def test_cap_bounds_every_layer_sigma():
    unclamped = _run()
    cap = unclamped["sigmaMax"] * 0.5
    clamped = _run(sigma_max=cap)
    # sigmaMax reports the raw drift; the applied scale is what the cap bounds.
    assert clamped["sigmaMax"] > cap
    assert clamped["sigmaMaxApplied"] <= cap + 1e-12
    assert clamped["sigmaCap"] == cap
    assert clamped["sigmaClamped"] > 0


def test_cap_above_the_maximum_changes_nothing():
    unclamped = _run()
    generous = _run(sigma_max=unclamped["sigmaMax"] * 10.0)
    assert generous["sigmaMax"] == pytest.approx(unclamped["sigmaMax"], rel=1e-6)
    assert generous["sigmaClamped"] == 0


def test_cap_does_not_affect_absolute_mode():
    stats = _run(sigma_mode="absolute", sigma=1e-2, sigma_max=1e-6)
    assert stats["sigma"] == pytest.approx(1e-2)
    assert stats["sigmaClamped"] == 0


def test_non_positive_cap_is_rejected():
    with pytest.raises(ValueError, match="sigma_max"):
        DustConfig(sigma_max=0.0)
