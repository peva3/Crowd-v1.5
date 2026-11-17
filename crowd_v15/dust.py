"""DUST: activation-space node perturbation, batched over draws.

One step:

1. a clean pass gives per-layer input activations and clean logits;
2. each population draw adds ``sigma * a`` to every hidden layer output, then
   the same draw with ``-a``, so ``delta_t = (L+_t - L-_t) / 2``;
3. ``reward_t = sum_{s>=t} gamma**(s-t) delta_s``;
4. ``Grad_W = (1 / (sigma * tokens)) * sum_t reward_t a_t x_t^T``, with ``x_t``
   taken from the clean pass;
5. the head uses the exact softmax gradient rather than a zeroth-order estimate.

No backward pass and no optimizer state beyond optional momentum.

Noise is a pure function of ``(base_seed, step, draw, layer)``, so a coordinator
can rebuild any draw's perturbations from the seed instead of receiving them,
and chunking cannot change the estimate.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .model import GPT, ModelConfig, Perturb


@dataclass
class DustConfig:
    population: int = 64      # antithetic draws per step
    draw_chunk: int = 8       # draws evaluated per batched pass
    sigma: float = 1e-3
    # "relative" scales the noise per layer as ``sigma_rel * rms(y_l)``, which is
    # what keeps the perturbation above the compute dtype's resolution as width
    # grows. "absolute" keeps the historical single sigma.
    sigma_mode: str = "absolute"
    sigma_rel: float = 0.01
    # Upper bound on the per-layer noise scale in "relative" mode. Relative
    # scaling tracks ``RMS(layer output)``, and that reference is not stationary:
    # over 5.2 M tokens of the 24 h 300M arm the per-layer maximum grew 0.032 ->
    # 6.66 (x207) while total weight RMS grew x1.18, so the estimator finished in
    # a far coarser perturbation regime than it started in. ``None`` keeps the
    # historical unclamped behaviour; a value pins the perturbation to a known
    # magnitude so a long run cannot silently walk out of the regime it was
    # tuned for. See TODO G37/G38.
    sigma_max: float | None = None
    gamma: float = 0.98
    lr: float = 1e-3
    momentum: float = 0.0
    grad_clip: float = 0.0
    warmup_steps: int = 0
    mode: str = "layer"       # "layer" isolates each layer (paper); "all" is one pass
    # Where the perturbation noise is *generated*. "cpu" makes a draw bit-identical
    # on every device, which is what lets a coordinator re-score a client's member
    # from (base_seed, step, draw) alone. "device" generates on the accelerator
    # instead, which is the only way a 300M step is affordable: one draw of
    # (rows x seq x 240,640) noise costs ~7 s of CPU randn versus ~0.3 s on a
    # 4060 Ti, i.e. 98 % of the step. "device" forfeits cross-device bit-equality;
    # the round record names which mode produced the estimate.
    noise_device: str = "cpu"

    def __post_init__(self) -> None:
        if self.population <= 0:
            raise ValueError("population must be positive")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")
        if self.sigma_mode not in ("absolute", "relative"):
            raise ValueError('sigma_mode must be "absolute" or "relative"')
        if self.sigma_rel <= 0:
            raise ValueError("sigma_rel must be positive")
        if self.sigma_max is not None and self.sigma_max <= 0:
            raise ValueError("sigma_max must be positive when set")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if self.draw_chunk <= 0:
            raise ValueError("draw_chunk must be positive")
        if self.mode not in ("layer", "all"):
            raise ValueError('mode must be "layer" or "all"')
        if self.noise_device not in ("cpu", "device"):
            raise ValueError('noise_device must be "cpu" or "device"')


def derive_seed(base_seed: int, step: int, draw: int, name: str) -> int:
    """Stable 63-bit seed for one layer of one draw of one step."""
    digest = hashlib.blake2b(
        f"{int(base_seed)}:{int(step)}:{int(draw)}:{name}".encode("ascii"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "little") >> 1


def _draw_one(name: str, out_features: int, rows: int, seq_len: int, *,
              base_seed: int, step: int, draw: int,
              device: torch.device, dtype: torch.dtype,
              noise_device: str = "cpu") -> torch.Tensor:
    """One layer's noise; CPU generation keeps every device bit-identical."""
    where = "cpu" if noise_device == "cpu" else device
    gen = torch.Generator(device=where).manual_seed(derive_seed(base_seed, step, draw, name))
    noise = torch.randn((rows, seq_len, out_features), generator=gen,
                        device=where, dtype=torch.float32)
    if where == "cpu":
        return noise.to(device=device, dtype=dtype)
    return noise.to(dtype=dtype)


def draw_noise(shapes: dict[str, int], *, rows: int, seq_len: int,
               base_seed: int, step: int, draws: list[int],
               device: torch.device, dtype: torch.dtype,
               noise_device: str = "cpu") -> dict[str, torch.Tensor]:
    """Batched noise for ``draws``: each layer's tensor is ``(len(draws)*rows, T, out)``."""
    batched: dict[str, torch.Tensor] = {}
    for name, out_features in shapes.items():
        parts = [
            _draw_one(name, out_features, rows, seq_len, base_seed=base_seed,
                      step=step, draw=draw, device=device, dtype=dtype,
                      noise_device=noise_device)
            for draw in draws
        ]
        batched[name] = torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]
    return batched


def reverse_discounted_cumsum(losses: torch.Tensor, gamma: float) -> torch.Tensor:
    """``r_t = sum_{s>=t} gamma**(s-t) c_s`` for ``c`` shaped ``(N, T)``."""
    if losses.dim() != 2:
        raise ValueError("losses must be (N, T)")
    if gamma == 0.0:
        return losses.clone()
    out = torch.empty_like(losses)
    out[:, -1] = losses[:, -1]
    for t in range(losses.shape[1] - 2, -1, -1):
        out[:, t] = losses[:, t] + gamma * out[:, t + 1]
    return out


def per_token_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy per position, shape ``(B, T)``."""
    vocab = logits.shape[-1]
    return F.cross_entropy(
        logits.reshape(-1, vocab), targets.reshape(-1), reduction="none"
    ).view(targets.shape)


def _param_map(model: GPT) -> dict[str, torch.Tensor]:
    """Canonical parameter name -> tensor.

    Tied embeddings make ``head.weight`` and ``tok_emb.weight`` the same tensor,
    so a plain ``named_parameters()`` dict would silently drop one of the two
    names. Keep the first occurrence and treat it as the canonical key.
    """
    seen: dict[int, str] = {}
    canonical: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters(remove_duplicate=False):
        key = id(param)
        if key in seen:
            continue
        seen[key] = name
        canonical[name] = param
    return canonical


def _head_name(model: GPT) -> str:
    target = model.head.weight
    for name, param in _param_map(model).items():
        if param is target:
            return name
    raise RuntimeError("head weight is not registered as a parameter")


def dust_gradients(model: GPT, cfg: DustConfig, idx: torch.Tensor, tgt: torch.Tensor,
                   *, base_seed: int, step: int) -> tuple[dict[str, torch.Tensor], dict]:
    """Return float32 gradient estimates keyed by parameter name, plus stats.

    ``mode="layer"`` perturbs one layer per pass, which is what the paper does:
    with every layer jittered at once, one token's loss change is a sum over all
    layers, so each layer's own signal is diluted by ``~sqrt(n_layers)``. The
    isolated passes cost more forwards and buy a much better-conditioned
    estimate. ``mode="all"`` is the cheap approximation.
    """
    if idx.shape != tgt.shape:
        raise ValueError("idx and tgt must share a shape")
    shapes = model.noise_shapes()
    params = _param_map(model)
    missing = [f"{n}.weight" for n in shapes if f"{n}.weight" not in params]
    if missing:
        raise RuntimeError(f"perturbed layers without parameters: {missing}")

    b, t = idx.shape
    device = idx.device
    dtype = model.tok_emb.weight.dtype
    vocab = model.cfg.vocab_size
    hidden_dim = model.cfg.d_model
    total_tokens = cfg.population * b * t
    grads = {name: torch.zeros_like(param, dtype=torch.float32)
             for name, param in params.items()}
    head_grad = grads[_head_name(model)]
    per_pass = [("all", None, shapes)] if cfg.mode == "all" else [
        (name, name, {name: width}) for name, width in shapes.items()
    ]

    with torch.no_grad():
        for start in range(0, cfg.population, cfg.draw_chunk):
            draws = list(range(start, min(start + cfg.draw_chunk, cfg.population)))
            rows = len(draws)
            idx_rep = idx.repeat(rows, 1).contiguous()
            tgt_rep = tgt.repeat(rows, 1).contiguous()

            record: dict[str, torch.Tensor] = {}
            layer_scales: dict[str, torch.Tensor] = {}
            clean_logits = model(idx_rep, Perturb(record=record, scales=layer_scales))

            relative = cfg.sigma_mode == "relative"
            if relative:
                # ``sigmas`` keeps the *raw* per-layer scale, before any cap. The
                # cap exists to stop an unbounded drift, so the stat that reports
                # the drift has to be the pre-cap value: reporting the post-cap one
                # makes a clamped arm look stationary by construction.
                sigmas = {name: cfg.sigma_rel * float(value)
                          for name, value in layer_scales.items()}
                sigma_map = sigmas
                if cfg.sigma_max is not None:
                    sigma_map = {name: min(value, cfg.sigma_max)
                                 for name, value in sigmas.items()}
            else:
                sigma_map = cfg.sigma
                sigmas = {}

            # The head is linear in its input, so its gradient is exact.
            hidden = record["__head__"].reshape(-1, hidden_dim).float()
            probs = torch.softmax(clean_logits.detach().float(), dim=-1).reshape(-1, vocab)
            probs[torch.arange(probs.shape[0], device=probs.device),
                  tgt_rep.reshape(-1)] -= 1.0
            head_grad += (probs.t() @ hidden) / total_tokens

            default_scale = cfg.sigma * total_tokens
            for _label, only, targets in per_pass:
                noise = draw_noise(targets, rows=b, seq_len=t, base_seed=base_seed,
                                   noise_device=cfg.noise_device,
                                   step=step, draws=draws, device=device, dtype=dtype)
                plus = model(idx_rep, Perturb(enabled=True, sigma=sigma_map, sign=1,
                                              noise=noise, only=only))
                minus = model(idx_rep, Perturb(enabled=True, sigma=sigma_map, sign=-1,
                                                noise=noise, only=only))
                delta = 0.5 * (per_token_ce(plus, tgt_rep) - per_token_ce(minus, tgt_rep))
                reward = reverse_discounted_cumsum(delta, cfg.gamma).reshape(-1, 1).float()

                for name, width in targets.items():
                    # Each layer is divided by the sigma that produced its own
                    # noise, otherwise a relative scale would bias wide layers.
                    scale = (sigmas[name] * total_tokens) if relative else default_scale
                    if scale <= 0.0:
                        continue
                    weighted = noise[name].reshape(-1, width).float() * reward
                    key = f"{name}.weight"
                    if name == "tok_emb":
                        # One-hot input: the outer product is a scatter-add into
                        # the rows of the tokens that actually appeared.
                        grads[key].index_add_(0, record["tok_emb"].reshape(-1),
                                              weighted / scale)
                    else:
                        x_flat = record[name].reshape(-1, record[name].shape[-1]).float()
                        grads[key] += (weighted.t() @ x_flat) / scale

    # ``sigma``/``sigmaMaxApplied`` describe what the model was actually perturbed
    # with; ``sigmaMax`` describes the drift the cap is fighting.
    applied = sigma_map if isinstance(sigma_map, dict) else None
    stats = {
        "population": cfg.population,
        "drawChunk": cfg.draw_chunk,
        "mode": cfg.mode,
        "draws": cfg.population,
        "tokens": total_tokens,
        "forwards": math.ceil(cfg.population / cfg.draw_chunk) * (1 + 2 * len(per_pass)),
        "sigma": (min(applied.values()) if applied else cfg.sigma),
        "noiseDevice": cfg.noise_device,
        "sigmaMode": cfg.sigma_mode,
        "sigmaRel": cfg.sigma_rel if sigmas else None,
        "sigmaMax": (max(sigmas.values()) if sigmas else cfg.sigma),
        "sigmaMaxApplied": (max(applied.values()) if applied else cfg.sigma),
        "sigmaCap": cfg.sigma_max,
        "sigmaClamped": (sum(1 for value in sigmas.values() if value > cfg.sigma_max)
                         if (sigmas and cfg.sigma_max is not None) else 0),
        "gamma": cfg.gamma,
        "gradNorm": float(math.sqrt(sum(float(g.pow(2).sum()) for g in grads.values()))),
    }
    return grads, stats


class DustOptimizer:
    """SGD with optional momentum, no per-parameter state unless momentum is on."""

    def __init__(self, model: GPT, lr: float, momentum: float = 0.0,
                 grad_clip: float = 0.0) -> None:
        self.model = model
        self.lr = lr
        self.momentum = momentum
        self.grad_clip = grad_clip
        self.params = _param_map(model)
        self.buffers: dict[str, torch.Tensor] = {}

    def step(self, grads: dict[str, torch.Tensor]) -> dict:
        if self.grad_clip > 0:
            total = math.sqrt(sum(float(g.pow(2).sum()) for g in grads.values()))
            if total > self.grad_clip:
                factor = self.grad_clip / total
                grads = {name: g * factor for name, g in grads.items()}

        applied = 0
        with torch.no_grad():
            for name, grad in grads.items():
                param = self.params.get(name)
                if param is None:
                    continue
                update = grad.to(param.dtype)
                if self.momentum > 0:
                    buf = self.buffers.get(name)
                    buf = update if buf is None else self.momentum * buf + update
                    self.buffers[name] = buf
                    update = buf
                param -= self.lr * update
                applied += 1
        return {"applied": applied, "parameters": len(self.params)}

    def lr_at(self, step: int, warmup_steps: int, base_lr: float) -> float:
        if warmup_steps <= 0:
            self.lr = base_lr
        else:
            self.lr = base_lr * min(1.0, (step + 1) / warmup_steps)
        return self.lr


def backprop_gradients(model: GPT, idx: torch.Tensor, tgt: torch.Tensor) -> dict[str, torch.Tensor]:
    """Reference gradient, only used by :func:`probe_cosine`. Never in training."""
    model.zero_grad(set_to_none=True)
    logits = model(idx, None)
    loss = per_token_ce(logits, tgt).mean()
    loss.backward()
    grads = {
        name: p.grad.detach().clone().float()
        for name, p in _param_map(model).items()
        if p.grad is not None
    }
    model.zero_grad(set_to_none=True)
    return grads


def probe_cosine(model: GPT, cfg: DustConfig, idx: torch.Tensor, tgt: torch.Tensor, *,
                 base_seed: int, step: int) -> dict:
    """Cosine alignment between the DUST estimate and the backprop gradient.

    This is the paper's Figure 5 diagnostic and the second of the v1.5 gates:
    if the estimate stops pointing at the descent direction as models grow,
    the method does not scale.
    """
    dust, _stats = dust_gradients(model, cfg, idx, tgt, base_seed=base_seed, step=step)
    reference = backprop_gradients(model, idx, tgt)

    def cosine(a: torch.Tensor, b: torch.Tensor) -> float | None:
        a = a.reshape(-1).float()
        b = b.reshape(-1).float()
        denom = float(a.norm() * b.norm())
        if denom == 0.0:
            return None
        return float(torch.dot(a, b) / denom)

    per_layer: dict[str, float | None] = {}
    for name, grad in dust.items():
        ref = reference.get(name)
        per_layer[name] = None if ref is None else cosine(grad, ref)

    grouped: dict[str, list[float]] = {}
    flat_dust, flat_ref = [], []
    # With tied embeddings the head's weights are ``tok_emb.weight``, so comparing
    # against the literal name "head.weight" silently bucketed the embedding with
    # the MLP blocks. Resolve the canonical key instead.
    head_key = _head_name(model)
    for name, value in per_layer.items():
        if value is None:
            continue
        kind = "embedding" if name == head_key else (
            "attn" if ".attn." in name else "mlp"
        )
        grouped.setdefault(kind, []).append(value)
        flat_dust.append(dust[name].reshape(-1))
        flat_ref.append(reference[name].reshape(-1))

    by_type = {k: sum(v) / len(v) for k, v in grouped.items()}
    overall = None
    if flat_dust:
        overall = cosine(torch.cat(flat_dust), torch.cat(flat_ref))
    return {"overall": overall, "byType": by_type, "perLayer": per_layer}


def dust_step(model: GPT, optimizer: DustOptimizer, cfg: DustConfig,
              idx: torch.Tensor, tgt: torch.Tensor, *,
              base_seed: int, step: int) -> dict:
    """One full DUST update; returns the stats plus what the optimizer applied."""
    grads, stats = dust_gradients(model, cfg, idx, tgt, base_seed=base_seed, step=step)
    stats.update(optimizer.step(grads))
    return stats
