# Crowd-v1.5
**Experimental Fork of CrowdGPT - https://github.com/Vxtzq/CrowdGPT**

Pretraining by **DUST** - perturbing activations instead of backpropagating - with
a coordinator that accepts DUST and backprop submissions on **separate routes**.

This is the ground work for the v2 pivot: v1 trained a 1000 million parameters model with
backprop and shipped dense weight deltas. v1.5 trains a **200m plain transformer**
and lets any device contribute, because DUST needs no backward pass, no optimizer
state and no gradient buffer. Backprop stays in the tree as a route that is simply
unused when DUST is active.

## Plain transformer, no engrams

The model is a plain decoder-only transformer: embeddings, pre-norm blocks
(RMSNorm, fused-QKV attention, SwiGLU MLP), a final norm and a tied output head.
Nothing else - **no engram table, no MoE, no adapters, no LoRA**.

Engrams are removed due to incompatibilites (for now) with Evolution Strategies (ES)

## What changes

The rule is enforced in code, in three places:

- `assert_plain(model)` checks the module tree and returns a parameter breakdown;
- `Coordinator.bootstrap()` refuses any checkpoint whose parameter names contain
  `engram`, `moe`, `expert`, `router`, `lora` or `adapter` (400 at the HTTP edge);
- `DustClient.sync()` refuses a manifest that advertises such a tensor, before it
  downloads 2.5 GB of it.

| preset | layers | d_model | heads | context | parameters |
|---|---|---|---|---|---|
| `tiny` | 4 | 128 | 4 | 64 | 1,082,496 |
| `10m` | 6 | 256 | 8 | 512 | 7,343,360 |
| `100m` | 12 | 768 | 12 | 1024 | 116,411,136 |
| **`200m`** | **16** | **1024** | **16** | **1024** | **200,000,000 (to verify)** |
| **`1b`** | **15** | **2048** | **16** | **2048** | **1,015,085,056** |
| `1b-deep` | 20 | 2048 | 16 | 2048 | 1,350,649,856 |
| `2b` | 32 | 2304 | 18 | 2048 | 2,727,495,936 |

`1b` is the literal billion. `1b-deep` is the sizing the maintainer's
`train_dust.py` calls "1b", kept so earlier measurements stay comparable. Counts
include the tied embedding once and every norm weight; the analytic count, the
built model and `param_report()` are asserted equal for every preset.

## Why the two halves are separable

A DUST client still uploads a **weight-shaped object**. The estimator is

```
c_s = (L(+a)_s - L(-a)_s) / 2                    antithetic centring
r_t = sum_{s>=t} gamma^(s-t) c_s                 temporal credit
G_W = (1/sigma) sum_t (r_t a_t) x_t^T            outer product with the input
```

and that last line needs the layer inputs, which only the client has. So the
coordinator cannot recompute a client's gradient from seeds; it can only check
that the submission is well formed, bound to the weights it claims, and
plausibly produced by the evidence it reports. That is what this repo enforces:

- `baseSha256` binds a gradient to the weights it was measured from. A gradient
  measured at one point in parameter space is not a gradient at another.
- `tokens`, `population`, `sigma`, `gamma` are the estimator's own record. A DUST
  submission with zero tokens or a non-positive sigma is refused before it can
  influence a step.
- one submission per client per round, one round at a time, no route mixing.

## Layout

```
crowd_v15/
  model.py        GPT: RMSNorm, fused-QKV attention, SwiGLU MLP, tied embeddings
  dust.py         estimator, optimizer, cosine probe against backprop
  tasks.py        identity / induction / copy / random data sources
  codec.py        NGD1 and NGD1R gradient containers (the v1 wire format, reused)
  protocol.py     header + body framing and every guard around it
  store.py        atomic weights, status.json, audit log
  coordinator.py  who counts, how they combine, when the model moves
  server/app.py   POST /submit/dust, POST /submit/backprop, model + status reads
  client.py       a DUST client: verify weights, train, upload, re-sync
  accel.py        device selection + per-backend sync/memory/autocast (CUDA, ROCm, MPS, CPU)
tests/            89 tests, ~15 s on CPU
```

## Run it

```bash
pip install -e .[dev]
python -m pytest tests -q          # 89 passed

# coordinator
export CROWDGPT_ADMIN_SECRET=change-me
export CROWDGPT_V15_ROOT=./v15_state
python -m uvicorn --factory crowd_v15.server.app:create_app --port 8000

# install the weights it will serve (one-time; body is a safetensors file)
curl -X POST http://127.0.0.1:8000/admin/bootstrap \
     -H "X-Admin-Secret: change-me" --data-binary @weights.safetensors

# a client (CPU is fine; add --device mps on a Mac, or --device auto)
python -m crowd_v15.client --server http://127.0.0.1:8000 \
       --preset tiny --task identity --rounds 2 --population 32
```

On macOS the system `python3` is 3.9 and this needs >= 3.10:
`uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e '.[dev]'`.
Apple silicon runs on MPS (`--device mps`); `tests/test_mps.py` checks the MPS
estimate against CPU and skips elsewhere.

`--dry-run` trains and reports without uploading, which is how you check a port
before spending bandwidth.

## Routes

| route | who | accepted when |
|---|---|---|
| `POST /submit/dust` | a v1.5 client | route open, round current, `baseSha256` matches, tokens/population/sigma sane |
| `POST /submit/backprop` | a v3.5 client | route open, round current, `baseSha256` matches |

Both take the same container; the route in the URL and the route in the header
must agree, so a noisy estimate cannot be re-labelled as an exact gradient.

Read-only endpoints (`/status`, `/model/manifest`, `/model/weights`) are open:
a client needs the model before it can train, and the weights are public anyway.
Administration (`/admin/bootstrap`, `/admin/finalize`, `/admin/routes`) reads the
secret from `X-Admin-Secret` or `Authorization: Bearer`, never the query string.
With no secret configured, every admin route answers 503 — fail closed.

## Wire format

```
<u32 header length><header JSON><container>
```

The header is JSON (`schema`, `route`, `clientId`, `roundId`, `baseSha256`,
`codec`, `names`, `sizes`, `params`, plus the estimator fields). The container is
the v1 NGD family, unchanged:

| codec | frame | container | bits/value |
|---|---|---|---|
| `dense` | raw bf16 | — | 16 |
| `ngd1` | `NGD1` | `NGD2` | 1 + one fp16 scale per 1024 values |
| `ngd1r` | `NGD1R` | `NGD3` | 2 + two fp16 scales per 1024 values |

Measured on a 5,000-value random gradient: dense 10,255 B, `ngd1` 951 B (10.8×),
`ngd1r` 1,603 B (6.4×); sign agreement >98%, mean-absolute ratio `ngd1` 0.59 /
`ngd1r` 0.33 of the input scale. On a real tiny-model DUST gradient `ngd1r`
uploads ~8× smaller than dense.

Reusing the v1 container is deliberate: the quantisation error and the abuse
surface are already understood, and a v1.5 coordinator can read a v1 client's
delta with no new code path.

## What a step does

`Coordinator.finalize()` runs when the quorum (`min_submissions`) is reached:

1. average the pending gradients, weighting a DUST estimate by the tokens it
   reports (a backprop gradient gets unit weight — it is exact, not sampled);
2. optionally clip the global norm;
3. apply momentum, then `param -= lr * update`;
4. write the weights atomically, bump `round`, record history and the audit log.

A coordinator restart **drops** accepted-but-unapplied submissions rather than
applying a step nobody can audit: the round number has not moved, so clients
simply resubmit.

## Error feedback

The 1-bit containers are *biased*: every value in a block decodes to
`±mean|x|`, so a small value far from its block mean lands well off target, and
the same blocks are quantized the same way every round. Seide et al.
(*1-bit SGD*, Interspeech 2014) showed this costs almost nothing **if the
quantization error is carried forward** — encode `g + e` instead of `g`, then
keep `e' = (g + e) - Q(g + e)` for the next round. Karimireddy et al.
(*Error Feedback Fixes SignSGD*, ICML 2019) proved the property we rely on: with
feedback the compression factor appears only as a second-order term, whereas
sign-based compressors can actively diverge without it.

`--error-feedback` enables it; the state lives on the client, so it survives
rounds but not a restart (the first round after a restart sends the plain
estimate). The header advertises `errorFeedback` and `residualNorm`, and the
coordinator's audit log records the flag per submission.

Measured with `tools/bench_feedback.py`, 30 identity-task rounds through the real
coordinator (`lr 0.05`, `max_abs_grad 1.0`, tiny preset, population 8):

| arm | bytes over dense | final loss | gap to dense |
| --- | --- | --- | --- |
| dense (bf16) | 1.00x | **3.7947** | — |
| `ngd1` | 16.4x smaller | 4.1121 | +0.317 |
| `ngd1r` | 8.1x smaller | 4.0070 | +0.212 |
| `ngd1` + feedback | 15.8x smaller | **3.8863** | +0.092 |
| `ngd1r` + feedback | 7.9x smaller | 3.8956 | +0.101 |

So carrying the error recovers about 71 % of the accuracy the 1-bit codec gave
up, at the same payload size.

**What it does not buy.** `finalize` converts each submission to a unit direction
and re-applies a capped magnitude, so feedback sharpens the *direction*, not the
applied step size. On a biased quantizer whose blocks are already well scaled the
gain is small; the win is concentrated where compression is most aggressive. The
residual also assumes a stationary gradient — with a decaying one it can
overshoot, which is why the round count and `lr` matter in the table above.

## Outer optimizer (pseudo-gradient momentum)

The aggregated estimate is *not* a minibatch gradient. `finalize` averages unit
directions and rescales them to the mean declared norm, so the result keeps a
declared magnitude instead of shrinking as the model approaches an optimum. DiLoCo's
scaling work consequently uses a separate **pseudo-gradient learning rate** (`0.7`)
with Nesterov momentum (`0.9`) rather than a small SGD learning rate.

`CoordConfig` exposes that as `outer_lr` / `outer_momentum` (both `0`, i.e. off, by
default, which keeps the plain `lr * g` path byte-identical):

```python
CoordConfig(lr=1e-2, outer_lr=0.7, outer_momentum=0.9)
```

With a zero buffer the first step is `(1 + momentum) * outer_lr * p`; afterwards the
buffer carries across rounds and across coordinator restarts only through the
in-memory dict, so a restart begins with a cold buffer.

Measured with `tools/bench_outer.py --rounds 30` (identity task, tiny preset,
population 8, `max_abs_grad` 1.0, same batch and starting weights):

| arm | final loss | vs plain |
|---|---|---|
| plain (`lr=0.05`) | 4.1121 | — |
| `outer_lr=0.02`, mom 0.9 | 3.6876 | -0.4244 |
| `outer_lr=0.05`, mom 0.9 | 3.2820 | -0.8300 |
| `outer_lr=0.1`, mom 0.9 | 3.1070 | -1.0050 |
| `outer_lr=0.7`, mom 0.9 | **0.5031** | **-3.6090** |

Three honest limits: this is one easy task (identity copying) on a 1M-parameter
model, so it says the outer step size is badly mis-scaled today, not that 0.7 is
optimal at 1B; the monotone trend means the sweep has not found the top; and the
declared norm is a claim by the *client*, so a coordinator that trusts it can be
pushed around by one client claiming a large norm — `max_abs_grad` is the only cap.

## Averaging draws inside a round

DUST keeps no optimizer state, so *k* draws at the **same** weights are just a bigger
Monte-Carlo batch: `--draws k` averages them client-side and uploads one estimate. The
payload is unchanged (one estimate either way) and there is no local-drift failure mode,
which is why it is a different knob from `--local-steps` (that one moves the model
between estimates and trades staleness for progress).

Gradient alignment against the backprop gradient, 10m preset, population 32, 2x16 tokens,
induction task (`tools/bench_draws.py --mode cosine`):

| draws | cosine | seconds |
|---|---|---|
| 1 | 0.1695 | 4.4 |
| 2 | 0.2398 | 9.0 |
| 4 | 0.3290 | 20.8 |
| 8 | **0.4393** | 36.4 |

And the loss effect at the tiny preset, 20 identity rounds, `ngd1`, outer optimizer on
(`--outer-lr 0.7 --outer-momentum 0.9`):

| draws | bytes/round | final loss |
|---|---|---|
| 1 | 126,900 | 1.4470 |
| 2 | 127,918 | 1.0193 |
| 4 | 130,315 | **0.4541** |

Cost is linear in draws (the client does k times the work per round) and the payload grows
slightly because a less noisy estimate is less compressible. The honest limit: this buys
*Monte-Carlo* variance only. It cannot fix the estimator's bias, and it does nothing about
the sigma/activation-scale mismatch behind the alignment numbers above.

## Measured (CPU, tiny preset)

- `tests`: 119 passed in ~14 s (`protocol` 22, `coordinator` 18, `app` 17,
  `e2e` 4, `feedback` 11, `model` 16, `dust` estimator/training 29).
- DUST gradient vs backprop gradient on the same batch (cosine, seed 0):
  **0.89** at population 64 on 1-layer d32; **0.55** on a 2-layer d64 model with
  per-layer draws (`mode="layer"`); **0.02** on the same 2-layer model when every
  layer is jittered in one pass — interference control is the difference between
  a usable estimate and noise.
- One step at lr 1e-3 from the same weights: loss 0.991684 -> backprop
  0.982955, DUST (population 64) 0.983358, so the sign and scale are right to
  ~4e-4.
- End to end: two clients -> one round -> weights move -> new digest served ->
  the next round verifies against it.

## Measured on gfx1150 (1B plain transformer, aptai2)

`1b` = 1,015,085,056 parameters, bf16, ROCm 7.2 / torch 2.14.1+rocm7.2, 30.48 GB
shared RAM. No code change was needed to run on ROCm: torch reports
`device.type == "cuda"`, so the bf16 paths light up unchanged.

| measurement | result |
|---|---|
| weights at bf16 | 1.89 GB |
| forward, seq 8 | 0.293 s, peak **1.92 GB** allocated / **1.96 GB** reserved |
| DUST step, seq 32, bs 1, population 2 | **8.18 s**, 153 forwards, **6.06 GB** peak reserved |
| fp32 gradient buffer for the whole model | 3.78 GB |
| evidence throughput of that step | 64 tokens = **7.8 tok/s** |

So a 1B plain transformer takes a real DUST step on a 30 GB APU with ~24 GB
headroom -- the claim backprop cannot make there (AdamW fp32 m/v alone is 8 GB,
before the graph). The cost driver is visible in the same numbers: per-layer
passes turn one token of evidence into ~2.4 forwards of a 1B model (153 / 64), so
throughput is dominated by the pass structure, not the arithmetic.

## Measured on a 16 GB consumer card (1B, CUDA)

Same `1b` preset (1,015,085,056 parameters, bf16) on an RTX 4060 Ti 16 GB,
torch 2.6.0+cu124, for the cross-vendor comparison:

| measurement | RTX 4060 Ti | Radeon 890M |
|---|---|---|
| weights at bf16 | 2.03 GB | 1.89 GB |
| forward, seq 8 | 0.27 s, 2.07 GB reserved | 0.29 s, 1.96 GB reserved |
| DUST step, seq 32, bs 1, population 2 | **2.08 s**, 153 forwards, **6.47 GB** | 8.18 s, 153 forwards, 6.06 GB |
| evidence throughput | **30.7 tok/s** (73.5 forwards/s) | 7.8 tok/s (18.7 forwards/s) |

So gate 1 of the pitch holds on consumer hardware with room to spare: a
literal-1B DUST step fits in **6.5 GB**, leaving ~9 GB of the card for a bigger
population or longer sequences. The same shapes run 3.9x faster on CUDA than on
the 8-CU APU, which is the first hardware-class comparison of this method we have.

## Measured on Apple silicon (1B, MPS)

Same `1b` preset on an Apple M5 Max (48 GB unified memory), macOS 26, torch
2.15.0, `--device mps`. Reproduce with:

```bash
uv venv --python 3.12 .venv            # system python3 on macOS is 3.9; need >= 3.10
uv pip install --python .venv/bin/python -e '.[dev]'
.venv/bin/python tools/bench_device.py --preset 1b --device mps --repeat 3
```

| measurement | M5 Max (MPS) |
|---|---|
| weights at bf16 | 1.89 GB |
| forward, seq 8 (warm, median of 3) | 0.006 s, 2.01 GB driver-allocated |
| DUST step, seq 32, bs 1, population 2 | **1.31 s** (noise on CPU) / **1.16 s** (noise on device), 153 forwards |
| evidence throughput | **48.9 tok/s** (116.9 forwards/s) / 55.2 tok/s (132.0 forwards/s) |
| fp32 gradient buffer for the whole model | 3.78 GB |

Caveats: torch has no peak-memory counter for MPS, so `tools/bench_device.py`
reports `torch.mps.driver_allocated_memory()` (includes the allocator cache;
~10 GB after warmup + 3 steps), which is not comparable to CUDA's peak
reserved. The forward time is a warm median; the CUDA/ROCm rows above appear to
include first-call cost. In fp32 with CPU-drawn noise the MPS estimate matches
CPU to cosine 1.000 (`tests/test_mps.py`); bf16 differs from fp32 by rounding
on every backend.

`crowd_v15/accel.py` picks the device (`auto` = CUDA/ROCm, then MPS, then CPU)
and wraps synchronize / memory / autocast per backend; the client takes
`--device mps` or `--device auto`.

End to end over HTTP on the same machine (uvicorn coordinator, two concurrent
clients per round, `10m` preset, induction, seq 64, batch 4, population 32,
draw-chunk 8, NGD1R uploads, 6 rounds):

| | MPS (bf16) | CPU (fp32) |
|---|---|---|
| client step | ~2.1 s | ~13.1 s |
| evidence per client | ~4,000 tok/s | ~625 tok/s |
| round wall time | ~3.1 s | ~13.9 s |
| upload per submission | 1.70 MB (dense would be 14.69 MB) | same |

That run uses the coordinator's default `lr=1e-3` with no outer optimizer, so
it measures plumbing and throughput, not learning. With `lr=0.05`,
`outer_lr=0.7`, momentum 0.9, `tiny` identity goes 4.84 -> 2.76 held-out loss in
30 rounds: 6.6 s on MPS vs 259 s on CPU.

## How a round scales submissions

The absolute scale of an ES estimate is an artefact of `sigma` and the token
count: a 64-token draw at `sigma = 1e-3` came back with a norm of **1.5e5** on the
1B model, so any fixed bound on `|g|` rejects every honest client.

So the coordinator normalises each submission to a unit direction at accept time,
keeps the declared norm in the round record, and applies the *mean declared norm*,
capped by `max_abs_grad` (and by `grad_clip` when set). Direction is what an ES
step needs; magnitude is the coordinator's learning rate times a bounded,
auditable number. A non-finite estimate never reaches the coordinator — the wire
format rejects it first.

## Honest limits


- The estimator is validated on tiny CPU models (cosine 0.89 at 1 layer, 0.55 at
  two with per-layer draws); the 1B numbers below are *mechanism* numbers from a
  two-draw probe, not a quality claim. Nothing here proves the
  paper's claim that larger models are *more* population-efficient at 1B; that
  needs the GPU box and the falsifiable gates (does a 1B-shaped step fit; does
  gradient alignment hold as size and tokens grow; how many draws to match
  backprop at fixed tokens — measured as time-to-target-loss, not FLOPs).
- The DUST path here draws each population member from a seed derived from
  `(seed, step, shape)` on a CPU generator, so a coordinator can in principle
  reproduce a member's noise. It does not yet re-score a submission; that is the
  audit primitive left to wire up (the v1 seed-derivation and signed-receipt work
  is the template).
- Weight serving is one safetensors file. A real 1B model needs ranged or sharded
  downloads; the manifest already reports `bytes`, `names`, `sizes` and `shapes`
  so a shard layer can be added without changing the client's checks.
- The task sources are identity/induction/copy/random. Identity is a learning
  test, not a pretraining corpus: real runs need real tokens (`--task copy
  --data chunks/chunk_0000.bin --data-dtype uint32`).
- No UI, no leaderboard, no reputation. The three-user verification pool of v1 is
  not reimplemented here.
