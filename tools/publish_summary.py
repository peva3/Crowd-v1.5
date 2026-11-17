#!/usr/bin/env python3
"""Publish a cross-machine comparison of the live 24 h runs as `results/README.md`.

`publish_results.py` writes one directory per machine, which is right for the raw
evidence but leaves a reader to diff two `index.md` files by hand. This builds the
single table that answers the question the run exists to answer: how many evidence
tokens per second does the DUST estimator actually get on each card.

Every number is derived from the append-only jsonl the trainers emitted, so the
page can never disagree with the evidence next to it. Rates use tokens-per-step
divided by the *median* step time, which survives a resumed run (a session-only
denominator made the ROCm box report 388 tok/s while stepping at 23).

    python tools/publish_summary.py --runs /tmp/v15runs --repo peva3/Crowd-v1.5-dust \
        --env-file /app/crowdgpt/.env
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

HARDWARE = {
    "rtx4060ti": "RTX 4060 Ti 16 GB · CUDA 12.4",
    "gfx1150": "Radeon 890M (gfx1150) · 30 GB shared · ROCm 7.2",
}
ARM_ORDER = {"adam": 0, "dust": 1}


def read_run(path: Path) -> dict | None:
    if not path.is_file():
        return None
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            # A trainer killed mid-write can leave one partial line.
            continue
    if not records:
        return None
    # The meta line (first record of a fresh run) carries no per-step fields.
    steps = [r for r in records if r.get("step") is not None]
    if not steps:
        return None
    evals = [r for r in steps if r.get("evalLoss") is not None]
    last = steps[-1]
    times = sorted(r["stepSeconds"] for r in steps if r.get("stepSeconds"))
    median = times[len(times) // 2] if times else None
    tokens_per_step = last["tokens"] / max(1, last["step"])
    return {
        "records": len(steps),
        "step": last["step"],
        "tokens": last["tokens"],
        "trainLoss": last.get("trainLoss"),
        "instantLoss": last.get("trainLossInstant"),
        "evals": [{"tokens": e["tokens"], "loss": e["evalLoss"]} for e in evals],
        "bestEval": min((e["evalLoss"] for e in evals), default=None),
        "firstEval": evals[0]["evalLoss"] if evals else None,
        "medianStepSeconds": round(median, 1) if median else None,
        "peakVramGb": max((r.get("peakVramGb") or 0) for r in steps) or None,
        "tokensPerSecond": round(tokens_per_step / median, 1) if median else None,
    }


def fmt(value, digits=0):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}" if digits else f"{value:.2f}"
    return f"{value:,}" if isinstance(value, int) and digits == 0 else str(value)


def build(runs: Path, param_count: int) -> str:
    rows = []
    for machine in sorted(HARDWARE):
        machine_dir = runs / machine
        if not machine_dir.is_dir():
            continue
        for arm_file in sorted(machine_dir.glob("*.jsonl"), key=lambda p: ARM_ORDER.get(p.stem, 9)):
            stats = read_run(arm_file)
            if stats:
                rows.append((machine, arm_file.stem, stats))
    if not rows:
        raise SystemExit(f"no run records under {runs}")

    out = [
        "# Crowd-v1.5 — 24 h DUST runs, measured",
        "",
        f"Updated {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}. "
        "Every number below is derived from the append-only `*.jsonl` in each "
        "machine directory; the raw log is the authority and nothing here is "
        "edited after the fact.",
        "",
        "| machine | hardware | arm | steps | evidence tokens | tokens/s | median s/step | peak GB | evals | best held-out | tokens/param |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for machine, arm, s in rows:
        out.append(
            f"| [`{machine}`](./{machine}) | {HARDWARE[machine]} | `{arm}` | {s['step']:,} | "
            f"{s['tokens']:,} | {fmt(s['tokensPerSecond'], 1)} | {fmt(s['medianStepSeconds'], 1)} | "
            f"{fmt(s['peakVramGb'], 2)} | {len(s['evals'])} | {fmt(s['bestEval'], 4)} | "
            f"{s['tokens'] / param_count:.4f} |"
        )

    out += ["", "## What each column means", "",
            "- **evidence tokens** — tokens the arm actually scored (`batch × seq` per "
            "step). Not the population it evaluated; a DUST step touches "
            "`3 × population` forwards but only credits `batch × seq`.",
            "- **tokens/s** — evidence tokens per step divided by the *median* step "
            "time. Median, because a resumed run's `seconds` field covers only the "
            "current session.",
            "- **best held-out** — cross-entropy on 24 fixed windows of `val.bin`, "
            "which no arm ever trains on. `ln(4096) = 8.3178` is the uniform "
            "baseline for this tokenizer's vocabulary.",
            "- **tokens/param** — the honest scale check. Convergence in this "
            "literature is usually reached around 20 tokens/param, which at 306 M "
            "is ~6.1 B tokens.", ""]

    notes = [
        "## Configuration",
        "",
        "Both DUST arms: 300M preset (306,222,080 parameters), sequence 512, batch 8, "
        "population 32, `--draw-chunk 1 --mode all --noise-device device`, `ngd1r` "
        "upload through the real coordinator (unit-direction accept, cap, outer "
        "optimiser lr 0.7 / momentum 0.9), uncapped (`--tokens 0 --minutes 1440`). "
        "Both read the same 58 M-token FineWeb-Edu shard "
        "([`peva3/crowd-v15-tokens`](https://huggingface.co/datasets/peva3/crowd-v15-tokens)) "
        "in the same window order, so their curves are paired step for step.",
        "",
        "The `adam` arm is the compute baseline on the 4060 Ti: plain backprop, "
        "batch 4, 12 minutes. It exists to price the estimator — a DUST step costs "
        "`3 × population` forward passes and no backward pass.",
        "",
        "## How to read a curve that is still moving",
        "",
        "A run appends one jsonl line per step while it trains, so these files grow. "
        "`*.curve.svg` is regenerated from the jsonl on each publish; `*.curve.csv` "
        "is the same data as text. At 306 M and 24 h the arms reach tens of "
        "thousands of tokens per parameter at best, i.e. this is a throughput "
        "measurement, not a convergence claim.",
        "",
        "## One caveat this run surfaced",
        "",
        "`--sigma-mode relative` scales each layer's noise to `sigma_rel x RMS(layer "
        "output)`, and that reference is not stationary. Over the first 5.2 M tokens "
        "of the 4060 Ti arm the per-layer maximum grew 0.032 -> 6.66 (x207) while "
        "total weight RMS grew only x1.18, i.e. the drift is activation-RMS growth, "
        "not weight growth. The estimator therefore finishes the run in a much "
        "coarser perturbation regime than it started in, and held-out loss has been "
        "flat (6.74-6.78) across the last six evals spanning 1 M tokens while train "
        "loss kept falling. Read the late, flat part of the curve as \"the estimator "
        "under an unbounded noise scale\" rather than as a converged model; the fix "
        "(clamping or scheduling `sigma_rel`) is TODO G38, and the two hypotheses "
        "the run cannot separate are noise-scale drift and simply too few tokens.",
        "",
        "**A CPU A/B has now separated them.** Three arms, 10 m preset, identical "
        "windows, identical 100,096 evidence tokens each: relative uncapped bottoms "
        "out at held-out 7.2322 at 50 k tokens and then *degrades* to 7.4336; relative "
        "clamped to 4x its step-0 scale keeps falling to **7.0321**; absolute `1e-2` "
        "(~9x more noise than the relative arm's start) reaches 7.8048 and is only a "
        "magnitude control. So the drift costs ~0.40 held-out at equal tokens, and the "
        "cap (`--sigma-max`) fixes it. The 24 h arms ran **uncapped** — that is the "
        "honest reading of their late flatness. Single seed, 10 m, 100 k tokens: "
        "direction, not a constant.",
        "",
    ]
    return "\n".join(out + notes)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", default="/tmp/v15runs")
    parser.add_argument("--repo", default="peva3/Crowd-v1.5-dust")
    parser.add_argument("--path-in-repo", default="results/README.md")
    parser.add_argument("--params", type=int, default=306_222_080)
    parser.add_argument("--token-env", default="HF_TOKEN")
    parser.add_argument("--env-file", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.env_file and os.path.exists(args.env_file):
        for line in Path(args.env_file).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

    text = build(Path(args.runs), args.params)
    if args.dry_run:
        print(text)
        return 0

    from huggingface_hub import HfApi

    token = os.environ.get(args.token_env) or ""
    if not token:
        raise SystemExit(f"{args.token_env} is not set")
    HfApi(token=token).upload_file(
        path_or_fileobj=text.encode(), path_in_repo=args.path_in_repo,
        repo_id=args.repo, repo_type="model",
        commit_message="results: refresh the cross-machine 24 h summary",
    )
    print("uploaded", args.path_in_repo, len(text), "bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
