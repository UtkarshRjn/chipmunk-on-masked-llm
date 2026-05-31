# mdm-chipmunk

Mask-Adaptive Sparse Kernels for Masked Diffusion Language Models — port of the
[Chipmunk](https://github.com/sandyresearch/chipmunk) kernel toolkit from image/video
DiTs to MDM language models such as LLaDA-8B.

Full research direction in [`research_proposal.md`](research_proposal.md).

## Status

**Month 1: baseline reproduction.** Standing up LLaDA-8B, FastDLLM, dKV-Cache, and
DualDiffusion under one harness so subsequent kernel work has a defensible ceiling
to compare against.

## Quickstart (CPU, smoke test)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install -e ./chipmunk  # optional — only needed for GPU kernel work later

python scripts/smoke_test_cpu.py
# → writes results/smoke.jsonl with a small MDLM-170M dense run on GSM8K.
```

Expect accuracy ≈ 0 on the smoke run — MDLM-170M is too small for GSM8K. The
goal is to verify the pipeline end-to-end before deploying to an H100 host.

## Layout

- `src/mdm_chipmunk/models/` — `MDMModel` interface + LLaDA / MDLM loaders.
- `src/mdm_chipmunk/methods/` — inference methods (`dense`, `fastdllm`, `dkv_cache`, …).
- `src/mdm_chipmunk/eval/` — task definitions, metrics, harness.
- `configs/` — YAML configs for models, tasks, methods (one file each, named).
- `scripts/` — entry-point shell scripts and smoke tests.
- `chipmunk/` — reference clone of upstream Chipmunk (read-only, do not edit).

## Running on H100

See `configs/host_h100.yaml` and the section at the end of this README once
Phase E lands.
