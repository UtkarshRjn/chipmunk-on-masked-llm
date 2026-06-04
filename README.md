# mdm-chipmunk

Mask-Adaptive Sparse Kernels for Masked Diffusion Language Models — port of the
[Chipmunk](https://github.com/sandyresearch/chipmunk) kernel toolkit from image/video
DiTs to MDM language models such as LLaDA-8B.

Full research direction in [`research_proposal.md`](research_proposal.md).

## Status

**Month 1: baseline reproduction.** Standing up LLaDA-8B, FastDLLM, dKV-Cache, and
DualDiffusion under one harness so subsequent kernel work has a defensible ceiling
to compare against.

- Live results table: [`results/summary/month1_baselines.md`](results/summary/month1_baselines.md)
- Per-sample logs (W&B): [`utranjan-uc-san-diego/mdm-chipmunk`](https://wandb.ai/utranjan-uc-san-diego/mdm-chipmunk) (set `WANDB_PROJECT=mdm-chipmunk` in your env to stream)

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
- `chipmunk/`, `fastdllm/`, `dkv_cache/` — reference clones of the upstream
  implementations. Gitignored; populated by `scripts/clone_baselines.sh`.

## Running on H100

The CPU pipeline only exercises shapes and the harness. Real speedup numbers
require an H100 (or A100 fallback). On a fresh GPU host:

```bash
# 1. System prep
git clone https://github.com/your-fork/chipmunk-on-masked-llm.git
cd chipmunk-on-masked-llm
bash scripts/clone_baselines.sh          # pulls chipmunk/, fastdllm/, dkv_cache/

# 2. Python env
python3.13 -m venv .venv && source .venv/bin/activate
pip install -e '.[gpu,dev]'
pip install -e ./chipmunk                # only if you'll run the kernel branch later

# 3. Model weights
bash scripts/download_llada.sh           # ~16 GB to ~/.cache/huggingface

# 4. Baseline reproduction (Month 1 milestone gate)
mdm-bench run --method dense       --model llada-8b-instruct --task gsm8k     --num-samples 100 --out results/llada_dense_gsm8k.jsonl
mdm-bench run --method fastdllm    --model llada-8b-instruct --task gsm8k     --num-samples 100 --out results/llada_fastdllm_gsm8k.jsonl
mdm-bench run --method dkv_cache   --model llada-8b-instruct --task gsm8k     --num-samples 100 --out results/llada_dkv_gsm8k.jsonl
mdm-bench run --method dualdiffusion --model llada-8b-instruct --task gsm8k   --num-samples 100 --out results/llada_dd_gsm8k.jsonl

# 5. Validate against published numbers (within ±5%)
for f in results/llada_*_gsm8k.jsonl; do mdm-bench compare --results $f; done
```

Targets and tolerances live in `configs/reproduction_targets.yaml`. Host-specific
overrides (HF cache dir, batch size, num_workers) live in `configs/hosts/*.yaml`.

## Results

Summary table (committed, one row per `method × model × task`):
[`results/summary/month1_baselines.md`](results/summary/month1_baselines.md).

Per-sample telemetry (latency, NFE, per-step times, peak memory) lives on W&B —
the harness streams to the project named by `WANDB_PROJECT` whenever that env
var is set:

```bash
pip install -e '.[wandb]'
wandb login
export WANDB_PROJECT=mdm-chipmunk     # plus WANDB_ENTITY=<your-org> if applicable
mdm-bench run --method dense --model llada-8b-instruct --task gsm8k \
    --num-samples 100 --out results/llada_dense_gsm8k.jsonl
```

Without `WANDB_PROJECT` the harness is a no-op for W&B — JSONL files in
`results/` remain the ground truth. After a run lands, paste its summary row
into `results/summary/month1_baselines.md` with the short SHA from
`git rev-parse --short HEAD` and a link to the W&B run.

## Layout

(see project root)

