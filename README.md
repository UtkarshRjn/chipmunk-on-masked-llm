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

## Quickstart

### CPU / synthetic model (no GPU needed)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

python -m pytest tests/          # 26 passed, 1 skipped (network test)
python scripts/smoke_test_cpu.py # synthetic model, 3 samples — verifies harness wiring
```

### GPU setup (RTX 3090 / A5000 / A100 — CUDA 12.x)

```bash
# 1. Create a conda env (adjust prefix to taste)
conda create --prefix ~/envs/chipmunk python=3.11 -y
conda activate ~/envs/chipmunk

# 2. Install PyTorch — match the cu build to your driver
#    Driver 535 / CUDA 12.2  →  cu121
#    Driver 550 / CUDA 12.4  →  cu124
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Install project + extras
pip install einops
pip install flash_attn --no-build-isolation   # ~15–20 min compile
pip install -e ".[dev,wandb]"

# 4. Verify
python -c "import torch; print(torch.cuda.get_device_name(0))"
python -m pytest tests/

# 5. GPU smoke test — MDLM-170M × dense × GSM8K (5 samples, ~10 s on A5000)
MDM_DEVICE=cuda python scripts/smoke_test_gpu.py
```

Expect accuracy ≈ 0 — MDLM-170M is OWT-trained, not math-tuned. The goal is
end-to-end pipeline validation before running LLaDA-8B.

**Environment variables for `smoke_test_gpu.py`:**

| Variable | Default | Options |
|----------|---------|---------|
| `MODEL` | `mdlm_170m` | `llada_8b_instruct` |
| `METHOD` | `dense` | `fastdllm`, `dkv_cache` |
| `TASK` | `gsm8k` | `synthetic` |
| `NUM_SAMPLES` | `5` | any int |
| `NUM_STEPS` | `16` | any int |

```bash
# Example: test fastdllm on MDLM-170M
METHOD=fastdllm MDM_DEVICE=cuda python scripts/smoke_test_gpu.py
```

**Compatibility notes:**
- Works with torch 2.5.x and 2.6.x — `utils/compat.py` patches flash_attn's
  `wrap_triton` shim and the `torch_dtype`/`dtype` rename automatically.
- MDLM-170M always loads in float32 regardless of device capability (required
  by the model's sinusoidal timestep embedder; does not affect generation quality).

## Layout

- `src/mdm_chipmunk/models/` — `MDMModel` interface + LLaDA / MDLM loaders.
- `src/mdm_chipmunk/methods/` — inference methods (`dense`, `fastdllm`, `dkv_cache`, …).
- `src/mdm_chipmunk/eval/` — task definitions, metrics, harness.
- `configs/` — YAML configs for models, tasks, methods (one file each, named).
- `scripts/` — entry-point shell scripts and smoke tests.
- `chipmunk/`, `fastdllm/`, `dkv_cache/` — reference clones of the upstream
  implementations. Gitignored; populated by `scripts/clone_baselines.sh`.

## Running on A100 / H100 (Month-1 milestone gate)

```bash
# 1. Clone and prep
git clone https://github.com/your-fork/chipmunk-on-masked-llm.git
cd chipmunk-on-masked-llm
bash scripts/clone_baselines.sh    # pulls chipmunk/, fastdllm/, dkv_cache/

# 2. Conda env (same as GPU Quickstart above, adjust cu build for the host driver)
conda create --prefix ~/envs/chipmunk python=3.11 -y
conda activate ~/envs/chipmunk
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install einops flash_attn --no-build-isolation
pip install -e ".[dev,wandb]"

# 3. Model weights
bash scripts/download_llada.sh     # ~16 GB to ~/.cache/huggingface

# 4. GPU smoke (validates pipeline on MDLM-170M before the expensive runs)
MDM_DEVICE=cuda python scripts/smoke_test_gpu.py

# 5. Month-1 baseline runs (LLaDA-8B, 100 samples each)
mdm-bench run --method dense         --model llada-8b-instruct --task gsm8k --num-samples 100 --out results/llada_dense_gsm8k.jsonl
mdm-bench run --method fastdllm      --model llada-8b-instruct --task gsm8k --num-samples 100 --out results/llada_fastdllm_gsm8k.jsonl
mdm-bench run --method dkv_cache     --model llada-8b-instruct --task gsm8k --num-samples 100 --out results/llada_dkv_gsm8k.jsonl
mdm-bench run --method dualdiffusion --model llada-8b-instruct --task gsm8k --num-samples 100 --out results/llada_dd_gsm8k.jsonl

# 6. Compare against published targets (±5% tolerance)
for f in results/llada_*_gsm8k.jsonl; do mdm-bench compare --results $f; done
```

Targets live in `configs/reproduction_targets.yaml`.

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


