# Plan — Start coding the Chipmunk-on-MDM research proposal

## Context

`research_proposal.md` lays out a 6-month research project: port Chipmunk's sparse attention/MLP kernel toolkit (currently for image/video DiTs) to Masked Diffusion Language Models like LLaDA-8B, add a magnitude-based active-set selection, and stack with FastDLLM / DualDiffusion. The target paper is MLSys/ICLR/NeurIPS with Dan Fu (Chipmunk senior author) as advisor.

**Current state (just surveyed):**
- Project directory `chipmunk-on-masked-llm/` is greenfield except for `research_proposal.md` and a freshly cloned `chipmunk/` reference repo.
- Chipmunk exposes a clean Python API (`SparseDiffAttn`, `SparseDiffMlp` from `chipmunk.modules`) over ThunderKittens CUDA kernels (Hopper) with Triton fallbacks.
- No prior scaffolding, no `pyproject.toml`, no Python files. Git repo on `main`, clean.

**User constraints (answered just now):**
- **Hardware:** CPU laptop only for now. No GPU access yet. Need code that runs as smoke tests on CPU/MPS, and is deploy-ready for an H100 host later.
- **Starting point:** Month 1 of the proposal — reproduce baselines (LLaDA-8B dense, FastDLLM, dKV-Cache, DualDiffusion). Defer kernel work (Months 2-3) until after baselines establish the speedup ceiling.
- **Layout:** Independent package (`mdm_chipmunk`) that pip-installs `chipmunk` as a dependency rather than forking it.

**Why this scope first:** §6 of the proposal flags the biggest risk as "FastDLLM already gets 27.6× — kernel work on top only buys 1.05-1.15×." Mitigation is to characterize first and pivot framing if needed. Month 1 is exactly that characterization. CPU-only forces us to invest in solid scaffolding and abstractions before there's any temptation to hack a one-shot benchmark script.

---

## Repository layout (new)

```
chipmunk-on-masked-llm/
├── chipmunk/                       (cloned reference — DO NOT EDIT)
├── research_proposal.md
├── pyproject.toml                  ← new (defines mdm_chipmunk package)
├── README.md                       ← new (short, points to proposal)
├── .gitignore                      ← new (Python, .venv, model cache, results/)
├── configs/
│   ├── models/
│   │   ├── llada_8b_instruct.yaml
│   │   ├── llada_1_5.yaml
│   │   └── mdlm_170m.yaml          (CPU smoke-test target)
│   ├── tasks/
│   │   ├── gsm8k.yaml              (5-shot, gen 256/512/1024)
│   │   ├── humaneval.yaml
│   │   ├── mmlu.yaml
│   │   └── ruler_8k.yaml
│   └── methods/
│       ├── dense.yaml
│       ├── fastdllm.yaml
│       ├── dkv_cache.yaml
│       └── dualdiffusion.yaml
├── src/mdm_chipmunk/
│   ├── __init__.py
│   ├── models/
│   │   ├── base.py                 (MDMModel abstract interface)
│   │   ├── llada.py                (HF loader for GSAI-ML/LLaDA-8B-Instruct)
│   │   └── mdlm.py                 (kuleshov-group/mdlm — CPU-runnable)
│   ├── methods/
│   │   ├── base.py                 (InferenceMethod interface: .generate(model, prompt, cfg))
│   │   ├── dense.py                (vanilla MDM denoising loop)
│   │   ├── fastdllm.py             (port from Wu et al. 2025 reference impl)
│   │   ├── dkv_cache.py            (port from Ma et al. 2025 reference impl)
│   │   └── dualdiffusion.py        (port from Goyal et al. 2026)
│   ├── eval/
│   │   ├── harness.py              (run method × task, log to JSONL)
│   │   ├── tasks/
│   │   │   ├── gsm8k.py
│   │   │   ├── humaneval.py
│   │   │   └── mmlu.py
│   │   └── metrics.py              (accuracy, latency, tok/s, peak VRAM)
│   ├── utils/
│   │   ├── device.py               (picks cpu/mps/cuda + dtype defaults)
│   │   ├── logging.py              (structured JSONL + rich console)
│   │   └── profiling.py            (torch profiler wrapper, timing context manager)
│   └── cli.py                      (Typer entry: `mdm-bench run --method X --task Y`)
├── scripts/
│   ├── download_llada.sh           (uses huggingface-cli, prompts for token)
│   └── smoke_test_cpu.py           (MDLM-170M × 10 GSM8K samples × dense — end-to-end on CPU)
├── tests/
│   ├── test_device.py
│   ├── test_models_smoke.py        (loads MDLM, runs 1 step, checks output shape)
│   ├── test_dense_method.py        (verifies denoising loop on toy model)
│   └── test_harness.py             (eval harness writes valid JSONL)
└── results/                        (gitignored — JSONL run logs land here)
```

---

## Implementation order

Tasks are sequenced so something works end-to-end on CPU before any baseline porting begins. GPU-only steps are flagged.

### Phase A — Scaffolding (CPU-only, ~half a day)

1. **`pyproject.toml`** — package `mdm_chipmunk`, deps: `torch`, `transformers`, `accelerate`, `datasets`, `typer`, `pyyaml`, `rich`, `pytest`. Pin `chipmunk` from local path (`-e ../chipmunk` dev install; switch to GitHub URL for non-laptop hosts). Add `[project.scripts] mdm-bench = "mdm_chipmunk.cli:app"`.
2. **`.gitignore`** — Python defaults + `.venv/`, `results/`, `~/.cache/huggingface/`, `*.safetensors`.
3. **`README.md`** — one paragraph + pointer to `research_proposal.md` + quickstart for smoke test.
4. **`utils/device.py`** — `get_device()` returns `"cuda"` > `"mps"` > `"cpu"`; `get_dtype(device)` returns `bf16` on cuda, `fp32` elsewhere. Single source of truth — no other module hardcodes device.
5. **`utils/logging.py`** — `get_logger(name)` returns rich-console logger; `JsonlWriter(path)` appends one JSON line per result row.

### Phase B — Model interface + smoke target (CPU works)

6. **`models/base.py`** — abstract `MDMModel`: `forward_logits(x_t, attention_mask) -> logits`, `tokenizer`, `mask_token_id`, `seq_len`, `num_layers`. Mirrors how Chipmunk's `SparseDiffAttn` plugs into a host model — keeps the option of wrapping these later.
7. **`models/mdlm.py`** — load `kuleshov-group/mdlm-owt` (170M, runs on CPU in fp32). This is the smoke-test workhorse for Phase A–E development.
8. **`models/llada.py`** — load `GSAI-ML/LLaDA-8B-Instruct`. Include `lazy=True` flag that skips actual weight download (just instantiates config + tokenizer) so CI/tests on laptop don't pull 16 GB. Real load only on GPU host.
9. **`tests/test_models_smoke.py`** — instantiate MDLM, push 1 masked token through `forward_logits`, assert output shape `(B, L, V)`. Runs in CI on CPU.

### Phase C — Dense baseline method + harness (CPU smoke works)

10. **`methods/base.py`** — `InferenceMethod` ABC: `generate(model, prompt_ids, gen_length, num_steps, seed) -> (output_ids, telemetry)`. Telemetry dict captures per-step time, total time, peak memory.
11. **`methods/dense.py`** — vanilla MDM denoising: init `x_T = [MASK]^L`, loop T steps doing `forward_logits` → confidence-weighted unmask → repeat. No caching. This is the speed floor and quality ceiling all other methods compare against.
12. **`eval/tasks/gsm8k.py`** — load GSM8K via `datasets`, format as MDM prompt, parse predicted number with regex. Just 10 samples for smoke; full eval is a CLI flag.
13. **`eval/metrics.py`** — `accuracy(predictions, gold)`, `mean_latency(telemetry)`, `tokens_per_sec(telemetry)`, peak memory delta.
14. **`eval/harness.py`** — `run(method, model, task, num_samples) -> results_df`. Writes one JSONL row per sample with `{method, task, model, sample_idx, correct, latency_s, tokens_s, telemetry_json}`.
15. **`cli.py`** — `mdm-bench run --method dense --model mdlm-170m --task gsm8k --num-samples 10 --out results/smoke.jsonl`. Loads YAML configs by name, dispatches to harness.
16. **`scripts/smoke_test_cpu.py`** — invokes the CLI with MDLM × dense × 10 GSM8K samples. Expected: completes in ~10-30 min on a laptop, accuracy ≈ 0 (MDLM-170M is too small for GSM8K — that's fine; we're testing the pipeline, not the model).
17. **`tests/test_dense_method.py`, `tests/test_harness.py`** — pytest verifies the denoising loop terminates with no mask tokens remaining, harness writes valid JSONL.

**Checkpoint:** at this point, `python -m mdm_chipmunk.cli run --method dense --model mdlm-170m --task gsm8k --num-samples 5` completes end-to-end on a CPU laptop with a results file. Everything below is layered on top.

### Phase D — Baseline method ports (write on CPU, validate on GPU later)

18. **`methods/fastdllm.py`** — locate Wu et al. 2025 reference implementation (their public repo, likely `NVlabs/Fast-dLLM` or similar — verify URL during work). Port the two key pieces: (a) block-wise approximate KV cache, (b) confidence-aware parallel decoding with the parallel-decoding theorem's threshold. Wire into `InferenceMethod` interface. Shape-check on MDLM. Real benchmark requires GPU.
19. **`methods/dkv_cache.py`** — port from Ma et al. 2025 ref impl. Both `dKV-Cache-Decode` (recency-based) and `dKV-Cache-Greedy` variants, since the proposal's magnitude-based selection (Contribution 2) compares directly against these. Carefully port the `concat_reorder` op — that's the algorithmic analogue of Chipmunk's gather-pack pattern.
20. **`methods/dualdiffusion.py`** — port from Goyal et al. 2026 ref impl. Uses FastDLLM as drafter, LLaDA as verifier. Depends on (18). If the paper hasn't released code yet (it's listed as 2026), stub the interface and skip the port for now.
21. **YAML configs in `configs/methods/`** for each — hyperparameters surfaced explicitly so we can ablate without code changes.
22. **`tests/test_methods.py`** — for each method, run on MDLM-170M with 1 GSM8K sample; assert termination + valid output. Quality is unspecified (these are baseline-correctness tests, not paper-correctness tests).

### Phase E — GPU-host readiness (no execution on laptop)

23. **`scripts/download_llada.sh`** — `huggingface-cli download GSAI-ML/LLaDA-8B-Instruct` to a configurable cache dir. Runs on GPU host before benchmarks.
24. **`configs/host_h100.yaml`** — host-specific overrides (cache dir, num_workers, batch size). Lets us keep model/task/method configs hardware-agnostic.
25. **README "Running on H100"** section — exact commands to reproduce each baseline number from the proposal's Table in §2.2.
26. **Sanity-run plan, written down** — for each `(method, model, task)` combination, the target metric from the original paper and tolerance (±5%). Lives in `configs/reproduction_targets.yaml`. The Month 1 success criterion in the proposal is "confirm published numbers within 5%" — this file makes that machine-checkable.

---

## Critical files to reuse / reference (no need to reinvent)

- `chipmunk/src/chipmunk/modules/attn.py` and `mlp.py` — the integration pattern (`SparseDiffAttn(layer_num, layer_counter)` that wraps a host model's attention) is the template our future Month 2-3 kernel work will mirror. Read these before writing `models/base.py` so the `MDMModel` interface stays compatible.
- `chipmunk/src/chipmunk/util/config.py` and `chipmunk/util/storage/` — patterns for YAML-driven configs and cache storage. Worth copying the style rather than designing fresh.
- `chipmunk/examples/hunyuan/chipmunk-config.yml`, `flux/chipmunk-config.yml` — reference YAML shape for sparsity / scheduling configs. Our `configs/methods/*.yaml` should look familiar to anyone who already uses Chipmunk.

---

## Verification

End-to-end CPU verification (achievable on the laptop today):
```bash
# After Phase C
pip install -e .
python scripts/smoke_test_cpu.py
# Expect: results/smoke.jsonl with 5-10 rows, each with valid JSON,
# `correct` field present, `latency_s` > 0. Accuracy may be 0 — fine.

pytest tests/ -v
# Expect: all tests green on CPU, no GPU required.
```

After Phase D (still CPU — shape/interface only):
```bash
for m in fastdllm dkv_cache; do
  python -m mdm_chipmunk.cli run --method $m --model mdlm-170m \
    --task gsm8k --num-samples 2 --out results/smoke_$m.jsonl
done
# Expect: each method runs without crashing, produces JSONL.
```

After Phase E, on GPU host:
```bash
bash scripts/download_llada.sh
python -m mdm_chipmunk.cli run --method dense --model llada-8b-instruct \
  --task gsm8k --num-samples 100 --out results/llada_dense_gsm8k.jsonl
python -m mdm_chipmunk.cli compare --baseline results/llada_dense_gsm8k.jsonl \
  --target configs/reproduction_targets.yaml
# Expect: dense LLaDA-8B matches published GSM8K number within 5%.
# Then repeat for fastdllm, dkv_cache, dualdiffusion.
```

The reproduction-targets check is the Month 1 milestone gate. Until it passes for all four methods on at least GSM8K + HumanEval, we don't start kernel work in Month 2.

---

## Out of scope for this round (deferred to later phases)

- ThunderKittens kernel writing (Month 2-3)
- `cp.async` scattered gather, `wgmma` packed GEMM (Month 2-3)
- Magnitude-based active set selection (Month 3-4; pure-Python prototype is cheap and could land earlier)
- Theoretical error bound (Month 5)
- Composition with DualDiffusion as spec-decoding (Month 5)
- Long-context RULER 8K-32K benchmark (Month 5; requires GPU with enough memory)

These all live in the proposal and the repo layout leaves clean slots for them (e.g., a future `src/mdm_chipmunk/kernels/` and `src/mdm_chipmunk/methods/chipmunk_mdm.py`) without restructuring.
