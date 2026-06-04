# Progress — Month 1 baseline reproduction

**Last updated:** 2026-06-04 · against commit `19e0e30`
**Plan:** [`docs/month1_baselines_plan.md`](month1_baselines_plan.md)
**Results table:** [`results/summary/month1_baselines.md`](../results/summary/month1_baselines.md)

This file tracks what's *implemented*, what's *validated*, and what's *next*.
Update it whenever a phase item moves status.

## At a glance

| Dimension | % |
|-----------|---|
| Code written (Phases A–D) | ~85 % |
| Code written (Phase E)    | ~70 % |
| Validated against a real MDM | ~10 % (only `pytest` on the synthetic model so far) |
| Published numbers reproduced | **0 / 7** rows in `reproduction_targets.yaml` |

The 0/7 row is the Month-1 milestone gate (proposal §7).

## Phase-by-phase status

Legend: ✅ done · ⚠️ partial / latent issue · ❌ missing

### Phase A — Scaffolding

| Item | Status | File |
|------|--------|------|
| `pyproject.toml` | ✅ | `pyproject.toml` |
| `.gitignore` | ✅ | `.gitignore` (with `!results/summary/` exception) |
| `README.md` | ✅ | `README.md` |
| `utils/device.py` | ✅ | `src/mdm_chipmunk/utils/device.py` |
| `utils/logging.py` | ✅ | `src/mdm_chipmunk/utils/logging.py` |

### Phase B — Model interface

| Item | Status | Notes |
|------|--------|-------|
| `models/base.py` (MDMModel ABC) | ✅ | |
| `models/mdlm.py` | ✅ | Tokenizer load fixed in `88dc5ad` — falls back to GPT-2 BPE since the MDLM HF repo doesn't register `AutoTokenizer` for its custom config. **Not yet run on GPU.** |
| `models/llada.py` | ✅ | `lazy=True` path validated; fp16 weight load **never executed** |
| `models/synthetic.py` | ✅ | Bonus — random-init MDM for CPU/CI |
| `tests/test_models_smoke.py` | ⚠️ | 3/4 pass; `test_registry_lazy_llada` hangs without `HF_HUB_OFFLINE=1` — **needs `@pytest.mark.network` skip-by-default** |

### Phase C — Dense baseline + harness

| Item | Status | Notes |
|------|--------|-------|
| `methods/base.py`, `methods/dense.py` | ✅ | LLaDA-style low-confidence remasking |
| `eval/harness.py` | ✅ | W&B logging added in `88dc5ad`, gated by `WANDB_PROJECT` env var |
| `eval/metrics.py`, `eval/tasks/{gsm8k,synthetic}.py` | ✅ | |
| `cli.py` (run / list / compare) | ✅ | `compare` never run against a real result |
| `scripts/smoke_test_cpu.py` | ✅ | Points at `mdlm-170m × gsm8k × dense` (`19e0e30`); **never completed end-to-end** on this Mac due to Pylance/disk contention — see [Known issues](#known-issues) |
| `tests/test_dense_method.py`, `tests/test_harness.py` | ✅ | Both pass on synthetic model |

### Phase D — Baseline method ports

| Item | Status | Notes |
|------|--------|-------|
| `methods/fastdllm.py` — `vanilla` | ✅ | Runs on any `MDMModel`. Validated on synthetic via `tests/test_fastdllm.py` |
| `methods/fastdllm.py` — `prefix_cache`, `dual_cache` | ⚠️ | Code present, **guarded** — needs Fast-dLLM's modified LLaDA fork. Never exercised end-to-end |
| `methods/dkv_cache.py` — `no_cache` | ✅ | Validated on synthetic |
| `methods/dkv_cache.py` — `decode`, `greedy` | ⚠️ | Same pattern — needs dKV-Cache-modified LLaDA fork. Never exercised |
| `methods/dualdiffusion.py` | ⚠️ | Clean-room (no upstream code release). Interface only, no real verifier ever run |
| `configs/methods/*.yaml` | ✅ | All 6 (`dense`, `fastdllm`, `fastdllm_vanilla`, `dkv_cache`, `dkv_cache_no_cache`, `dualdiffusion`) |
| `tests/test_fastdllm.py` (6 tests) | ✅ | All pass |
| `tests/test_dkv_cache.py` (6 tests) | ⚠️ | 5/6 pass — `test_dkv_cache_greedy_requires_random_remasking` fails: divisibility check fires before the `NotImplementedError` the test is asserting. **Test bug, not method bug** |
| `tests/test_dualdiffusion.py` (3 tests) | ✅ | All pass |

### Phase E — GPU-host readiness

| Item | Status | Notes |
|------|--------|-------|
| `scripts/clone_baselines.sh`, `scripts/download_llada.sh` | ✅ | |
| `configs/hosts/{h100,a100,laptop}.yaml` | ⚠️ | Files exist, **but `cli.py` doesn't read them** — dead config |
| README "Running on H100" section | ✅ | |
| `configs/reproduction_targets.yaml` | ✅ | 7 target rows |
| `results/summary/month1_baselines.md` (committed table) | ✅ | All rows currently `—` |

## What's structurally missing vs the plan

1. **HumanEval task** — `eval/tasks/humaneval.py` + `configs/tasks/humaneval.yaml`
2. **MMLU task** — same
3. **RULER 8K task** — Phase E item; long-context, GPU-only
4. **CLI ↔ host config wiring** — `--host` flag or env var that loads `configs/hosts/*.yaml`

## Known issues

| # | Issue | Severity | Where |
|---|-------|----------|-------|
| 1 | `test_registry_lazy_llada` hangs on stock laptops (no timeout on HF tokenizer fetch) | medium | `tests/test_models_smoke.py:32` |
| 2 | `test_dkv_cache_greedy_requires_random_remasking` fails (block_length=32 vs gen_length=4) | low | `tests/test_dkv_cache.py:45` |
| 3 | CPU smoke against MDLM-170M never completed on macOS — Pylance re-indexing the venv made disk reads ~100× slower than usual | env-only | `scripts/smoke_test_cpu.py` |
| 4 | `configs/hosts/*.yaml` are not loaded by anything | low | `src/mdm_chipmunk/cli.py` |

## Validated vs. not validated

| Component | On `synthetic` | On `mdlm-170m` | On `llada-8b` |
|-----------|----------------|----------------|---------------|
| `dense` | ✅ pytest | ❌ | ❌ |
| `fastdllm vanilla` | ✅ pytest | ❌ | ❌ |
| `fastdllm prefix_cache / dual_cache` | n/a (guard rejects) | ❌ | ❌ |
| `dkv_cache no_cache` | ✅ pytest | ❌ | ❌ |
| `dkv_cache decode / greedy` | n/a (guard rejects) | ❌ | ❌ |
| `dualdiffusion` | ✅ pytest | ❌ | ❌ |
| Harness end-to-end (writes JSONL) | ✅ (`results/smoke.jsonl`) | ❌ | ❌ |
| W&B streaming | ❌ (env var never set in any run) | ❌ | ❌ |
| `mdm-bench compare` vs `reproduction_targets.yaml` | n/a | n/a | ❌ |

## Next concrete steps

In rough priority order, with the hardware each step needs.

### Local (laptop CPU or RTX 2080 12 GB)

1. **Fix Known Issue #2** (one-line: pass `block_length=4` in the failing test).
2. **Fix Known Issue #1** — mark `test_registry_lazy_llada` with `@pytest.mark.network` and skip-by-default, or wrap the HF fetch in a timeout.
3. **First real-MDM smoke** — run `MDM_DEVICE=cuda python scripts/smoke_test_cpu.py` on the 2080. Should complete in ~30 s, not 60+ min. Validates Phase B + C end-to-end against actual MDM weights.
4. **Method-port smokes on MDLM-170M** — `mdm-bench run --method {fastdllm,dkv_cache,dense} --model mdlm-170m --task gsm8k --num-samples 20`. Validates the no-cache/vanilla variants on a real `forward_logits`. Writes JSONL the W&B run can consume.

### Rented A100 80 GB (~$1.20/hr, target ~$60–120 spend)

5. **Month-1 milestone gate** — for each of `dense`, `fastdllm`, `dkv_cache` × `gsm8k` (100 samples) on LLaDA-8B-Instruct, write JSONL → `mdm-bench compare` against `configs/reproduction_targets.yaml`. Update [`results/summary/month1_baselines.md`](../results/summary/month1_baselines.md) rows with real accuracy + commit SHA.
6. **Add HumanEval task** + repeat above for the compute-bound tasks (`fastdllm` only gets 3.7–4× here per Wu et al. — confirming this is the proposal's strategic pivot point).

### Rented H100 SXM5 (~$2.69/hr, target ~$600–1200 over Months 2–6)

7. **Same baselines on H100** for the cross-arch ablation (proposal §4).
8. **Begin Month 2 work** — ThunderKittens kernel v1 (gather + pack + dense GEMM). Out of scope for this doc.

## How to update this doc

After landing meaningful work:

1. Bump the **Last updated** date and `commit` line at the top.
2. Move items between ✅ / ⚠️ / ❌ as their state changes.
3. If a row in `results/summary/month1_baselines.md` flips from `—` to a real number, mark the corresponding "Validated vs. not validated" cell ✅ and add a one-line entry under the rented-hardware section explaining what landed.
