# Month 1 — Baseline reproduction summary

One row per `(method, model, task)`. Numbers come from `mdm-bench run` output
(per-sample JSONL in `results/`); see the linked W&B run for full telemetry.

Status legend: `—` = not yet run, `🟡` = ran but below repro tolerance,
`🟢` = within ±5 % of published.

## GSM8K (5-shot, gen 256 / 512 steps unless noted)

| Method        | Model              | N   | Acc   | Mean lat (s) | tok/s | NFE | Peak VRAM | Status | Commit | W&B run |
|---------------|--------------------|-----|-------|--------------|-------|-----|-----------|--------|--------|---------|
| dense         | mdlm-170m          | —   | —     | —            | —     | —   | —         | —      | —      | —       |
| dense         | llada-8b-instruct  | —   | —     | —            | —     | —   | —         | —      | —      | —       |
| fastdllm      | mdlm-170m          | —   | —     | —            | —     | —   | —         | —      | —      | —       |
| fastdllm      | llada-8b-instruct  | —   | —     | —            | —     | —   | —         | —      | —      | —       |
| dkv_cache     | mdlm-170m          | —   | —     | —            | —     | —   | —         | —      | —      | —       |
| dkv_cache     | llada-8b-instruct  | —   | —     | —            | —     | —   | —         | —      | —      | —       |
| dualdiffusion | llada-8b-instruct  | —   | —     | —            | —     | —   | —         | —      | —      | —       |

## HumanEval (pass@1, gen 512)

Task not yet implemented — see Phase D in `docs/month1_baselines_plan.md`.

## MMLU (5-shot)

Task not yet implemented.

## Reproduction targets

See `configs/reproduction_targets.yaml`. `mdm-bench compare --results <file>`
checks each row in this table against the published target within ±5 %.

## How rows land here

1. Run `mdm-bench run --method X --model Y --task Z --num-samples N --out results/X_Y_Z.jsonl`.
2. The harness writes per-sample JSONL **and** (if `WANDB_PROJECT` is set in the
   env) streams telemetry to W&B.
3. Paste the resulting summary numbers into this file, with the short commit SHA
   from `git rev-parse --short HEAD` at run time and a link to the W&B run.

Rows are committed so the table is a stable reference. The JSONL files are
gitignored — W&B holds the per-sample logs.
