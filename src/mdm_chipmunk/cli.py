from __future__ import annotations

import json
from pathlib import Path

import typer
import yaml
from rich.console import Console
from rich.table import Table

from .eval.harness import run
from .eval.tasks.registry import build_task, list_tasks
from .methods.registry import build_method, list_methods
from .models.registry import build_model, list_models
from .utils.device import device_info
from .utils.logging import get_logger

app = typer.Typer(help="mdm-bench: MDM inference benchmarking harness.")
console = Console()
_LOG = get_logger("mdm_chipmunk.cli")


@app.command("run")
def run_cmd(
    method: str = typer.Option(..., help="Method config name (see `mdm-bench list methods`)"),
    model: str = typer.Option(..., help="Model config name (see `mdm-bench list models`)"),
    task: str = typer.Option(..., help="Task config name (see `mdm-bench list tasks`)"),
    num_samples: int = typer.Option(10),
    num_steps: int = typer.Option(128, help="MDM denoising steps"),
    seed: int = typer.Option(0),
    out: Path = typer.Option(Path("results/run.jsonl")),
    lazy_model: bool = typer.Option(
        False, help="Lazy-load model (no weights). Useful for shape-checking the pipeline."
    ),
) -> None:
    info = device_info()
    _LOG.info("Device: %s (%s, %s)", info.name, info.device, info.dtype)

    mdm = build_model(model, lazy=lazy_model)
    inference = build_method(method)
    eval_task = build_task(task)

    summary = run(
        method=inference,
        model=mdm,
        task=eval_task,
        num_samples=num_samples,
        num_steps=num_steps,
        seed=seed,
        out_path=out,
    )
    console.print(summary.to_dict())


@app.command("list")
def list_cmd(kind: str = typer.Argument(..., help="One of: models, methods, tasks")) -> None:
    options = {"models": list_models, "methods": list_methods, "tasks": list_tasks}
    if kind not in options:
        raise typer.BadParameter(f"kind must be one of {list(options)}")
    for name in options[kind]():
        console.print(f"- {name}")


@app.command("compare")
def compare_cmd(
    results: Path = typer.Option(..., help="JSONL results file produced by `mdm-bench run`"),
    targets: Path = typer.Option(
        Path(__file__).resolve().parents[2] / "configs" / "reproduction_targets.yaml",
        help="YAML of reproduction targets (default: bundled config)",
    ),
) -> None:
    """Check whether reproduced numbers match published targets within tolerance."""
    if not results.exists():
        raise typer.BadParameter(f"Results file not found: {results}")
    with open(targets, encoding="utf-8") as f:
        tgt_cfg = yaml.safe_load(f)
    tolerance = float(tgt_cfg.get("tolerance", 0.05))

    rows = [json.loads(line) for line in results.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        console.print("[yellow]No rows in results file.[/yellow]")
        return

    method = rows[0]["method"]
    model = rows[0]["model"]
    task = rows[0]["task"]
    correct = sum(1 for r in rows if r["correct"]) / len(rows)

    match = None
    for t in tgt_cfg.get("targets", []):
        if t["method"] == method and t["model"] == model and t["task"] == task:
            match = t
            break

    table = Table(title=f"Reproduction check — {method} × {model} × {task}")
    table.add_column("Metric")
    table.add_column("Observed")
    table.add_column("Target")
    table.add_column("Δ%")
    table.add_column("Pass?")

    if match is None:
        console.print(
            f"[yellow]No target row for {method!r}/{model!r}/{task!r} — only summary printed.[/yellow]"
        )
        table.add_row("accuracy", f"{correct:.3f}", "—", "—", "—")
        console.print(table)
        return

    target_acc = match.get("accuracy")
    if target_acc is not None:
        delta = (correct - target_acc) / target_acc if target_acc else float("inf")
        passed = abs(delta) <= tolerance
        table.add_row(
            "accuracy",
            f"{correct:.3f}",
            f"{target_acc:.3f}",
            f"{delta * 100:+.1f}%",
            "[green]PASS[/green]" if passed else "[red]FAIL[/red]",
        )
    console.print(table)


if __name__ == "__main__":
    app()
