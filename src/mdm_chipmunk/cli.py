from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

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


if __name__ == "__main__":
    app()
