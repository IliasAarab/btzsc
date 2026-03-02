from __future__ import annotations

import json
from pathlib import Path

import click
from tabulate import tabulate


@click.group()
def main() -> None:
    """BTZSC benchmark CLI."""


@main.command()
@click.option("--model", "model_name", required=True, help="HF model name or path")
@click.option("--type", "model_type", type=click.Choice(["embedding", "nli", "reranker", "llm"]), required=True)
@click.option("--batch-size", type=int, default=32)
@click.option("--tasks", default="", help="Comma-separated task groups or dataset names")
@click.option("--max-samples", type=int, default=None)
@click.option("--output", type=click.Path(path_type=Path), default=None)
@click.option(
    "--output-json",
    type=click.Path(path_type=Path),
    default=None,
    help="Write results to a JSON file in BTZSC leaderboard format.",
)
def evaluate(
    model_name: str,
    model_type: str | None,
    batch_size: int,
    tasks: str,
    max_samples: int | None,
    output: Path | None,
    output_json: Path | None,
) -> None:
    """Run benchmark evaluation from the CLI.

    Args:
        model_name: HuggingFace model name or local model path.
        model_type: Explicit model adapter type.
        batch_size: Inference batch size.
        tasks: Comma-separated task-group or dataset names.
        max_samples: Optional per-dataset sample limit.
        output: Optional CSV path for per-dataset results.
        output_json: Optional JSON path for leaderboard submission output.
    """
    from btzsc.benchmark import BTZSCBenchmark

    selected = [t.strip() for t in tasks.split(",") if t.strip()] or None
    benchmark = BTZSCBenchmark(tasks=selected)
    results = benchmark.evaluate(model_name, model_type=model_type, batch_size=batch_size, max_samples=max_samples)

    df = results.per_dataset()
    click.echo(tabulate(df, headers="keys", tablefmt="github", showindex=False, floatfmt=".4f"))
    click.echo("\nSummary:")
    click.echo(tabulate(results.summary(), headers="keys", tablefmt="github", showindex=False, floatfmt=".4f"))

    if output is not None:
        results.to_csv(output)
        click.echo(f"\nSaved per-dataset results to {output}")

    if output_json is not None:
        results.to_json(output_json)
        click.echo(f"Saved leaderboard JSON to {output_json}")


@main.command()
@click.option("--metric", type=click.Choice(["f1", "accuracy", "precision", "recall", "roc"]), default="f1")
@click.option("--top", type=int, default=10)
def baselines(metric: str, top: int) -> None:
    """Show top baseline models for a metric.

    Args:
        metric: Metric used to rank baselines.
        top: Number of rows to display.
    """
    from btzsc.baselines import get_baselines

    df = get_baselines(metric=metric)
    means = df.drop(columns=["mdl"]).mean(axis=1)
    out = (
        df[["mdl"]]
        .assign(score=means)
        .rename(columns={"mdl": "model"})
        .sort_values("score", ascending=False)
        .head(top)
        .reset_index(drop=True)
    )
    click.echo(tabulate(out, headers="keys", tablefmt="github", showindex=False, floatfmt=".4f"))


@main.command("list-datasets")
def list_datasets() -> None:
    """List available BTZSC datasets with their task groups."""
    from btzsc.data import BTZSC_DATASETS, TASK_GROUPS

    rows = [
        {"dataset": ds, "task": next((t for t, dss in TASK_GROUPS.items() if ds in dss), "unknown")}
        for ds in BTZSC_DATASETS
    ]
    click.echo(tabulate(rows, headers="keys", tablefmt="github", showindex=False))


@main.command("list-model-types")
def list_model_types() -> None:
    """Print supported model adapter types as JSON."""
    click.echo(json.dumps(["embedding", "nli", "reranker", "llm"], indent=2))


@main.command("validate-result")
@click.argument("json_path", type=click.Path(path_type=Path, exists=True, dir_okay=False))
def validate_result(json_path: Path) -> None:
    """Validate a leaderboard JSON file against BTZSC schema checks."""
    from btzsc.leaderboard.validate import validate_result_file

    errors = validate_result_file(json_path)
    if errors:
        click.echo(f"Validation failed ({len(errors)} errors):")
        for err in errors:
            click.echo(f"  - {err}")
        raise SystemExit(1)

    click.echo("Validation passed.")
