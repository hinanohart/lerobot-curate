"""Command-line interface for lerobot-curate.

The headline verb is ``curate``: stream a dataset, select a subset, and
optionally materialize a derived LeRobot v3 dataset (``--push-to``). Standalone
stage commands (``dedup``/``diversity``/``mislabel``/``select``) expose the same
pipeline pieces. ``doctor`` reports backend availability honestly.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from ._version import __version__
from .core.dedup import find_near_duplicates
from .core.diversity import diversity_of
from .core.mislabel import MislabelItem, detect_mislabels
from .curate import CurateConfig, prepare
from .curate import curate as curate_pipeline
from .diagnostics import doctor_text
from .embed import resolve_embedder
from .io import materialize, open_dataset
from .ir import SCOPE_DISCLAIMER, CurationReport, MislabelStatus, SelectionResult

app = typer.Typer(add_completion=False, help="CPU-only LeRobot v3 dataset curation.")


def _summary_text(result: SelectionResult, report: CurationReport) -> str:
    by_reason: dict[str, int] = {}
    for r in result.drop_reasons.values():
        by_reason[r] = by_reason.get(r, 0) + 1
    drops = ", ".join(f"{n} {reason}" for reason, n in sorted(by_reason.items())) or "none"
    bytes_dl = report.provenance.get("bytes_downloaded")
    badge = "0 GB (streamed)" if bytes_dl == 0 else f"{bytes_dl} bytes"
    lines = [
        f"source:   {report.source_repo}",
        f"embedder: {report.embedder.get('id')} ({report.embedder.get('source')})",
        f"kept {report.n_episodes_out} / {report.n_episodes_in} episodes",
        f"dropped:  {drops}",
        f"diversity (Vendi): {report.diversity.get('vendi_subset', report.diversity.get('vendi_full'))}",
        f"downloaded: {badge}",
    ]
    return "\n".join(lines)


def _html_report(report: CurationReport) -> str:
    d = report.to_dict()
    rows = "".join(
        f"<tr><td>{ep}</td><td>{reason}</td></tr>" for ep, reason in d["drop_reasons"].items()
    )
    return (
        "<!doctype html><meta charset='utf-8'>"
        f"<title>lerobot-curate report: {d['source_repo']}</title>"
        "<style>body{font-family:system-ui,sans-serif;margin:2rem;max-width:50rem}"
        "table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:4px 8px}"
        ".disc{background:#fffbe6;border:1px solid #f0d000;padding:8px;border-radius:6px}</style>"
        f"<h1>lerobot-curate report</h1><p><b>{d['source_repo']}</b></p>"
        f"<p>kept {d['n_episodes_out']} / {d['n_episodes_in']} episodes; "
        f"embedder {d['embedder'].get('id')} ({d['embedder'].get('source')}); "
        f"downloaded {d['provenance'].get('bytes_downloaded')} bytes.</p>"
        f"<div class='disc'>{SCOPE_DISCLAIMER}</div>"
        f"<h2>Dropped episodes</h2><table><tr><th>episode</th><th>reason</th></tr>{rows}</table>"
    )


@app.command()
def version() -> None:
    """Print the version."""
    typer.echo(__version__)


@app.command()
def doctor() -> None:
    """Report which embedder/decoder/export backends are available."""
    typer.echo(doctor_text())


def _config(
    budget: int | None, seed: int, cosine_threshold: float, max_frames: int
) -> CurateConfig:
    return CurateConfig(
        budget=budget, seed=seed, cosine_threshold=cosine_threshold, max_frames=max_frames
    )


@app.command()
def curate(
    repo_id: str = typer.Argument(..., help="HF repo id or local path of a LeRobot v3 dataset"),
    budget: int | None = typer.Option(None, "--budget", "-b", help="max episodes to keep"),
    push_to: str | None = typer.Option(
        None, "--push-to", help="write a derived dataset to this dir"
    ),
    html: str | None = typer.Option(None, "--html", help="write an HTML report to this path"),
    report_out: str | None = typer.Option(
        None, "--report", help="write the JSON report to this path"
    ),
    embedder: str | None = typer.Option(None, "--embedder", help="stub|local-onnx|hf-api (auto)"),
    seed: int = typer.Option(0, "--seed"),
    cosine_threshold: float = typer.Option(0.99, "--cosine-threshold"),
    max_frames: int = typer.Option(8, "--max-frames"),
) -> None:
    """Curate a dataset and optionally materialize a derived subset."""
    ds = open_dataset(repo_id)
    emb = resolve_embedder(embedder)
    result, report = curate_pipeline(ds, emb, _config(budget, seed, cosine_threshold, max_frames))
    typer.echo(_summary_text(result, report))
    if report_out:
        Path(report_out).write_text(report.to_json())
        typer.echo(f"report -> {report_out}")
    if html:
        Path(html).write_text(_html_report(report))
        typer.echo(f"html -> {html}")
    if push_to:
        mat = materialize(ds, result, push_to, report)
        typer.echo(f"materialized {mat.n_episodes} episodes -> {mat.out_root}")


@app.command()
def select(
    repo_id: str = typer.Argument(...),
    budget: int | None = typer.Option(None, "--budget", "-b"),
    embedder: str | None = typer.Option(None, "--embedder"),
    seed: int = typer.Option(0, "--seed"),
) -> None:
    """Run selection only (no materialize); print kept episode ids."""
    ds = open_dataset(repo_id)
    emb = resolve_embedder(embedder)
    result, report = curate_pipeline(ds, emb, _config(budget, seed, 0.99, 8))
    typer.echo(_summary_text(result, report))
    typer.echo("kept: " + ", ".join(str(e) for e in result.kept_episode_ids))


@app.command()
def dedup(
    repo_id: str = typer.Argument(...),
    cosine_threshold: float = typer.Option(0.99, "--cosine-threshold"),
    embedder: str | None = typer.Option(None, "--embedder"),
    seed: int = typer.Option(0, "--seed"),
) -> None:
    """Report near-duplicate episodes."""
    ds = open_dataset(repo_id)
    prep = prepare(ds, resolve_embedder(embedder), _config(None, seed, cosine_threshold, 8))
    drop_idx, pairs = find_near_duplicates(prep.image_means, cosine_threshold, None, seed)
    typer.echo(f"{len(drop_idx)} near-duplicate episodes")
    for a, b in pairs:
        typer.echo(f"  episode {prep.ep_index[b]} ~ {prep.ep_index[a]}")


@app.command()
def diversity(
    repo_id: str = typer.Argument(...),
    embedder: str | None = typer.Option(None, "--embedder"),
    seed: int = typer.Option(0, "--seed"),
) -> None:
    """Report the Vendi diversity score of the dataset."""
    ds = open_dataset(repo_id)
    prep = prepare(ds, resolve_embedder(embedder), _config(None, seed, 0.99, 8))
    v = diversity_of(prep.features, seed=seed) if prep.features.size else 0.0
    typer.echo(f"feature_mode={prep.feat_mode} n={len(prep.ep_index)} vendi={v:.4f}")


@app.command()
def mislabel(
    repo_id: str = typer.Argument(...),
    embedder: str | None = typer.Option(None, "--embedder"),
    seed: int = typer.Option(0, "--seed"),
) -> None:
    """Report cross-modal mislabel flags."""
    ds = open_dataset(repo_id)
    prep = prepare(ds, resolve_embedder(embedder), _config(None, seed, 0.99, 8))
    items = [
        MislabelItem(
            prep.ep_index[i], prep.fes[i].mean_vector, prep.fes[i].text_vector, prep.refs[i].task
        )
        for i in range(len(prep.ep_index))
    ]
    flags = detect_mislabels(items)
    counts: dict[str, int] = {}
    for f in flags:
        counts[f.status] = counts.get(f.status, 0) + 1
    typer.echo(", ".join(f"{n} {s}" for s, n in counts.items()) or "no episodes")
    for f in flags:
        if f.status is MislabelStatus.SUSPECTED:
            typer.echo(f"  suspected episode {f.episode_index}: {f.reason}")


@app.command()
def report(path: str = typer.Argument(..., help="path to a curation_report.json")) -> None:
    """Pretty-print a saved curation report."""
    rep = CurationReport.from_dict(json.loads(Path(path).read_text()))
    typer.echo(_summary_text_from_report(rep))


def _summary_text_from_report(rep: CurationReport) -> str:
    return (
        f"source: {rep.source_repo}\n"
        f"kept {rep.n_episodes_out} / {rep.n_episodes_in}\n"
        f"embedder: {rep.embedder.get('id')} ({rep.embedder.get('source')})\n"
        f"disclaimer present: {any('does NOT reproduce' in d for d in rep.disclaimers)}"
    )


@app.command(name="export-fiftyone")
def export_fiftyone(
    repo_id: str = typer.Argument(...),
    embedder: str | None = typer.Option(None, "--embedder"),
) -> None:
    """Export a curation result to FiftyOne (requires the [fiftyone] extra)."""
    try:
        import fiftyone  # noqa: F401
    except ImportError as exc:
        typer.echo(
            "FiftyOne is not installed. Install with: pip install 'lerobot-curate[fiftyone]'"
        )
        raise typer.Exit(code=1) from exc
    typer.echo("FiftyOne export is a v0.1.1 feature; the extra is installed.")


if __name__ == "__main__":
    app()
