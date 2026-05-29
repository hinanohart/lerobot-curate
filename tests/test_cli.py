import json

from typer.testing import CliRunner

from lerobot_curate._version import __version__
from lerobot_curate.cli import app
from lerobot_curate.io import make_synthetic_v3

runner = CliRunner()


def _make(tmp_path):
    make_synthetic_v3(tmp_path, n_episodes=8, n_frames=8, dup_groups=[[0, 1]], seed=0)
    return str(tmp_path)


def test_cli_version():
    res = runner.invoke(app, ["version"])
    assert res.exit_code == 0
    assert __version__ in res.stdout


def test_cli_doctor():
    res = runner.invoke(app, ["doctor"])
    assert res.exit_code == 0
    assert "embedder:stub" in res.stdout
    assert "never bundled" in res.stdout.lower()


def test_cli_curate_with_materialize(tmp_path):
    src = _make(tmp_path / "src")
    out = tmp_path / "derived"
    rep = tmp_path / "report.json"
    html = tmp_path / "report.html"
    res = runner.invoke(
        app,
        [
            "curate",
            src,
            "--budget",
            "3",
            "--embedder",
            "stub",
            "--push-to",
            str(out),
            "--report",
            str(rep),
            "--html",
            str(html),
        ],
    )
    assert res.exit_code == 0, res.stdout
    assert "kept" in res.stdout
    assert rep.exists() and html.exists()
    assert (out / "meta" / "info.json").exists()
    # report.json has non-empty drop_reasons (the dedup'd episodes), DoD smoke (P8)
    data = json.loads(rep.read_text())
    assert data["n_episodes_in"] == 8
    assert len(data["drop_reasons"]) >= 1
    # html carries disclaimer
    assert "does NOT reproduce" in html.read_text()


def test_cli_stage_commands(tmp_path):
    src = _make(tmp_path / "src")
    for cmd in (["dedup", src], ["diversity", src], ["mislabel", src], ["select", src, "-b", "3"]):
        res = runner.invoke(app, [*cmd, "--embedder", "stub"] if "--embedder" not in cmd else cmd)
        assert res.exit_code == 0, (cmd, res.stdout)


def test_cli_report_roundtrip(tmp_path):
    src = _make(tmp_path / "src")
    rep = tmp_path / "report.json"
    runner.invoke(app, ["curate", src, "--embedder", "stub", "--report", str(rep)])
    res = runner.invoke(app, ["report", str(rep)])
    assert res.exit_code == 0
    assert "disclaimer present: True" in res.stdout


def test_cli_export_fiftyone_degrades_cleanly(tmp_path):
    src = _make(tmp_path / "src")
    res = runner.invoke(app, ["export-fiftyone", src])
    # fiftyone not installed in dev -> clean message, no traceback
    if res.exit_code != 0:
        assert "fiftyone" in res.stdout.lower()
