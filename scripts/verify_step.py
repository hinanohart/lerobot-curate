#!/usr/bin/env python3
"""Per-phase DoD verifier for the lerobot-curate autonomous build.

Usage: python scripts/verify_step.py S0_5

Each check makes REAL assertions against the working tree (never a vacuous
``return True``). Steps that have not been implemented yet fail loudly with
exit code 1 so progress cannot be faked. The build loop only advances a phase
when the matching check exits 0.
"""

from __future__ import annotations

import importlib.util
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "lerobot_curate"


class CheckError(AssertionError):
    pass


def _exists(rel: str) -> pathlib.Path:
    p = ROOT / rel
    if not p.exists():
        raise CheckError(f"missing required path: {rel}")
    return p


def _contains(rel: str, needle: str) -> None:
    text = _exists(rel).read_text(errors="ignore")
    if needle not in text:
        raise CheckError(f"{rel} does not contain expected text: {needle!r}")


# Only this project's own packages are ever imported here. modname always comes
# from the literal CHECKS table below, never from user/CLI input.
_ALLOWED_PREFIX = "lerobot_curate"


def _import(modname: str) -> object:
    if not (modname == _ALLOWED_PREFIX or modname.startswith(_ALLOWED_PREFIX + ".")):
        raise CheckError(f"refusing to import non-project module: {modname}")
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    spec = importlib.util.find_spec(modname)
    if spec is None:
        raise CheckError(f"cannot import {modname}")
    # nosemgrep: python.lang.security.audit.non-literal-import.non-literal-import
    return importlib.import_module(modname)  # guarded by _ALLOWED_PREFIX allowlist above


def _pytest(expr: str) -> None:
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-k", expr, str(ROOT / "tests")],
        cwd=ROOT,
    )
    if r.returncode != 0:
        raise CheckError(f"pytest -k {expr!r} failed (rc={r.returncode})")


# ---------------------------------------------------------------------------


def check_S0_5() -> None:
    for f in (
        "pyproject.toml",
        "LICENSE",
        "NOTICE",
        ".gitignore",
        "README.md",
        "src/lerobot_curate/__init__.py",
        "src/lerobot_curate/_version.py",
        "scripts/verify_step.py",
        "scripts/heartbeat.py",
        ".github/workflows/ci.yml",
    ):
        _exists(f)
    _contains("LICENSE", "Apache License")
    notice = _exists("NOTICE").read_text(errors="ignore").lower()
    if "never bundled" not in notice and "never committed" not in notice:
        raise CheckError("NOTICE does not state weights are never bundled/committed")
    _contains("NOTICE", "does NOT reproduce")
    _contains(".gitignore", "*.onnx")
    _contains("pyproject.toml", "[project.scripts]")
    _contains("pyproject.toml", 'lerobot-curate = "lerobot_curate.cli:app"')
    mod = _import("lerobot_curate")
    if getattr(mod, "__version__", None) != "0.1.0a1":
        raise CheckError(f"unexpected __version__: {getattr(mod, '__version__', None)!r}")


def check_S1() -> None:
    _import("lerobot_curate.ir")
    _pytest("ir")


def check_S2() -> None:
    _import("lerobot_curate.core.signature")
    _import("lerobot_curate.core.diversity")
    _pytest("signature or diversity")


def check_S3() -> None:
    _import("lerobot_curate.embed.stub")
    _import("lerobot_curate.io.lerobot_v3")
    _pytest("embed or io or stub")


def check_S4() -> None:
    _import("lerobot_curate.core.dedup")
    _import("lerobot_curate.core.mislabel")
    _import("lerobot_curate.curate")
    _pytest("dedup or mislabel or pipeline")


def check_S5() -> None:
    _import("lerobot_curate.cli")
    _pytest("cli or api or materialize")


def check_S6() -> None:
    res = _exists("results/v0.1.0a1_metrics.json")
    import json

    d = json.loads(res.read_text())
    for k in ("dedup_recall", "mislabel_precision", "negative_fpr", "diversity_monotonic"):
        if d.get(k) is None:
            raise CheckError(f"results metric {k} is null")


def check_S7() -> None:
    # honest-marketing grep is enforced in CI (.github/workflows). Here we assert
    # the verbatim scope disclaimer is present and no obvious placeholder remains.
    _contains("README.md", "does NOT reproduce their reported policy-performance gains")
    text = _exists("README.md").read_text()
    if "MEASURED@S6" in text or "TODO" in text:
        raise CheckError("README still has placeholders")


CHECKS = {
    "S0_5": check_S0_5,
    "S1": check_S1,
    "S2": check_S2,
    "S3": check_S3,
    "S4": check_S4,
    "S5": check_S5,
    "S6": check_S6,
    "S7": check_S7,
}


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[1] not in CHECKS:
        print(f"usage: verify_step.py <{'|'.join(CHECKS)}>", file=sys.stderr)
        return 2
    step = argv[1]
    try:
        CHECKS[step]()
    except CheckError as e:
        print(f"[verify {step}] FAIL: {e}", file=sys.stderr)
        return 1
    except Exception as e:  # noqa: BLE001
        print(f"[verify {step}] ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    print(f"[verify {step}] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
