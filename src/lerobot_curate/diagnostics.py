"""Honest backend availability report (the ``doctor`` command).

Reports what is actually importable in the current environment instead of a
misleading all-green table. Used by the CLI and by tests.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass


@dataclass
class BackendStatus:
    name: str
    available: bool
    note: str


def _has(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


def doctor() -> list[BackendStatus]:
    onnx = _has("onnxruntime")
    pil = _has("PIL")
    tok = _has("tokenizers")
    av = _has("av")
    cv = _has("cv2")
    return [
        BackendStatus("embedder:stub", True, "deterministic; default in CI, no weights/network"),
        BackendStatus(
            "embedder:local-onnx (vision)",
            onnx and pil,
            "ready" if (onnx and pil) else "install lerobot-curate[onnx] (onnxruntime, pillow)",
        ),
        BackendStatus(
            "embedder:local-onnx (text head / mislabel)",
            onnx and tok,
            "ready"
            if (onnx and tok)
            else "needs tokenizers (in [onnx]); text head enables mislabel",
        ),
        BackendStatus(
            "embedder:hf-api", True, "opt-in, OFF by default; free tier is rate-limited (degraded)"
        ),
        BackendStatus(
            "frame-decode:pyav",
            av,
            "ready" if av else "install lerobot-curate[video] for real Hub video frames",
        ),
        BackendStatus(
            "frame-decode:opencv",
            cv,
            "ready" if cv else "optional fallback: lerobot-curate[opencv]",
        ),
        BackendStatus(
            "export:fiftyone",
            _has("fiftyone"),
            "ready" if _has("fiftyone") else "optional: lerobot-curate[fiftyone]",
        ),
    ]


def doctor_text() -> str:
    lines = ["lerobot-curate doctor — backend availability", ""]
    for s in doctor():
        mark = "ok " if s.available else "-- "
        lines.append(f"  [{mark}] {s.name}: {s.note}")
    lines.append("")
    lines.append("Weights for local-onnx download on demand to your HF cache; never bundled.")
    return "\n".join(lines)
