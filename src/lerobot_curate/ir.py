"""Normative intermediate representation (IR) for lerobot-curate.

Every downstream module (embed/, core/, curate, report, cli) binds to these
types. Design rules:

* In-memory numerical objects (``FrameEmbedding``, ``EpisodeSignature``) may hold
  numpy arrays and are never serialized with pickle.
* The only serialized artifact is ``CurationReport`` -> plain JSON. It stores
  summaries and provenance, never raw embedding/signature arrays.
* ``EpisodeSignature`` carries a machine-readable ``scope`` invariant
  (``intra-dataset-only``); signatures from different source datasets must never
  be compared, and this is asserted at construction.
* Mislabel results are degrade-first-class: a missing/degenerate task string
  yields ``MislabelStatus.NOT_EVALUATED`` and is never silently reported ``OK``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

# Verbatim scope disclaimer. Reused by README generation, the CLI banner, and
# every CurationReport. Keep this exact wording (CI greps for the key phrase).
SCOPE_DISCLAIMER = (
    "lerobot-curate is a CPU reference implementation of the SCIZOR (embedding "
    "dedup/suboptimal) and FAKTUAL (signature-kernel diversity) methods. It does "
    "NOT reproduce their reported policy-performance gains (those require GPU "
    "policy training and are out of scope). Validated on algorithm-correctness "
    "metrics only."
)

SIGNATURE_SCOPE = "intra-dataset-only"


class EmbeddingSource(StrEnum):
    LOCAL_ONNX = "local-onnx"
    HF_API = "hf-api"
    STUB = "stub"


class DropReason(StrEnum):
    NEAR_DUPLICATE = "near-duplicate"
    LOW_DIVERSITY = "low-diversity"
    SUBOPTIMAL = "suboptimal"
    MISLABEL_SUSPECTED = "mislabel-suspected"


class MislabelStatus(StrEnum):
    OK = "ok"
    SUSPECTED = "suspected"
    NOT_EVALUATED = "not-evaluated"


@dataclass(frozen=True)
class EpisodeRef:
    """A reference to one episode inside a (streamed) LeRobot v3 dataset."""

    repo_id: str
    episode_index: int
    num_frames: int
    fps: float | None = None
    task: str | None = None
    video_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.episode_index < 0:
            raise ValueError("episode_index must be non-negative")
        if self.num_frames < 0:
            raise ValueError("num_frames must be non-negative")


@dataclass
class FrameEmbedding:
    """Representative-frame embeddings for one episode.

    ``vectors`` has shape ``(n_keyframes, dim)`` (image embeddings).
    ``text_vector`` (shape ``(dim,)``) is the task-text embedding when the
    backend has a text head; ``None`` means cross-modal mislabel cannot run.
    """

    episode_index: int
    vectors: np.ndarray
    source: EmbeddingSource
    model_id: str
    text_vector: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.vectors.ndim != 2:
            raise ValueError(f"vectors must be 2D (n_frames, dim), got {self.vectors.shape}")
        if self.text_vector is not None and self.text_vector.ndim != 1:
            raise ValueError("text_vector must be 1D (dim,)")
        if self.text_vector is not None and self.text_vector.shape[0] != self.vectors.shape[1]:
            raise ValueError("text_vector dim must match image embedding dim")

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    @property
    def mean_vector(self) -> np.ndarray:
        mv: np.ndarray = np.asarray(self.vectors.mean(axis=0))
        return mv


@dataclass
class EpisodeSignature:
    """Truncated path-signature feature vector for one episode.

    ``scope`` is a machine-readable invariant: signatures are only comparable
    within a single source dataset (cross-dataset comparison is deferred to the
    v0.1.1 optimal-transport path).
    """

    episode_index: int
    coeffs: np.ndarray
    depth: int
    rff_dim: int
    scope: str = SIGNATURE_SCOPE

    def __post_init__(self) -> None:
        if self.coeffs.ndim != 1:
            raise ValueError("signature coeffs must be 1D")
        if self.depth < 1:
            raise ValueError("signature depth must be >= 1")
        if self.scope != SIGNATURE_SCOPE:
            raise ValueError(
                f"EpisodeSignature.scope must be {SIGNATURE_SCOPE!r}; "
                "cross-dataset signature comparison is not supported in a1"
            )


@dataclass
class MislabelFlag:
    episode_index: int
    status: MislabelStatus
    score: float | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ep": self.episode_index,
            "status": self.status.value,
            "score": self.score,
            "reason": self.reason,
        }


@dataclass
class SelectionResult:
    """Outcome of the selection pipeline for a single dataset."""

    kept_episode_ids: list[int]
    drop_reasons: dict[int, DropReason] = field(default_factory=dict)
    dedup_pairs: list[tuple[int, int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        overlap = set(self.kept_episode_ids) & set(self.drop_reasons)
        if overlap:
            raise ValueError(f"episodes both kept and dropped: {sorted(overlap)}")

    @property
    def n_kept(self) -> int:
        return len(self.kept_episode_ids)

    @property
    def n_dropped(self) -> int:
        return len(self.drop_reasons)


@dataclass
class CurationReport:
    """The single JSON artifact produced by a curation run."""

    source_repo: str
    tool_version: str
    schema_version: int = 1
    tool_name: str = "lerobot-curate"
    source_revision: str | None = None
    embedder: dict[str, Any] = field(default_factory=dict)
    n_episodes_in: int = 0
    n_episodes_out: int = 0
    kept_episode_ids: list[int] = field(default_factory=list)
    drop_reasons: dict[int, str] = field(default_factory=dict)
    dedup_pairs: list[tuple[int, int]] = field(default_factory=list)
    diversity: dict[str, Any] = field(default_factory=dict)
    mislabel_flags: list[MislabelFlag] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    disclaimers: list[str] = field(default_factory=lambda: [SCOPE_DISCLAIMER])

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "tool": {"name": self.tool_name, "version": self.tool_version},
            "source_repo": self.source_repo,
            "source_revision": self.source_revision,
            "embedder": self.embedder,
            "n_episodes_in": self.n_episodes_in,
            "n_episodes_out": self.n_episodes_out,
            "kept_episode_ids": list(self.kept_episode_ids),
            "drop_reasons": {str(k): v for k, v in self.drop_reasons.items()},
            "dedup_pairs": [list(p) for p in self.dedup_pairs],
            "diversity": self.diversity,
            "mislabel_flags": [m.to_dict() for m in self.mislabel_flags],
            "provenance": self.provenance,
            "disclaimers": list(self.disclaimers),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CurationReport:
        return cls(
            source_repo=d["source_repo"],
            tool_version=d.get("tool", {}).get("version", "unknown"),
            schema_version=d.get("schema_version", 1),
            tool_name=d.get("tool", {}).get("name", "lerobot-curate"),
            source_revision=d.get("source_revision"),
            embedder=d.get("embedder", {}),
            n_episodes_in=d.get("n_episodes_in", 0),
            n_episodes_out=d.get("n_episodes_out", 0),
            kept_episode_ids=list(d.get("kept_episode_ids", [])),
            drop_reasons={int(k): v for k, v in d.get("drop_reasons", {}).items()},
            dedup_pairs=[(int(a), int(b)) for a, b in d.get("dedup_pairs", [])],
            diversity=d.get("diversity", {}),
            mislabel_flags=[
                MislabelFlag(
                    episode_index=m["ep"],
                    status=MislabelStatus(m["status"]),
                    score=m.get("score"),
                    reason=m.get("reason"),
                )
                for m in d.get("mislabel_flags", [])
            ],
            provenance=d.get("provenance", {}),
            disclaimers=list(d.get("disclaimers", [SCOPE_DISCLAIMER])),
        )
