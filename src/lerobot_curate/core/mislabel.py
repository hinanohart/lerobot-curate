"""Cross-modal mislabel detection (degrade-first-class).

For each episode we compare the task-text embedding to the mean frame embedding.
A low cosine relative to the rest of the dataset suggests the text does not
describe the video (a mislabel). Episodes without a usable task string, or run
with a backend that has no text head, are reported ``not-evaluated`` — never
silently ``ok``.

Threshold: an episode is ``suspected`` only when its cosine is in the bottom
``q`` quantile of the dataset AND below an absolute floor. The AND avoids
flagging a whole dataset that simply has uniformly modest cosines.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from ..ir import MislabelFlag, MislabelStatus

_PLACEHOLDER = re.compile(r"^\s*(n/?a|none|null|todo|task|episode|unknown|\.+)\s*$", re.IGNORECASE)


def usable_task(task: str | None, min_len: int = 3) -> bool:
    if task is None:
        return False
    t = task.strip()
    if len(t) < min_len:
        return False
    if _PLACEHOLDER.match(t):
        return False
    return True


@dataclass
class MislabelItem:
    episode_index: int
    image_mean: np.ndarray
    text_vector: np.ndarray | None
    task: str | None


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def detect_mislabels(
    items: list[MislabelItem],
    q: float = 0.05,
    abs_floor: float = 0.10,
) -> list[MislabelFlag]:
    """Return one flag per item, sorted by episode index."""
    flags: list[MislabelFlag] = []
    scored: list[tuple[int, float]] = []
    for it in items:
        if not usable_task(it.task):
            flags.append(
                MislabelFlag(
                    it.episode_index,
                    MislabelStatus.NOT_EVALUATED,
                    None,
                    "missing/degenerate task text",
                )
            )
        elif it.text_vector is None:
            flags.append(
                MislabelFlag(
                    it.episode_index,
                    MislabelStatus.NOT_EVALUATED,
                    None,
                    "embedder has no text head",
                )
            )
        else:
            scored.append((it.episode_index, _cosine(it.image_mean, it.text_vector)))

    if scored:
        cosines = np.array([s for _, s in scored])
        qv = float(np.quantile(cosines, q))
        for ep, s in scored:
            if s <= qv and s < abs_floor:
                flags.append(
                    MislabelFlag(
                        ep,
                        MislabelStatus.SUSPECTED,
                        round(s, 6),
                        f"cosine {s:.3f} in bottom {q:g} quantile and below floor {abs_floor:g}",
                    )
                )
            else:
                flags.append(MislabelFlag(ep, MislabelStatus.OK, round(s, 6), None))

    return sorted(flags, key=lambda f: f.episode_index)
