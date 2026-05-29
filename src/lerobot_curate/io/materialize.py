"""Materialize a derived LeRobot v3 dataset from a selection (the killer hook).

Given a source reader and a ``SelectionResult``, write a new on-disk dataset
containing only the kept episodes, re-indexed contiguously, plus the curation
report and a provenance record. This is what makes a curation run actionable:
``curate --push-to <dir>`` yields a ready-to-train subset without re-downloading.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ..ir import CurationReport, SelectionResult
from .lerobot_v3 import LeRobotV3Reader


@dataclass
class MaterializeResult:
    out_root: str
    n_episodes: int
    report_path: str


def materialize(
    reader: LeRobotV3Reader,
    result: SelectionResult,
    out_root: str,
    report: CurationReport | None = None,
) -> MaterializeResult:
    out = Path(out_root)
    (out / "meta").mkdir(parents=True, exist_ok=True)
    (out / "data").mkdir(parents=True, exist_ok=True)

    src_info = reader.info()
    kept = sorted(result.kept_episode_ids)
    episodes_meta: list[dict[str, object]] = []

    for new_ep, src_ep in enumerate(kept):
        cols = dict(reader.episode_columns(src_ep))
        if "episode_index" in cols:
            cols["episode_index"] = [new_ep] * len(cols["episode_index"])
        table = pa.table(cols)
        pq.write_table(table, out / "data" / f"episode_{new_ep:06d}.parquet")  # type: ignore[no-untyped-call]
        task = reader.episode_task(src_ep)
        length = len(next(iter(cols.values()))) if cols else 0
        episodes_meta.append(
            {
                "episode_index": new_ep,
                "length": length,
                "tasks": [task] if task is not None else [],
                "source_episode_index": src_ep,
            }
        )

    info = dict(src_info)
    info["total_episodes"] = len(kept)
    info["derived_from"] = reader.root
    info["derived_by"] = "lerobot-curate"
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=2))
    with (out / "meta" / "episodes.jsonl").open("w") as f:
        for row in episodes_meta:
            f.write(json.dumps(row) + "\n")

    report_path = out / "curation_report.json"
    if report is not None:
        report_path.write_text(report.to_json())
    else:
        report_path.write_text(json.dumps({"kept_episode_ids": kept}, indent=2))

    return MaterializeResult(out_root=str(out), n_episodes=len(kept), report_path=str(report_path))
