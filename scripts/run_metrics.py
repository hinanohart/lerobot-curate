#!/usr/bin/env python3
"""Compute algorithm-correctness metrics and write results/v0.1.0a1_metrics.json.

Synthetic, deterministic, CPU-only. Numbers here describe selection-algorithm
correctness, NOT downstream policy performance (out of scope; see README).
"""

from __future__ import annotations

import datetime
import json
import pathlib
import platform
import sys

from lerobot_curate._version import __version__
from lerobot_curate.metrics import compute_metrics

ROOT = pathlib.Path(__file__).resolve().parent.parent


def main(seed: int = 0) -> int:
    m = compute_metrics(seed)
    out = ROOT / "results" / "v0.1.0a1_metrics.json"
    out.parent.mkdir(exist_ok=True)
    m["env"] = {
        "hw": platform.machine(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "date": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d"),
        "seed": seed,
        "version": __version__,
    }
    m["results_path"] = str(out)
    out.write_text(json.dumps(m, indent=2))
    print(json.dumps(m, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 0))
