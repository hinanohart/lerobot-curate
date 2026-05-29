#!/usr/bin/env python3
"""Inject the metrics table from results/*.json into README between markers.

Keeps README numbers machine-generated (never hand-written).
"""

from __future__ import annotations

import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent


def render(m: dict) -> str:
    rows = [
        ("dedup recall (injected exact duplicates)", m["dedup_recall"]),
        ("dedup precision", m["dedup_precision"]),
        ("mislabel precision (injected label swaps)", m["mislabel_precision"]),
        ("mislabel recall", m["mislabel_recall"]),
        ("diversity monotone under redundancy (Spearman rho)", m["diversity_spearman_rho"]),
        ("clean-data false-positive rate", m["negative_fpr"]),
    ]
    lines = ["| metric | value |", "|---|---|"]
    lines += [f"| {name} | {val} |" for name, val in rows]
    env = m.get("env", {})
    lines.append("")
    lines.append(
        f"_mode: {m['dataset']['mode']}; python {env.get('python')}; "
        f"lerobot-curate {env.get('version')}; seed {env.get('seed')}_"
    )
    return "\n".join(lines)


def main() -> int:
    res = json.loads((ROOT / "results" / "v0.1.0a1_metrics.json").read_text())
    readme = ROOT / "README.md"
    text = readme.read_text()
    block = render(res)
    new = re.sub(
        r"<!-- METRICS:START -->.*<!-- METRICS:END -->",
        "<!-- METRICS:START -->\n" + block + "\n<!-- METRICS:END -->",
        text,
        flags=re.S,
    )
    readme.write_text(new)
    print("README metrics injected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
