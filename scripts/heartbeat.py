#!/usr/bin/env python3
"""Update the build session heartbeat in .lerobot-curate-progress.json.

Run after each shell step so a parallel session can detect this one is alive
(see bootstrap protocol session_lock semantics). Uses json read-modify-write,
never echo/sed, to avoid corrupting the state file.
"""

from __future__ import annotations

import datetime
import json
import os
import pathlib

STATE = pathlib.Path(__file__).resolve().parent.parent / ".lerobot-curate-progress.json"


def main() -> int:
    if not STATE.exists():
        print(f"[heartbeat] no state file at {STATE}", flush=True)
        return 0
    d = json.loads(STATE.read_text())
    now = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    lock = d.setdefault("session_lock", {})
    lock.setdefault("pid", os.getpid())
    lock.setdefault("started_at_utc", now)
    lock["last_heartbeat_utc"] = now
    STATE.write_text(json.dumps(d, indent=2))
    print(f"[heartbeat] {now} pid={lock['pid']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
