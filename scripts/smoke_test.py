#!/usr/bin/env python
"""Lightweight smoke test for the Ticket Classifier service.

What it does:
1. Hit /health/live and /health/ready
2. Hit /version to confirm model metadata presence
3. Submit a classification request
4. Display colored per-field results + basic assertions

Exit codes:
 0 success, non-zero if any check fails.

Use:  python scripts/smoke_test.py --base-url https://<domain>
Defaults to http://localhost:8000 if not provided.
"""
from __future__ import annotations
import argparse
import sys
import time
import json
from typing import Any

import requests

RESET = "\x1b[0m"
COLORS = {
    "green": "\x1b[32m",
    "red": "\x1b[31m",
    "yellow": "\x1b[33m",
    "cyan": "\x1b[36m",
    "bold": "\x1b[1m",
}

def color(txt: str, c: str) -> str:
    return f"{COLORS.get(c,'')}{txt}{RESET}"


def status_line(label: str, ok: bool, detail: str = ""):
    symbol = "✔" if ok else "✘"
    c = "green" if ok else "red"
    line = f"{symbol} {label}"
    if detail:
        line += f" - {detail}"
    print(color(line, c))


def fetch_json(base: str, path: str, timeout: float = 10.0) -> tuple[bool, Any, int | None]:
    url = base.rstrip("/") + path
    try:
        r = requests.get(url, timeout=timeout)
        return r.status_code == 200, (r.json() if r.headers.get("content-type","" ).startswith("application/json") else None), r.status_code
    except Exception as e:  # noqa: BLE001
        return False, str(e), None


def post_json(base: str, path: str, payload: dict[str, Any], timeout: float = 15.0) -> tuple[bool, Any, int | None]:
    url = base.rstrip("/") + path
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        ok = r.status_code == 200
        data: Any
        try:
            data = r.json()
        except Exception:  # noqa: BLE001
            data = r.text
        return ok, data, r.status_code
    except Exception as e:  # noqa: BLE001
        return False, str(e), None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000", help="Base URL of service")
    parser.add_argument("--show-json", action="store_true", help="Print raw JSON bodies")
    args = parser.parse_args()
    base = args.base_url

    print(color(f"Smoke Test Target: {base}", "cyan"))

    overall_ok = True

    # 1. Liveness
    ok_live, live_json, live_code = fetch_json(base, "/health/live")
    status_line("/health/live", ok_live, f"code={live_code}")
    if args.show_json and isinstance(live_json, dict):
        print(json.dumps(live_json, indent=2))
    overall_ok &= ok_live

    # 2. Readiness (retry a couple times if not ready yet)
    ready_attempts = 0
    ok_ready = False
    ready_json = None
    ready_code = None
    while ready_attempts < 3 and not ok_ready:
        ok_ready, ready_json, ready_code = fetch_json(base, "/health/ready")
        if not ok_ready:
            time.sleep(1.0)
        ready_attempts += 1
    status_line("/health/ready", ok_ready, f"code={ready_code}")
    if args.show_json and isinstance(ready_json, dict):
        print(json.dumps(ready_json, indent=2))
    overall_ok &= ok_ready

    # 3. Version
    ok_version, version_json, version_code = fetch_json(base, "/version")
    model_loaded = bool(version_json and version_json.get("model_loaded")) if isinstance(version_json, dict) else False
    status_line("/version", ok_version and model_loaded, f"code={version_code} loaded={model_loaded}")
    if args.show_json and isinstance(version_json, dict):
        print(json.dumps(version_json, indent=2))
    overall_ok &= ok_version and model_loaded

    # 4. Classification
    sample = {
        "title": "Server outage",
        "description": "Primary API cluster unreachable all regions"
    }
    ok_cls, cls_json, cls_code = post_json(base, "/classify", sample)
    status_line("/classify", ok_cls, f"code={cls_code}")
    if ok_cls and isinstance(cls_json, dict):
        # Show fields nicely
        print(color("Prediction: ", "bold") + json.dumps({
            "priority": cls_json.get("predicted_priority"),
            "priority_conf": cls_json.get("priority_confidence"),
            "department": cls_json.get("predicted_department"),
            "department_conf": cls_json.get("department_confidence"),
        }, indent=2))
    else:
        print(cls_json)
    if args.show_json and isinstance(cls_json, dict):
        print(json.dumps(cls_json, indent=2))
    overall_ok &= ok_cls

    print()
    if overall_ok:
        print(color("ALL SMOKE TESTS PASSED", "green"))
        return 0
    print(color("SMOKE TESTS FAILED", "red"))
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
