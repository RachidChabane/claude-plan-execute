"""
FastAPI backend for the claude-plan-execute workflow dashboard.

Serves the dashboard UI and provides API endpoints for task management.
Launches claude-plan-execute as a subprocess and streams output via WebSocket.

Usage:
    claude-workflow-dashboard                          # Default port 8420
    claude-workflow-dashboard --port 9000              # Custom port
    claude-workflow-dashboard --tasks docs/tasks.yaml  # Custom tasks file
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import signal
import subprocess
import sys
from pathlib import Path

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
except ImportError:
    print("Error: FastAPI + uvicorn required. pip install 'claude-plan-execute[dashboard]'",
          file=sys.stderr)
    sys.exit(1)

import yaml

from claude_plan_execute.aggregations import (
    AggregationConfigError,
    AggregationRuntimeError,
    compute_aggregation,
    find_aggregations_file,
    load_aggregations,
)
from claude_plan_execute.loader import find_tasks_file, load_tasks_data
from claude_plan_execute.memory import (
    _drop_memory_file,
    memory_dir_for_cwd,
    parse_memory_frontmatter,
)
from claude_plan_execute.state import State

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PACKAGE_DIR = Path(__file__).resolve().parent
DASHBOARD_PATH = PACKAGE_DIR / "dashboard" / "index.html"

# Whitelist for task_id path segments. Matches int task ids (digits) and
# slug-like string ids (letters/digits/dash/underscore). Blocks `..`, `/`,
# `.`, spaces, and anything else that could escape the plans_dir tree.
_SAFE_TASK_ID_RE = re.compile(r"[A-Za-z0-9_-]+")

# T12: aggregation names use the same charset as task IDs and gate
# names. Aliased rather than re-declared so the relationship is
# documented in one place.
_SAFE_AGGREGATION_NAME_RE = _SAFE_TASK_ID_RE


def _resolve_cpe_script() -> str | None:
    """Locate the claude-plan-execute entry script.

    Lookup order:
      1. shutil.which("claude-plan-execute") — installed/symlinked on PATH.
      2. Repo-root script next to the source tree (dev-from-checkout).
      3. None — caller must error.
    """
    found = shutil.which("claude-plan-execute")
    if found:
        return found
    candidate = PACKAGE_DIR.parent.parent / "claude-plan-execute"
    if candidate.exists():
        return str(candidate)
    return None


CPE_SCRIPT = _resolve_cpe_script()

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="claude-plan-execute")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

work_dir: Path = Path.cwd()
tasks_file: str = "tasks.yaml"
active_process: subprocess.Popen | None = None
ws_clients: list[WebSocket] = []
config: dict = {
    "model": "opus",
    "effort": "max",
    "max_review_rounds": 3,
    "test_command": "pytest",
    "skip_permissions": False,
    "tasks_file": "tasks.yaml",
    "work_dir": ".",
}


def load_tasks() -> dict:
    """Locate + raw-read the tasks file. Returns {"tasks": []} on miss."""
    path = find_tasks_file(tasks_file, work_dir)
    if not path:
        return {"tasks": []}
    return load_tasks_data(path)


def load_state() -> dict:
    """Read state.json from the resolved plans_dir. {"tasks": {}} on miss."""
    defaults = load_tasks().get("defaults", {})
    plans_dir = defaults.get("plans_dir", "plans")
    state_path = work_dir / plans_dir / "state.json"
    return State(state_path).data


def load_services() -> list[dict]:
    """Read services.json (live snapshot) and tail each log_file.

    Returns [] when services.json is missing (no services declared,
    or the runner hasn't started any yet).
    """
    defaults = load_tasks().get("defaults", {})
    plans_dir = defaults.get("plans_dir", "plans")
    status_path = work_dir / plans_dir / "services.json"
    if not status_path.exists():
        return []
    try:
        with open(status_path, encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []
    services = payload.get("services") or []
    for s in services:
        s["log_tail"] = _tail_log_lines(s.get("log_file"), lines=200)
    return services


def _tail_log_lines(log_file: str | None, *, lines: int) -> list[str]:
    if not log_file:
        return []
    p = Path(log_file)
    if not p.exists():
        return []
    try:
        size = p.stat().st_size
        cap = 64 * 1024
        with open(p, "rb") as f:
            if size > cap:
                f.seek(size - cap)
            tail = f.read().decode("utf-8", errors="replace")
    except OSError:
        return []
    return tail.splitlines()[-lines:]


async def broadcast(msg: dict):
    dead = []
    for ws in ws_clients:
        try:
            await ws.send_json(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        ws_clients.remove(ws)


async def broadcast_log(level: str, message: str):
    await broadcast({"type": "log", "level": level, "message": message})


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    if DASHBOARD_PATH.exists():
        return FileResponse(DASHBOARD_PATH, media_type="text/html")
    return JSONResponse({"error": "Dashboard HTML not found"}, status_code=404)


@app.get("/api/tasks")
async def get_tasks():
    return JSONResponse(load_tasks())


@app.get("/api/state")
async def get_state():
    return JSONResponse(load_state())


@app.get("/api/resolved-config/{task_id}")
async def get_resolved_config(task_id: str):
    """Return the resolved-config.json audit artifact for a task.

    Path: {work_dir}/{plans_dir}/task-{task_id}/resolved-config.json.

    Returns 200 + JSON body when the file exists and parses; 400 when
    `task_id` fails the safe-segment regex (defends against `..`); 404
    when the artifact has not yet been generated; 500 if the on-disk
    file is unreadable or malformed (defensive).
    """
    if not _SAFE_TASK_ID_RE.fullmatch(task_id):
        return JSONResponse({"error": "Invalid task id"}, status_code=400)

    defaults = load_tasks().get("defaults", {})
    plans_dir = defaults.get("plans_dir", "plans")
    cfg_path = work_dir / plans_dir / f"task-{task_id}" / "resolved-config.json"

    if not cfg_path.exists():
        return JSONResponse({"error": "Not found"}, status_code=404)

    try:
        with open(cfg_path) as f:
            return JSONResponse(json.load(f))
    except (json.JSONDecodeError, OSError) as exc:
        return JSONResponse(
            {"error": f"Could not read artifact: {exc}"}, status_code=500
        )


@app.get("/api/gates/{task_id}")
async def get_gates(task_id: str):
    """List gate output files for a task.

    Returns {"gates": [{"name": str, "size": int, "kind": str}]},
    sorted by name. Empty list when the gates directory does not
    exist yet.

    T10 §D17: pending human gates are surfaced via their
    `<name>.request.md` file when neither `.approval` nor
    `.rejection` companion exists yet. Completed gates (any kind)
    are listed via their `<name>.txt` output. The `kind` field is
    "awaiting_human" or "completed" so the dashboard can decide
    which controls to render.
    """
    if not _SAFE_TASK_ID_RE.fullmatch(task_id):
        return JSONResponse({"error": "Invalid task id"}, status_code=400)

    defaults = load_tasks().get("defaults", {})
    plans_dir = defaults.get("plans_dir", "plans")
    gates_dir = work_dir / plans_dir / f"task-{task_id}" / "gates"

    if not gates_dir.exists():
        return JSONResponse({"gates": []})

    gates: list[dict] = []
    seen_names: set[str] = set()

    # Completed gates — `*.txt` files (every kind writes one
    # post-results).
    for f in sorted(gates_dir.glob("*.txt")):
        if not _SAFE_TASK_ID_RE.fullmatch(f.stem):
            continue
        gates.append({
            "name": f.stem,
            "size": f.stat().st_size,
            "kind": "completed",
        })
        seen_names.add(f.stem)

    # Pending human gates — `*.request.md` without a decision yet.
    for f in sorted(gates_dir.glob("*.request.md")):
        # Strip the `.request` suffix (Path.stem leaves `.md` part
        # already gone, so f.stem is `<name>.request`).
        name = f.stem[: -len(".request")] if f.stem.endswith(".request") else f.stem
        if not _SAFE_TASK_ID_RE.fullmatch(name):
            continue
        if name in seen_names:
            continue
        approval = gates_dir / f"{name}.approval"
        rejection = gates_dir / f"{name}.rejection"
        if approval.exists() or rejection.exists():
            continue
        gates.append({
            "name": name,
            "size": f.stat().st_size,
            "kind": "awaiting_human",
        })

    gates.sort(key=lambda g: g["name"])
    return JSONResponse({"gates": gates})


@app.get("/api/gates/{task_id}/{gate_name}")
async def get_gate_output(task_id: str, gate_name: str):
    """Return the text content of a single gate output file.

    Path: {plans_dir}/task-{task_id}/gates/{gate_name}.txt
    Returns {"content": str} on success; 400/404/500 on error.
    """
    if not _SAFE_TASK_ID_RE.fullmatch(task_id):
        return JSONResponse({"error": "Invalid task id"}, status_code=400)
    if not _SAFE_TASK_ID_RE.fullmatch(gate_name):
        return JSONResponse({"error": "Invalid gate name"}, status_code=400)

    defaults = load_tasks().get("defaults", {})
    plans_dir = defaults.get("plans_dir", "plans")
    gate_path = work_dir / plans_dir / f"task-{task_id}" / "gates" / f"{gate_name}.txt"

    if not gate_path.exists():
        return JSONResponse({"error": "Not found"}, status_code=404)

    try:
        return JSONResponse({"content": gate_path.read_text(encoding="utf-8")})
    except OSError as exc:
        return JSONResponse({"error": f"Could not read file: {exc}"}, status_code=500)


# T10 (v0.3 §2.4): write a sentinel for a pending human gate so the
# runner's poll loop wakes up and the gate transitions to PASS / FAIL.
# 4 KB note cap mirrors `state._VERBATIM_CAP_BYTES`. Path-traversal
# guards on both `task_id` and `name` (same as the GET endpoints).
# See plans/task-10/plan.md §D14.

_HUMAN_GATE_NOTE_CAP_BYTES = 4096


@app.post("/api/gates/{task_id}/{name}/decision")
async def post_gate_decision(task_id: str, name: str, body: dict):
    """Write a sentinel file for a pending human gate.

    Body: `{"decision": "approve"|"reject", "note": str}`.
    Path: `{plans_dir}/task-{task_id}/gates/{name}.{approval|rejection}`.
    Returns 200 `{"ok": true, "path": "..."}` on success; 400 on
    invalid decision / path-traversal; 500 on filesystem error.
    """
    if not _SAFE_TASK_ID_RE.fullmatch(task_id):
        return JSONResponse({"error": "Invalid task id"}, status_code=400)
    if not _SAFE_TASK_ID_RE.fullmatch(name):
        return JSONResponse({"error": "Invalid gate name"}, status_code=400)
    decision = body.get("decision") if isinstance(body, dict) else None
    if decision not in ("approve", "reject"):
        return JSONResponse(
            {"error": "decision must be 'approve' or 'reject'"},
            status_code=400,
        )
    # Defensive str() cast: a client sending {"note": 12} (int) would
    # otherwise crash `.encode("utf-8")` with AttributeError. Cost is
    # zero; coerces None / int / bool / list-via-str to text first.
    raw_note = body.get("note") if isinstance(body, dict) else ""
    note = str(raw_note) if raw_note is not None else ""
    note_capped = note.encode("utf-8")[:_HUMAN_GATE_NOTE_CAP_BYTES].decode(
        "utf-8", errors="ignore",
    )

    defaults = load_tasks().get("defaults", {})
    plans_dir = defaults.get("plans_dir", "plans")
    gates_dir = work_dir / plans_dir / f"task-{task_id}" / "gates"
    gates_dir.mkdir(parents=True, exist_ok=True)

    suffix = "approval" if decision == "approve" else "rejection"
    sentinel = gates_dir / f"{name}.{suffix}"
    try:
        sentinel.write_text(note_capped, encoding="utf-8")
    except OSError as exc:
        return JSONResponse(
            {"error": f"Could not write sentinel: {exc}"},
            status_code=500,
        )
    return JSONResponse({"ok": True, "path": str(sentinel)})


# T16 (v0.3): memory-curate gate sidecar findings + merge-suggestions
# endpoints. The merge-suggestions store lives at
# `{plans_dir}/memory-merge-suggestions.json`; per-memory file bodies
# live under `memory_dir_for_cwd(work_dir)`.

_SAFE_SUGGESTION_ID_RE = re.compile(r"[A-Za-z0-9_-]+")
_SAFE_MEMORY_NAME_RE = re.compile(r"[A-Za-z0-9_-]+")


@app.get("/api/gates/{task_id}/{gate_name}/findings")
async def get_gate_findings(task_id: str, gate_name: str):
    """Return the JSON sidecar (T11) for a gate. 404 when no
    sidecar was written (gate emits no appendix or the appendix
    was malformed).
    """
    if not _SAFE_TASK_ID_RE.fullmatch(task_id):
        return JSONResponse({"error": "Invalid task id"}, status_code=400)
    if not _SAFE_TASK_ID_RE.fullmatch(gate_name):
        return JSONResponse({"error": "Invalid gate name"}, status_code=400)
    defaults = load_tasks().get("defaults", {})
    plans_dir = defaults.get("plans_dir", "plans")
    json_path = (
        work_dir / plans_dir / f"task-{task_id}"
        / "gates" / f"{gate_name}.json"
    )
    if not json_path.exists():
        return JSONResponse({"error": "Not found"}, status_code=404)
    try:
        with open(json_path) as f:
            return JSONResponse(json.load(f))
    except (OSError, json.JSONDecodeError) as exc:
        return JSONResponse(
            {"error": f"Could not read findings: {exc}"},
            status_code=500,
        )


@app.get("/api/memory/merge-suggestions")
async def list_merge_suggestions():
    """Return the pending memory-merge suggestions for the
    current project.

    Path: `{plans_dir}/memory-merge-suggestions.json`. Returns
    200 + `{"suggestions": []}` when the file is absent (not
    404 — matches the aggregations endpoint's "absent vs
    broken" distinction). Returns 500 on JSON parse failure.

    Each entry: `{id, from, into, reason, status, ts}` —
    status is `pending`, `accepted`, or `declined`. The
    dashboard filters to status=pending by default.
    """
    defaults = load_tasks().get("defaults", {})
    plans_dir = defaults.get("plans_dir", "plans")
    path = work_dir / plans_dir / "memory-merge-suggestions.json"
    if not path.exists():
        return JSONResponse({"suggestions": []})
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return JSONResponse(
            {"error": f"Could not read suggestions: {exc}"},
            status_code=500,
        )
    if not isinstance(data, list):
        return JSONResponse(
            {"error": "suggestions file shape invalid"},
            status_code=500,
        )
    return JSONResponse({"suggestions": data})


@app.post("/api/memory/merge-suggestions/{sid}/{action}")
async def update_merge_suggestion(sid: str, action: str):
    """Accept or decline a pending merge suggestion.

    `action` ∈ {`accept`, `decline`}. Path-traversal-safe (sid
    matches the same regex as task ids).

    On `accept`: deletes the `from` memory file and removes its
    MEMORY.md line. The `into` memory survives unchanged. The
    operator is responsible for any content salvage — the
    dashboard's side-by-side view surfaces both bodies before
    they decide.

    On `decline`: the suggestion's status flips to `declined`
    and stays in the file (so the dashboard can show the
    history).

    Returns 400 on invalid id/action; 404 when no entry matches
    sid; 409 when already accepted/declined; 500 on filesystem
    error.
    """
    if not _SAFE_SUGGESTION_ID_RE.fullmatch(sid):
        return JSONResponse(
            {"error": "Invalid suggestion id"}, status_code=400,
        )
    if action not in ("accept", "decline"):
        return JSONResponse(
            {"error": "action must be 'accept' or 'decline'"},
            status_code=400,
        )
    defaults = load_tasks().get("defaults", {})
    plans_dir = defaults.get("plans_dir", "plans")
    path = work_dir / plans_dir / "memory-merge-suggestions.json"
    if not path.exists():
        return JSONResponse({"error": "Not found"}, status_code=404)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return JSONResponse(
            {"error": f"Could not read suggestions: {exc}"},
            status_code=500,
        )
    if not isinstance(data, list):
        return JSONResponse(
            {"error": "suggestions file shape invalid"},
            status_code=500,
        )
    target = next(
        (s for s in data
         if isinstance(s, dict) and s.get("id") == sid),
        None,
    )
    if target is None:
        return JSONResponse({"error": "Not found"}, status_code=404)
    if target.get("status") != "pending":
        return JSONResponse(
            {"error": f"Suggestion already {target.get('status')!r}"},
            status_code=409,
        )

    if action == "accept":
        memory_dir = memory_dir_for_cwd(work_dir)
        from_name = target.get("from", "")
        if not isinstance(from_name, str) or not from_name:
            return JSONResponse(
                {"error": "Suggestion missing valid 'from' name"},
                status_code=500,
            )
        try:
            _drop_memory_file(memory_dir, from_name)
        except OSError as exc:
            return JSONResponse(
                {"error": f"Drop failed: {exc}"}, status_code=500,
            )
        target["status"] = "accepted"
    else:
        target["status"] = "declined"

    try:
        path.write_text(
            json.dumps(data, indent=2) + "\n", encoding="utf-8",
        )
    except OSError as exc:
        return JSONResponse(
            {"error": f"Could not write suggestions: {exc}"},
            status_code=500,
        )
    return JSONResponse({"ok": True, "status": target["status"]})


@app.get("/api/memory/file/{name}")
async def get_memory_file(name: str):
    """Return the body of `<memory_dir>/<name>.md`. Used by the
    dashboard's side-by-side merge-suggestion view.

    Path-traversal-safe (name matches the slug regex). Returns
    `{"body": str, "frontmatter": dict}` on success. 404 when
    the file does not exist.
    """
    if not _SAFE_MEMORY_NAME_RE.fullmatch(name):
        return JSONResponse(
            {"error": "Invalid memory name"}, status_code=400,
        )
    memory_dir = memory_dir_for_cwd(work_dir)
    path = memory_dir / f"{name}.md"
    if not path.is_file():
        return JSONResponse({"error": "Not found"}, status_code=404)
    try:
        text = path.read_text(encoding="utf-8")
        fm, body = parse_memory_frontmatter(text)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        return JSONResponse(
            {"error": f"Could not read: {exc}"}, status_code=500,
        )
    return JSONResponse({"body": body, "frontmatter": fm})


# v0.3 task 12 (Plan §2.14): cross-task aggregation panel.
# Read-only post-hoc; walks {plans_dir}/task-*/gates/ on each request.
# aggregations.yaml lives at the work_dir root or under docs/, mirroring
# tasks.yaml discovery.

def _aggregations_plans_dir() -> Path:
    """Resolve the active plans_dir (tasks.yaml override OR 'plans')."""
    defaults = load_tasks().get("defaults", {})
    plans_dir_name = defaults.get("plans_dir", "plans")
    return work_dir / plans_dir_name


@app.get("/api/aggregations")
async def list_aggregations():
    """List configured aggregations.

    Returns 200 with `{"aggregations": []}` when aggregations.yaml is
    absent (NOT 404 — the dashboard distinguishes "absent" from
    "broken"). Returns 500 on YAML parse failure or 400 on per-entry
    shape error so the frontend can surface a toast instead of an
    empty panel.
    """
    af = find_aggregations_file(work_dir)
    if af is None:
        return JSONResponse({"aggregations": []})
    try:
        specs = load_aggregations(af)
    except AggregationRuntimeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)
    except AggregationConfigError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    return JSONResponse({
        "aggregations": [
            {"name": s.name, "gate": s.gate, "mode": s.mode}
            for s in specs
        ],
    })


@app.get("/api/aggregate/{name}")
async def get_aggregation(name: str):
    """Compute aggregation rows for the named aggregation.

    Returns 400 when `name` fails the safe-id regex (path-traversal
    guard). Returns 404 when aggregations.yaml is absent OR no entry
    matches `name`. Returns 500 on YAML parse failure or bad regex
    compile. Returns 400 on per-entry shape error.
    """
    if not _SAFE_AGGREGATION_NAME_RE.fullmatch(name):
        return JSONResponse(
            {"error": "Invalid aggregation name"}, status_code=400,
        )
    af = find_aggregations_file(work_dir)
    if af is None:
        return JSONResponse({"error": "Not found"}, status_code=404)
    try:
        specs = load_aggregations(af)
    except AggregationRuntimeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)
    except AggregationConfigError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    spec = next((s for s in specs if s.name == name), None)
    if spec is None:
        return JSONResponse({"error": "Not found"}, status_code=404)
    plans_dir = _aggregations_plans_dir()
    try:
        rows = compute_aggregation(spec, plans_dir)
    except AggregationRuntimeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)
    return JSONResponse({
        "name": spec.name,
        "gate": spec.gate,
        "mode": spec.mode,
        "rows": rows,
    })


# v0.3 task 8 (Plan §D6.2): services endpoints. Reuse
# _SAFE_TASK_ID_RE for the name guard — the pattern is identical
# and ServiceSpec.name (Plan §D1.2) enforces the same character
# whitelist at load time. The endpoint check is defense-in-depth.

@app.get("/api/services")
async def get_services():
    """Return [{name, status, pid, lifetime, log_file, log_tail,
    crash_count, depends_on, command, ready_check}].

    Empty list when services.json is missing.
    """
    return JSONResponse({"services": load_services()})


@app.post("/api/services/{name}/restart")
async def restart_service(name: str):
    """Send SIGTERM to the service's recorded PID.

    Trade-off (Plan §D2.6): the runner's next poll_health() sees
    the dead process and auto-restarts via the SAME crash-counter
    codepath. So a restart click = one crash-count tick.
    """
    if not _SAFE_TASK_ID_RE.fullmatch(name):
        return JSONResponse(
            {"error": "Invalid service name"}, status_code=400,
        )
    services = load_services()
    target = next((s for s in services if s.get("name") == name), None)
    if target is None:
        return JSONResponse({"error": "Not found"}, status_code=404)
    pid = target.get("pid")
    if not pid:
        return JSONResponse(
            {"error": "Service is not running"}, status_code=409,
        )
    try:
        pgid = os.getpgid(pid)
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, OSError) as exc:
        # Process already dead — no-op success.
        return JSONResponse({"ok": True, "note": str(exc)})
    return JSONResponse({"ok": True, "pid": pid})


@app.get("/api/config")
async def get_config():
    return JSONResponse(config)


@app.post("/api/config")
async def set_config(body: dict):
    config.update(body)
    return JSONResponse({"ok": True})


@app.post("/api/run")
async def run_tasks(body: dict = {}):
    global active_process

    if active_process and active_process.poll() is None:
        return JSONResponse({"error": "Already running"}, status_code=409)

    if not CPE_SCRIPT:
        return JSONResponse(
            {"error": "claude-plan-execute script not found. Install via pip "
                      "or symlink onto PATH."},
            status_code=500,
        )

    cmd = [sys.executable, CPE_SCRIPT]
    tf = find_tasks_file(tasks_file, work_dir)
    if tf:
        cmd.extend(["--tasks", str(tf)])

    if body.get("task_id"):
        cmd.extend(["--task", str(body["task_id"])])
    if body.get("skip_to"):
        cmd.extend(["--skip-to", body["skip_to"]])

    model = body.get("model", config.get("model", "opus"))
    effort = body.get("effort", config.get("effort", "max"))
    cmd.extend(["--model", model, "--effort", effort])

    if body.get("skip_permissions", config.get("skip_permissions")):
        cmd.append("--skip-permissions")

    cmd.append("--no-color")

    await broadcast_log("info", f"Launching: {' '.join(cmd[-6:])}")

    active_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(work_dir),
        bufsize=1,
        preexec_fn=os.setsid,  # New process group so we can kill the entire tree
    )

    asyncio.create_task(stream_output(active_process))
    return JSONResponse({"ok": True, "pid": active_process.pid})


async def stream_output(proc: subprocess.Popen):
    """Read process output line by line and broadcast via WebSocket."""
    loop = asyncio.get_event_loop()
    try:
        while True:
            line = await loop.run_in_executor(None, proc.stdout.readline)
            if not line and proc.poll() is not None:
                break
            line = line.rstrip()
            if not line:
                continue

            level = "dim"
            if "ERROR" in line or "Error" in line:
                level = "error"
            elif "APPROVED" in line or "complete" in line.lower() or "COMPLETE" in line:
                level = "success"
            elif "WARNING" in line or "NEEDS REVISION" in line:
                level = "warn"
            elif "Phase" in line or "TASK" in line or "Review round" in line:
                level = "info"

            await broadcast_log(level, line)
            await broadcast({"type": "state", "data": load_state()})

    except Exception as e:
        await broadcast_log("error", f"Stream error: {e}")
    finally:
        rc = proc.wait()
        await broadcast_log("success" if rc == 0 else "error", f"Process exited with code {rc}")
        await broadcast({"type": "state", "data": load_state()})


@app.post("/api/stop")
async def stop_execution():
    global active_process
    if active_process and active_process.poll() is None:
        try:
            pgid = os.getpgid(active_process.pid)
            os.killpg(pgid, signal.SIGTERM)
            await broadcast_log("warn", "Sent SIGTERM to process group")
        except (ProcessLookupError, OSError):
            active_process.send_signal(signal.SIGTERM)
            await broadcast_log("warn", "Sent SIGTERM to process")

        await asyncio.sleep(5)
        if active_process and active_process.poll() is None:
            try:
                pgid = os.getpgid(active_process.pid)
                os.killpg(pgid, signal.SIGKILL)
                await broadcast_log("error", "Force-killed process group (SIGKILL)")
            except (ProcessLookupError, OSError):
                pass

        return JSONResponse({"ok": True})
    return JSONResponse({"error": "Nothing running"}, status_code=404)


@app.post("/api/reset")
async def reset_tasks(body: dict = {}):
    if not CPE_SCRIPT:
        return JSONResponse(
            {"error": "claude-plan-execute script not found."},
            status_code=500,
        )
    task_id = body.get("task_id")
    cmd = [sys.executable, CPE_SCRIPT]
    tf = find_tasks_file(tasks_file, work_dir)
    if tf:
        cmd.extend(["--tasks", str(tf)])
    if task_id:
        cmd.extend(["--reset", str(task_id)])
    else:
        cmd.extend(["--reset"])

    subprocess.run(cmd, cwd=str(work_dir), capture_output=True)
    await broadcast({"type": "state", "data": load_state()})
    await broadcast_log("info", f"Reset {'task ' + str(task_id) if task_id else 'all tasks'}")
    return JSONResponse({"ok": True})


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    ws_clients.append(ws)
    try:
        await ws.send_json({"type": "state", "data": load_state()})
        while True:
            try:
                await asyncio.wait_for(ws.receive_text(), timeout=30)
            except asyncio.TimeoutError:
                await ws.send_json({"type": "ping"})
    except WebSocketDisconnect:
        pass
    finally:
        if ws in ws_clients:
            ws_clients.remove(ws)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global work_dir, tasks_file

    parser = argparse.ArgumentParser(description="Workflow dashboard server")
    parser.add_argument("--port", type=int, default=8420)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--tasks", default="tasks.yaml")
    parser.add_argument("--dir", default=".")
    args = parser.parse_args()

    work_dir = Path(args.dir).resolve()
    tasks_file = args.tasks
    config["tasks_file"] = args.tasks
    config["work_dir"] = str(work_dir)

    print(f"\n  claude-plan-execute dashboard")
    print(f"  http://{args.host}:{args.port}")
    print(f"  Working dir: {work_dir}")
    print(f"  Tasks file:  {tasks_file}\n")

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
