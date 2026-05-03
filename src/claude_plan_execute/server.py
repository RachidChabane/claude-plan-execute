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

from claude_plan_execute.loader import find_tasks_file, load_tasks_data
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

    Returns {"gates": [{"name": str, "size": int}]}, sorted by name.
    Empty list when the gates directory does not exist yet.
    """
    if not _SAFE_TASK_ID_RE.fullmatch(task_id):
        return JSONResponse({"error": "Invalid task id"}, status_code=400)

    defaults = load_tasks().get("defaults", {})
    plans_dir = defaults.get("plans_dir", "plans")
    gates_dir = work_dir / plans_dir / f"task-{task_id}" / "gates"

    if not gates_dir.exists():
        return JSONResponse({"gates": []})

    gates = [
        {"name": f.stem, "size": f.stat().st_size}
        for f in sorted(gates_dir.glob("*.txt"))
        if _SAFE_TASK_ID_RE.fullmatch(f.stem)
    ]
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
