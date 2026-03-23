#!/usr/bin/env python3
"""
FastAPI backend for the claude-plan-execute workflow dashboard.

Serves the dashboard UI and provides API endpoints for task management.
Launches claude-plan-execute as a subprocess and streams output via WebSocket.

Usage:
    python scripts/workflow-server.py                        # Default port 8420
    python scripts/workflow-server.py --port 9000            # Custom port
    python scripts/workflow-server.py --tasks docs/tasks.yaml # Custom tasks file
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("Error: PyYAML required. pip install pyyaml", file=sys.stderr)
    sys.exit(1)

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
except ImportError:
    print("Error: FastAPI + uvicorn required. pip install fastapi uvicorn", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
DASHBOARD_PATH = SCRIPT_DIR / "workflow-dashboard.html"
CPE_SCRIPT = SCRIPT_DIR / "claude-plan-execute"

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="claude-plan-execute")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Global state
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


def find_tasks_file() -> Path | None:
    for candidate in [tasks_file, f"docs/{tasks_file}"]:
        p = work_dir / candidate
        if p.exists():
            return p
    return None


def load_tasks() -> dict:
    path = find_tasks_file()
    if not path:
        return {"tasks": []}
    with open(path) as f:
        data = yaml.safe_load(f) if path.suffix in (".yaml", ".yml") else json.load(f)
    return data


def load_state() -> dict:
    defaults = load_tasks().get("defaults", {})
    plans_dir = defaults.get("plans_dir", "plans")
    state_path = work_dir / plans_dir / "state.json"
    if state_path.exists():
        with open(state_path) as f:
            return json.load(f)
    return {"tasks": {}}


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
    data = load_tasks()
    return JSONResponse(data)


@app.get("/api/state")
async def get_state():
    return JSONResponse(load_state())


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

    cmd = [sys.executable, str(CPE_SCRIPT)]
    tf = find_tasks_file()
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

            # Also broadcast state update so UI refreshes
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
            # Kill the entire process group: cpe script + all claude child processes
            pgid = os.getpgid(active_process.pid)
            os.killpg(pgid, signal.SIGTERM)
            await broadcast_log("warn", "Sent SIGTERM to process group")
        except (ProcessLookupError, OSError):
            active_process.send_signal(signal.SIGTERM)
            await broadcast_log("warn", "Sent SIGTERM to process")

        # Give it 5 seconds, then force kill
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
    task_id = body.get("task_id")
    cmd = [sys.executable, str(CPE_SCRIPT)]
    tf = find_tasks_file()
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
        # Send initial state
        await ws.send_json({"type": "state", "data": load_state()})
        # Keep alive
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
