"""Tests for the FastAPI dashboard server."""
from __future__ import annotations

import json as _json
import os
import signal
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from claude_plan_execute import loader, server
from claude_plan_execute.server import app


_DEFAULT_CONFIG = {
    "model": "opus",
    "effort": "max",
    "max_review_rounds": 3,
    "test_command": "pytest",
    "skip_permissions": False,
    "tasks_file": "tasks.yaml",
    "work_dir": ".",
}


@pytest.fixture(autouse=True)
def _reset_server_globals(tmp_path):
    """Reset server module globals between tests so order doesn't matter."""
    server.work_dir = tmp_path
    server.tasks_file = "tasks.yaml"
    server.active_process = None
    server.ws_clients = []
    server.config = dict(_DEFAULT_CONFIG)
    yield
    server.active_process = None
    server.ws_clients = []


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def fake_cpe_script(monkeypatch):
    """Stub _resolve_cpe_script result so tests don't depend on PATH."""
    monkeypatch.setattr(server, "CPE_SCRIPT", "/fake/bin/claude-plan-execute")
    return "/fake/bin/claude-plan-execute"


def _write_tasks_yaml(work_dir: Path, body: str | None = None):
    body = body or """\
version: 1
defaults:
  plans_dir: plans
tasks:
  - id: 1
    title: "First"
    description: "First desc"
  - id: 2
    title: "Second"
    description: "Second desc"
"""
    p = work_dir / "tasks.yaml"
    p.write_text(body)
    return p


def _write_state_json(work_dir: Path, body: dict | None = None):
    body = body or {"tasks": {"1": {"status": "done"}}}
    plans = work_dir / "plans"
    plans.mkdir(exist_ok=True)
    p = plans / "state.json"
    p.write_text(_json.dumps(body))
    return p


class FakeRunningPopen:
    """Popen stub for tests where the child must appear to be running."""
    def __init__(self, cmd=None, **kwargs):
        self.cmd = cmd or []
        self.kwargs = kwargs
        self.pid = 12345
        self.stdout = MagicMock()
        self.stdout.readline = MagicMock(return_value="")
        self._poll_returns: int | None = None

    def poll(self):
        return self._poll_returns

    def wait(self, timeout=None):
        if self._poll_returns is None:
            self._poll_returns = 0
        return self._poll_returns

    def terminate(self):
        self._poll_returns = -15

    def send_signal(self, sig):
        pass

    def kill(self):
        self._poll_returns = -9


class FakeExitedPopen(FakeRunningPopen):
    """Popen stub that reports already-exited from poll()."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._poll_returns = 0


def _patch_popen(monkeypatch, popen_cls=FakeExitedPopen):
    """Install a Popen stub. Returns the captured-state dict."""
    captured = {}

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        instance = popen_cls(cmd, **kwargs)
        captured["instance"] = instance
        return instance

    monkeypatch.setattr(server.subprocess, "Popen", fake_popen)
    return captured


# ---------------------------------------------------------------------------
# A. Dashboard HTML serving
# ---------------------------------------------------------------------------

def test_root_serves_dashboard_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "<!DOCTYPE html>" in r.text or "<html" in r.text


def test_root_returns_404_when_dashboard_missing(client, monkeypatch):
    monkeypatch.setattr(server, "DASHBOARD_PATH", Path("/no/such/dashboard.html"))
    r = client.get("/")
    assert r.status_code == 404


def test_root_response_content_type_is_text_html(client):
    r = client.get("/")
    assert r.headers["content-type"].startswith("text/html")


# ---------------------------------------------------------------------------
# B. /api/tasks
# ---------------------------------------------------------------------------

def test_get_tasks_returns_parsed_yaml_data(client, tmp_path):
    _write_tasks_yaml(tmp_path)
    r = client.get("/api/tasks")
    assert r.status_code == 200
    data = r.json()
    assert len(data["tasks"]) == 2
    assert data["tasks"][0]["title"] == "First"


def test_get_tasks_returns_empty_when_no_file(client):
    r = client.get("/api/tasks")
    assert r.status_code == 200
    assert r.json() == {"tasks": []}


def test_get_tasks_finds_docs_subdirectory(client, tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "tasks.yaml").write_text(
        "version: 1\ntasks: [{id: 1, title: T, description: D}]\n"
    )
    r = client.get("/api/tasks")
    assert len(r.json()["tasks"]) == 1


def test_get_tasks_respects_custom_tasks_filename(client, tmp_path):
    server.tasks_file = "alt.yaml"
    (tmp_path / "alt.yaml").write_text(
        "version: 1\ntasks: [{id: 9, title: A, description: B}]\n"
    )
    r = client.get("/api/tasks")
    assert r.json()["tasks"][0]["id"] == 9


def test_get_tasks_returns_empty_dict_for_empty_yaml(client, tmp_path):
    (tmp_path / "tasks.yaml").write_text("")
    r = client.get("/api/tasks")
    # load_tasks_data returns {} for empty yaml; load_tasks does not wrap
    # — the endpoint returns whatever load_tasks_data returned.
    assert r.json() == {}


def test_get_tasks_handles_json_extension(client, tmp_path):
    server.tasks_file = "tasks.json"
    (tmp_path / "tasks.json").write_text(
        '{"version": 1, "tasks": [{"id": 1, "title": "JT", "description": "D"}]}'
    )
    r = client.get("/api/tasks")
    assert r.json()["tasks"][0]["title"] == "JT"


# ---------------------------------------------------------------------------
# C. /api/state
# ---------------------------------------------------------------------------

def test_get_state_returns_state_data_when_file_exists(client, tmp_path):
    _write_tasks_yaml(tmp_path)
    _write_state_json(tmp_path, {"tasks": {"1": {"status": "done"}}})
    r = client.get("/api/state")
    assert r.json() == {
        "schema_version": 3,
        "tasks": {
            "1": {"status": "done", "attempts": [], "current_attempt": 1},
        },
    }


def test_get_state_returns_empty_tasks_dict_when_no_state_file(client, tmp_path):
    _write_tasks_yaml(tmp_path)
    r = client.get("/api/state")
    assert r.json() == {"schema_version": 3, "tasks": {}}


def test_get_state_uses_plans_dir_from_defaults(client, tmp_path):
    _write_tasks_yaml(tmp_path, """\
version: 1
defaults:
  plans_dir: custom-plans
tasks:
  - id: 1
    title: T
    description: D
""")
    custom = tmp_path / "custom-plans"
    custom.mkdir()
    (custom / "state.json").write_text(
        _json.dumps({"tasks": {"1": {"status": "implementing"}}})
    )
    r = client.get("/api/state")
    assert r.json() == {
        "schema_version": 3,
        "tasks": {
            "1": {"status": "implementing", "attempts": [], "current_attempt": 1},
        },
    }


def test_get_state_falls_back_to_default_plans_dir_when_no_defaults_block(
    client, tmp_path,
):
    (tmp_path / "tasks.yaml").write_text(
        "version: 1\ntasks: [{id: 1, title: T, description: D}]\n"
    )
    _write_state_json(tmp_path, {"tasks": {"1": {"status": "approved"}}})
    r = client.get("/api/state")
    assert r.json() == {
        "schema_version": 3,
        "tasks": {
            "1": {"status": "approved", "attempts": [], "current_attempt": 1},
        },
    }


# ---------------------------------------------------------------------------
# C2. /api/resolved-config (task 5)
# ---------------------------------------------------------------------------


def _write_resolved_config(work_dir: Path, task_id, body: dict | None = None,
                            plans_dir: str = "plans"):
    body = body or {"model": {"value": "opus", "from": "task"}}
    d = work_dir / plans_dir / f"task-{task_id}"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "resolved-config.json"
    p.write_text(_json.dumps(body))
    return p


def test_get_resolved_config_returns_json_when_file_exists(client, tmp_path):
    _write_tasks_yaml(tmp_path)
    _write_resolved_config(tmp_path, 1)
    r = client.get("/api/resolved-config/1")
    assert r.status_code == 200
    assert r.json() == {"model": {"value": "opus", "from": "task"}}


def test_get_resolved_config_returns_404_when_file_missing(client, tmp_path):
    _write_tasks_yaml(tmp_path)
    r = client.get("/api/resolved-config/999")
    assert r.status_code == 404


def test_get_resolved_config_uses_resolved_plans_dir_from_defaults(client, tmp_path):
    _write_tasks_yaml(tmp_path, """\
version: 1
defaults:
  plans_dir: custom-plans
tasks:
  - id: 1
    title: T
    description: D
""")
    _write_resolved_config(
        tmp_path, 1,
        body={"model": {"value": "haiku", "from": "task"}},
        plans_dir="custom-plans",
    )
    r = client.get("/api/resolved-config/1")
    assert r.status_code == 200
    assert r.json()["model"]["value"] == "haiku"


def test_get_resolved_config_falls_back_to_default_plans_dir_when_no_defaults_block(
    client, tmp_path,
):
    (tmp_path / "tasks.yaml").write_text(
        "version: 1\ntasks: [{id: 1, title: T, description: D}]\n"
    )
    _write_resolved_config(tmp_path, 1)
    r = client.get("/api/resolved-config/1")
    assert r.status_code == 200


def test_get_resolved_config_handles_string_task_id(client, tmp_path):
    _write_tasks_yaml(tmp_path)
    _write_resolved_config(tmp_path, "abc-feature")
    r = client.get("/api/resolved-config/abc-feature")
    assert r.status_code == 200


def test_get_resolved_config_rejects_dot_dot_path_traversal(client, tmp_path):
    _write_tasks_yaml(tmp_path)
    # `..` is rejected at one of two layers: the test client normalises
    # the URL and hits no route (404), or a raw client preserves it and
    # the safe-id regex blocks it (400). Either way, no traversal occurs.
    r = client.get("/api/resolved-config/..")
    assert r.status_code in (400, 404)


def test_get_resolved_config_rejects_url_encoded_path_traversal(client, tmp_path):
    """%2F decodes to '/' but FastAPI path params already block it; the safe-id
    regex is the second line of defence on whatever does pass through."""
    _write_tasks_yaml(tmp_path)
    r = client.get("/api/resolved-config/.%2E")
    assert r.status_code in (400, 404)


def test_get_resolved_config_rejects_task_id_with_space(client, tmp_path):
    _write_tasks_yaml(tmp_path)
    r = client.get("/api/resolved-config/foo%20bar")
    assert r.status_code == 400


def test_get_resolved_config_returns_500_when_file_is_malformed_json(client, tmp_path):
    _write_tasks_yaml(tmp_path)
    d = tmp_path / "plans" / "task-1"
    d.mkdir(parents=True)
    (d / "resolved-config.json").write_text("not valid json {{{")
    r = client.get("/api/resolved-config/1")
    assert r.status_code == 500


# ---------------------------------------------------------------------------
# D. /api/config
# ---------------------------------------------------------------------------

def test_get_config_returns_default_config(client):
    r = client.get("/api/config")
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "opus"
    assert body["effort"] == "max"


def test_post_config_merges_supplied_keys(client):
    r = client.post("/api/config", json={"model": "sonnet"})
    assert r.status_code == 200
    assert server.config["model"] == "sonnet"


def test_post_config_preserves_unrelated_keys(client):
    server.config["test_command"] = "pytest tests/"
    client.post("/api/config", json={"model": "haiku"})
    assert server.config["test_command"] == "pytest tests/"


def test_post_config_adds_new_keys(client):
    client.post("/api/config", json={"new_key": "new_val"})
    assert server.config["new_key"] == "new_val"


# ---------------------------------------------------------------------------
# E. /api/run
# ---------------------------------------------------------------------------

def test_post_run_returns_409_when_already_running(client, fake_cpe_script):
    server.active_process = FakeRunningPopen()
    r = client.post("/api/run", json={})
    assert r.status_code == 409


def test_post_run_returns_500_when_cpe_script_unresolved(client, monkeypatch):
    monkeypatch.setattr(server, "CPE_SCRIPT", None)
    r = client.post("/api/run", json={})
    assert r.status_code == 500


def test_post_run_launches_subprocess_with_default_flags(
    client, monkeypatch, fake_cpe_script, tmp_path,
):
    captured = _patch_popen(monkeypatch)
    _write_tasks_yaml(tmp_path)
    r = client.post("/api/run", json={})
    assert r.status_code == 200
    cmd = captured["cmd"]
    assert cmd[0] == sys.executable
    assert cmd[1] == fake_cpe_script
    assert "--model" in cmd and cmd[cmd.index("--model") + 1] == "opus"
    assert "--effort" in cmd and cmd[cmd.index("--effort") + 1] == "max"
    assert "--no-color" in cmd


def test_post_run_includes_task_id_when_provided(
    client, monkeypatch, fake_cpe_script,
):
    captured = _patch_popen(monkeypatch)
    client.post("/api/run", json={"task_id": 7})
    cmd = captured["cmd"]
    assert "--task" in cmd
    assert cmd[cmd.index("--task") + 1] == "7"


def test_post_run_includes_skip_to_when_provided(
    client, monkeypatch, fake_cpe_script,
):
    captured = _patch_popen(monkeypatch)
    client.post("/api/run", json={"task_id": 3, "skip_to": "implement"})
    cmd = captured["cmd"]
    assert "--skip-to" in cmd
    assert cmd[cmd.index("--skip-to") + 1] == "implement"


def test_post_run_overrides_model_from_body(
    client, monkeypatch, fake_cpe_script,
):
    captured = _patch_popen(monkeypatch)
    client.post("/api/run", json={"model": "sonnet"})
    cmd = captured["cmd"]
    assert cmd[cmd.index("--model") + 1] == "sonnet"


def test_post_run_overrides_effort_from_body(
    client, monkeypatch, fake_cpe_script,
):
    captured = _patch_popen(monkeypatch)
    client.post("/api/run", json={"effort": "low"})
    cmd = captured["cmd"]
    assert cmd[cmd.index("--effort") + 1] == "low"


def test_post_run_passes_skip_permissions_from_body(
    client, monkeypatch, fake_cpe_script,
):
    captured = _patch_popen(monkeypatch)
    client.post("/api/run", json={"skip_permissions": True})
    assert "--skip-permissions" in captured["cmd"]


def test_post_run_passes_skip_permissions_from_config(
    client, monkeypatch, fake_cpe_script,
):
    captured = _patch_popen(monkeypatch)
    server.config["skip_permissions"] = True
    client.post("/api/run", json={})
    assert "--skip-permissions" in captured["cmd"]


def test_post_run_includes_tasks_file_when_found(
    client, monkeypatch, fake_cpe_script, tmp_path,
):
    captured = _patch_popen(monkeypatch)
    p = _write_tasks_yaml(tmp_path)
    client.post("/api/run", json={})
    cmd = captured["cmd"]
    assert "--tasks" in cmd
    assert cmd[cmd.index("--tasks") + 1] == str(p)


def test_post_run_omits_tasks_file_when_not_found(
    client, monkeypatch, fake_cpe_script,
):
    captured = _patch_popen(monkeypatch)
    client.post("/api/run", json={})
    assert "--tasks" not in captured["cmd"]


def test_post_run_appends_no_color_flag(
    client, monkeypatch, fake_cpe_script,
):
    captured = _patch_popen(monkeypatch)
    client.post("/api/run", json={})
    assert "--no-color" in captured["cmd"]


def test_post_run_returns_pid_in_response(
    client, monkeypatch, fake_cpe_script,
):
    _patch_popen(monkeypatch)
    r = client.post("/api/run", json={})
    assert r.json()["pid"] == 12345


def test_post_run_uses_setsid_for_process_group(
    client, monkeypatch, fake_cpe_script,
):
    # Load-bearing pin: /api/stop's killpg(pgid, SIGTERM) only reaches the
    # tree if /api/run's Popen used preexec_fn=os.setsid. The parallel
    # pin in test_runner.py (test_run_claude_does_NOT_set_preexec_fn)
    # asserts the run_claude side; this one asserts the server side.
    captured = _patch_popen(monkeypatch)
    client.post("/api/run", json={})
    assert captured["kwargs"].get("preexec_fn") is os.setsid


def test_post_run_invokes_cpe_script_via_python_executable(
    client, monkeypatch, fake_cpe_script,
):
    captured = _patch_popen(monkeypatch)
    client.post("/api/run", json={})
    assert captured["cmd"][0] == sys.executable
    assert captured["cmd"][1] == fake_cpe_script


def test_post_run_sets_cwd_to_work_dir(
    client, monkeypatch, fake_cpe_script, tmp_path,
):
    captured = _patch_popen(monkeypatch)
    client.post("/api/run", json={})
    assert captured["kwargs"]["cwd"] == str(tmp_path)


# ---------------------------------------------------------------------------
# F. /api/stop
# ---------------------------------------------------------------------------

async def _no_sleep(_seconds):
    pass


def test_post_stop_returns_404_when_nothing_running(client):
    r = client.post("/api/stop")
    assert r.status_code == 404


def test_post_stop_returns_404_when_active_process_already_exited(client):
    server.active_process = FakeExitedPopen()
    r = client.post("/api/stop")
    assert r.status_code == 404


def test_post_stop_kills_active_process_group(client, monkeypatch):
    monkeypatch.setattr(server.asyncio, "sleep", _no_sleep)
    fake = FakeRunningPopen()
    server.active_process = fake
    killed = []
    monkeypatch.setattr(server.os, "getpgid", lambda pid: pid + 100)

    def _killpg(pid, sig):
        killed.append((pid, sig))
        if sig == signal.SIGTERM:
            fake.terminate()

    monkeypatch.setattr(server.os, "killpg", _killpg)
    r = client.post("/api/stop")
    assert r.status_code == 200
    assert killed[0] == (12445, signal.SIGTERM)


def test_post_stop_force_kills_when_sigterm_does_not_exit(client, monkeypatch):
    monkeypatch.setattr(server.asyncio, "sleep", _no_sleep)
    fake = FakeRunningPopen()
    fake.terminate = lambda: None
    server.active_process = fake
    killed = []
    monkeypatch.setattr(server.os, "getpgid", lambda pid: pid + 100)
    monkeypatch.setattr(server.os, "killpg", lambda pid, sig: killed.append(sig))
    r = client.post("/api/stop")
    assert r.status_code == 200
    assert signal.SIGTERM in killed
    assert signal.SIGKILL in killed


def test_post_stop_falls_back_to_signal_when_killpg_raises(client, monkeypatch):
    monkeypatch.setattr(server.asyncio, "sleep", _no_sleep)
    fake = FakeRunningPopen()
    server.active_process = fake
    sent = []

    def _send_signal(sig):
        sent.append(sig)
        fake.terminate()

    fake.send_signal = _send_signal

    def _raise_pl(_pid):
        raise ProcessLookupError()

    monkeypatch.setattr(server.os, "getpgid", _raise_pl)

    r = client.post("/api/stop")
    assert sent == [signal.SIGTERM]
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# G. /api/reset
# ---------------------------------------------------------------------------

def test_post_reset_returns_500_when_cpe_script_unresolved(client, monkeypatch):
    monkeypatch.setattr(server, "CPE_SCRIPT", None)
    r = client.post("/api/reset", json={})
    assert r.status_code == 500


def test_post_reset_shells_out_to_cli_with_reset_flag(
    client, monkeypatch, fake_cpe_script,
):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    monkeypatch.setattr(server.subprocess, "run", fake_run)
    r = client.post("/api/reset", json={})
    assert r.status_code == 200
    assert captured["cmd"][0] == sys.executable
    assert captured["cmd"][1] == fake_cpe_script
    assert "--reset" in captured["cmd"]


def test_post_reset_includes_task_id_when_provided(
    client, monkeypatch, fake_cpe_script,
):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    monkeypatch.setattr(server.subprocess, "run", fake_run)
    client.post("/api/reset", json={"task_id": 5})
    cmd = captured["cmd"]
    idx = cmd.index("--reset")
    assert cmd[idx + 1] == "5"


def test_post_reset_omits_task_id_when_not_provided(
    client, monkeypatch, fake_cpe_script,
):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    monkeypatch.setattr(server.subprocess, "run", fake_run)
    client.post("/api/reset", json={})
    cmd = captured["cmd"]
    idx = cmd.index("--reset")
    # Nothing follows --reset (it is the last argument, or nothing else uses
    # the next slot for a task id).
    assert idx == len(cmd) - 1 or not cmd[idx + 1].isdigit()


def test_post_reset_includes_tasks_file_arg_when_found(
    client, monkeypatch, fake_cpe_script, tmp_path,
):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    monkeypatch.setattr(server.subprocess, "run", fake_run)
    p = _write_tasks_yaml(tmp_path)
    client.post("/api/reset", json={})
    cmd = captured["cmd"]
    assert "--tasks" in cmd
    assert cmd[cmd.index("--tasks") + 1] == str(p)


def test_post_reset_returns_ok(client, monkeypatch, fake_cpe_script):
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    monkeypatch.setattr(server.subprocess, "run", fake_run)
    r = client.post("/api/reset", json={})
    assert r.status_code == 200
    assert r.json() == {"ok": True}


# ---------------------------------------------------------------------------
# H. /ws (WebSocket)
# ---------------------------------------------------------------------------

def test_websocket_accepts_connection(client, tmp_path):
    _write_tasks_yaml(tmp_path)
    with client.websocket_connect("/ws") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "state"


def test_websocket_sends_initial_state_on_connect(client, tmp_path):
    _write_tasks_yaml(tmp_path)
    _write_state_json(tmp_path, {"tasks": {"1": {"status": "done"}}})
    with client.websocket_connect("/ws") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "state"
        assert msg["data"] == {
            "schema_version": 3,
            "tasks": {
                "1": {"status": "done", "attempts": [], "current_attempt": 1},
            },
        }


def test_websocket_appends_to_clients_list(client, tmp_path):
    _write_tasks_yaml(tmp_path)
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        assert len(server.ws_clients) == 1
    assert server.ws_clients == []


# ---------------------------------------------------------------------------
# I. Loader integration
# ---------------------------------------------------------------------------

def test_loader_find_tasks_file_uses_default_filename_in_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "tasks.yaml").write_text("tasks: []\n")
    p = loader.find_tasks_file()
    assert p is not None
    assert p.name == "tasks.yaml"


def test_loader_find_tasks_file_with_custom_filename(tmp_path):
    (tmp_path / "alt.yaml").write_text("tasks: []\n")
    p = loader.find_tasks_file("alt.yaml", start_dir=tmp_path)
    assert p == tmp_path / "alt.yaml"


def test_loader_find_tasks_file_with_custom_start_dir(tmp_path):
    (tmp_path / "tasks.yaml").write_text("tasks: []\n")
    p = loader.find_tasks_file(start_dir=tmp_path)
    assert p == tmp_path / "tasks.yaml"


def test_loader_find_tasks_file_searches_docs_subdir(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "tasks.yaml").write_text("tasks: []\n")
    p = loader.find_tasks_file(start_dir=tmp_path)
    assert p == docs / "tasks.yaml"


def test_loader_find_tasks_file_returns_none_when_not_found(tmp_path):
    assert loader.find_tasks_file(start_dir=tmp_path) is None


def test_loader_load_tasks_data_parses_yaml(tmp_path):
    p = tmp_path / "x.yaml"
    p.write_text("version: 1\ntasks: [{id: 1}]\n")
    data = loader.load_tasks_data(p)
    assert data["version"] == 1
    assert data["tasks"][0]["id"] == 1


def test_loader_load_tasks_data_parses_json(tmp_path):
    p = tmp_path / "x.json"
    p.write_text('{"version": 1, "tasks": [{"id": 2}]}')
    data = loader.load_tasks_data(p)
    assert data["tasks"][0]["id"] == 2


def test_loader_load_tasks_data_returns_empty_dict_for_empty_yaml(tmp_path):
    p = tmp_path / "x.yaml"
    p.write_text("")
    assert loader.load_tasks_data(p) == {}


def test_loader_load_tasks_data_does_NOT_validate(tmp_path):
    """Unlike load_tasks_file, this helper accepts malformed input without sys.exit."""
    p = tmp_path / "bad.yaml"
    p.write_text("not a tasks dict — just a string")
    assert isinstance(loader.load_tasks_data(p), str)


# ---------------------------------------------------------------------------
# J. Gates API
# ---------------------------------------------------------------------------

class TestGatesAPI:
    """Tests for GET /api/gates/{task_id} and /api/gates/{task_id}/{gate_name}."""

    def _write_gate(self, work_dir: Path, task_id: str, name: str, content: str = "output\n"):
        d = work_dir / "plans" / f"task-{task_id}" / "gates"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{name}.txt").write_text(content)

    # ── List endpoint ──

    def test_list_empty_when_no_gates_dir(self, client, tmp_path):
        _write_tasks_yaml(tmp_path)
        r = client.get("/api/gates/1")
        assert r.status_code == 200
        assert r.json() == {"gates": []}

    def test_list_returns_sorted_files(self, client, tmp_path):
        _write_tasks_yaml(tmp_path)
        self._write_gate(tmp_path, "1", "tests", "passed\n")
        self._write_gate(tmp_path, "1", "lint", "clean\n")
        r = client.get("/api/gates/1")
        assert r.status_code == 200
        names = [g["name"] for g in r.json()["gates"]]
        assert names == ["lint", "tests"]

    def test_list_includes_file_size(self, client, tmp_path):
        _write_tasks_yaml(tmp_path)
        self._write_gate(tmp_path, "1", "tests", "hello\n")
        r = client.get("/api/gates/1")
        assert r.json()["gates"][0]["size"] == 6  # "hello\n"

    def test_list_rejects_path_traversal_task_id(self, client, tmp_path):
        r = client.get("/api/gates/..%2Fetc")
        assert r.status_code in (400, 422)

    def test_list_rejects_dotted_task_id(self, client, tmp_path):
        r = client.get("/api/gates/task.one")
        assert r.status_code == 400

    # ── Content endpoint ──

    def test_get_output_returns_content(self, client, tmp_path):
        _write_tasks_yaml(tmp_path)
        self._write_gate(tmp_path, "2", "tests", "all green\n")
        r = client.get("/api/gates/2/tests")
        assert r.status_code == 200
        assert r.json()["content"] == "all green\n"

    def test_get_output_404_when_missing(self, client, tmp_path):
        _write_tasks_yaml(tmp_path)
        r = client.get("/api/gates/1/nonexistent")
        assert r.status_code == 404

    def test_get_output_rejects_dotted_gate_name(self, client, tmp_path):
        r = client.get("/api/gates/1/foo.bar")
        assert r.status_code == 400

    def test_get_output_rejects_path_traversal_task_id(self, client, tmp_path):
        r = client.get("/api/gates/bad%2Fid/tests")
        assert r.status_code in (400, 404, 422)

    def test_get_output_rejects_path_traversal_gate_name(self, client, tmp_path):
        r = client.get("/api/gates/1/bad%2Fname")
        assert r.status_code in (400, 404, 422)

    def test_get_output_uses_custom_plans_dir(self, client, tmp_path):
        _write_tasks_yaml(tmp_path, """\
version: 1
defaults:
  plans_dir: artifacts
tasks:
  - id: 1
    title: T
    description: D
""")
        d = tmp_path / "artifacts" / "task-1" / "gates"
        d.mkdir(parents=True)
        (d / "tests.txt").write_text("ok\n")
        r = client.get("/api/gates/1/tests")
        assert r.status_code == 200
        assert r.json()["content"] == "ok\n"


class TestGatesAPIHumanRequestExtension:
    """T10 §D17 — list endpoint surfaces pending human gates via
    `<name>.request.md` files when no decision sentinel exists.
    """

    def test_get_gates_lists_pending_human_gate_via_request_md(
        self, client, tmp_path,
    ):
        _write_tasks_yaml(tmp_path)
        d = tmp_path / "plans" / "task-1" / "gates"
        d.mkdir(parents=True)
        (d / "review.request.md").write_text("# please review\n")
        r = client.get("/api/gates/1")
        assert r.status_code == 200
        names = [g["name"] for g in r.json()["gates"]]
        assert "review" in names
        kinds = {g["name"]: g["kind"] for g in r.json()["gates"]}
        assert kinds["review"] == "awaiting_human"

    def test_get_gates_omits_human_gate_with_decision_in_place(
        self, client, tmp_path,
    ):
        _write_tasks_yaml(tmp_path)
        d = tmp_path / "plans" / "task-1" / "gates"
        d.mkdir(parents=True)
        (d / "review.request.md").write_text("# please review\n")
        (d / "review.approval").write_text("ok\n")
        r = client.get("/api/gates/1")
        assert r.status_code == 200
        # The pending entry is suppressed; if a `.txt` exists it would
        # show up as "completed", but we deliberately did not write
        # one yet.
        names = [g["name"] for g in r.json()["gates"]]
        assert "review" not in names

    def test_get_gates_existing_txt_listing_unchanged_for_completed_gates(
        self, client, tmp_path,
    ):
        _write_tasks_yaml(tmp_path)
        d = tmp_path / "plans" / "task-1" / "gates"
        d.mkdir(parents=True)
        (d / "tests.txt").write_text("passed\n")
        r = client.get("/api/gates/1")
        assert r.status_code == 200
        gates = r.json()["gates"]
        assert len(gates) == 1
        assert gates[0]["name"] == "tests"
        assert gates[0]["kind"] == "completed"


class TestGateDecisionAPI:
    """T10 §D14 — POST /api/gates/{task_id}/{name}/decision."""

    def test_post_decision_approve_writes_approval_sentinel(
        self, client, tmp_path,
    ):
        _write_tasks_yaml(tmp_path)
        r = client.post(
            "/api/gates/1/review/decision",
            json={"decision": "approve", "note": "lgtm"},
        )
        assert r.status_code == 200
        assert r.json()["ok"] is True
        sentinel = tmp_path / "plans" / "task-1" / "gates" / "review.approval"
        assert sentinel.exists()
        assert sentinel.read_text() == "lgtm"

    def test_post_decision_reject_writes_rejection_sentinel(
        self, client, tmp_path,
    ):
        _write_tasks_yaml(tmp_path)
        r = client.post(
            "/api/gates/1/review/decision",
            json={"decision": "reject", "note": "no go"},
        )
        assert r.status_code == 200
        sentinel = tmp_path / "plans" / "task-1" / "gates" / "review.rejection"
        assert sentinel.exists()
        assert sentinel.read_text() == "no go"

    def test_post_decision_invalid_decision_returns_400(
        self, client, tmp_path,
    ):
        _write_tasks_yaml(tmp_path)
        r = client.post(
            "/api/gates/1/review/decision",
            json={"decision": "abstain", "note": ""},
        )
        assert r.status_code == 400

    def test_post_decision_path_traversal_task_id_returns_400(
        self, client, tmp_path,
    ):
        _write_tasks_yaml(tmp_path)
        r = client.post(
            "/api/gates/bad.id/review/decision",
            json={"decision": "approve", "note": ""},
        )
        assert r.status_code == 400

    def test_post_decision_path_traversal_gate_name_returns_400(
        self, client, tmp_path,
    ):
        _write_tasks_yaml(tmp_path)
        r = client.post(
            "/api/gates/1/bad.name/decision",
            json={"decision": "approve", "note": ""},
        )
        assert r.status_code == 400

    def test_post_decision_caps_note_at_4kb(self, client, tmp_path):
        _write_tasks_yaml(tmp_path)
        big_note = "x" * (10 * 1024)  # 10 KB; cap is 4 KB
        r = client.post(
            "/api/gates/1/review/decision",
            json={"decision": "approve", "note": big_note},
        )
        assert r.status_code == 200
        sentinel = tmp_path / "plans" / "task-1" / "gates" / "review.approval"
        encoded = sentinel.read_text().encode("utf-8")
        assert len(encoded) == 4096

    def test_post_decision_empty_body_writes_empty_sentinel(
        self, client, tmp_path,
    ):
        _write_tasks_yaml(tmp_path)
        r = client.post(
            "/api/gates/1/review/decision",
            json={"decision": "approve"},  # no `note` key
        )
        assert r.status_code == 200
        sentinel = tmp_path / "plans" / "task-1" / "gates" / "review.approval"
        assert sentinel.exists()
        assert sentinel.read_text() == ""

    def test_post_decision_uses_custom_plans_dir(
        self, client, tmp_path,
    ):
        _write_tasks_yaml(tmp_path, """\
version: 1
defaults:
  plans_dir: artifacts
tasks:
  - id: 1
    title: T
    description: D
""")
        r = client.post(
            "/api/gates/1/review/decision",
            json={"decision": "approve", "note": "ok"},
        )
        assert r.status_code == 200
        sentinel = (
            tmp_path / "artifacts" / "task-1" / "gates" / "review.approval"
        )
        assert sentinel.exists()

    def test_post_decision_creates_gates_dir_if_missing(
        self, client, tmp_path,
    ):
        _write_tasks_yaml(tmp_path)
        # No plans/task-1 directory yet — endpoint should create it.
        r = client.post(
            "/api/gates/1/review/decision",
            json={"decision": "approve", "note": "ok"},
        )
        assert r.status_code == 200
        assert (
            tmp_path / "plans" / "task-1" / "gates" / "review.approval"
        ).exists()


# ---------------------------------------------------------------------------
# v0.3 task 8: services endpoints (Plan §D6 / §D10.6)
# ---------------------------------------------------------------------------


def _write_services_json(work_dir: Path, services: list[dict]) -> Path:
    """Write a services.json fixture matching the schema_version=1 shape."""
    p = work_dir / "plans" / "services.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(_json.dumps({
        "schema_version": 1, "services": services,
    }))
    return p


class TestGetServices:
    def test_get_services_returns_empty_when_no_status_file(self, client, tmp_path):
        """Plan §D6.1: missing services.json yields []."""
        _write_tasks_yaml(tmp_path)
        r = client.get("/api/services")
        assert r.status_code == 200
        assert r.json() == {"services": []}

    def test_get_services_returns_status_from_services_json(self, client, tmp_path):
        """Plan §D6.1: the live snapshot is returned verbatim."""
        _write_tasks_yaml(tmp_path)
        _write_services_json(tmp_path, [{
            "name": "dev", "command": "npm run dev",
            "lifetime": "run",
            "ready_check": {"kind": "tcp", "port": 5173},
            "log_file": None, "status": "running",
            "pid": 12345, "crash_count": 0, "depends_on": [],
        }])
        r = client.get("/api/services")
        assert r.status_code == 200
        body = r.json()
        assert len(body["services"]) == 1
        svc = body["services"][0]
        assert svc["name"] == "dev"
        assert svc["pid"] == 12345
        assert svc["status"] == "running"

    def test_get_services_includes_log_tail_when_log_file_present(
        self, client, tmp_path,
    ):
        """Plan §D6.1: each service gets a log_tail (last 200 lines)."""
        _write_tasks_yaml(tmp_path)
        log_path = tmp_path / "plans" / "dev.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("line-a\nline-b\nline-c\n")
        _write_services_json(tmp_path, [{
            "name": "dev", "command": "x", "lifetime": "run",
            "ready_check": {"kind": "tcp", "port": 80},
            "log_file": str(log_path), "status": "running",
            "pid": 1, "crash_count": 0, "depends_on": [],
        }])
        r = client.get("/api/services")
        body = r.json()
        assert body["services"][0]["log_tail"] == ["line-a", "line-b", "line-c"]

    def test_get_services_handles_missing_log_file_gracefully(
        self, client, tmp_path,
    ):
        """Plan §D6.1: a non-existent log_file yields log_tail=[]."""
        _write_tasks_yaml(tmp_path)
        _write_services_json(tmp_path, [{
            "name": "dev", "command": "x", "lifetime": "run",
            "ready_check": {"kind": "tcp", "port": 80},
            "log_file": str(tmp_path / "missing.log"),
            "status": "running",
            "pid": 1, "crash_count": 0, "depends_on": [],
        }])
        r = client.get("/api/services")
        body = r.json()
        assert body["services"][0]["log_tail"] == []


class TestPostServiceRestart:
    def test_post_service_restart_sends_sigterm_to_pid(
        self, client, tmp_path, monkeypatch,
    ):
        """Plan §D6.2: restart sends SIGTERM to the recorded pid's
        process group (mirrors the runner's kill ladder)."""
        _write_tasks_yaml(tmp_path)
        _write_services_json(tmp_path, [{
            "name": "dev", "command": "x", "lifetime": "run",
            "ready_check": {"kind": "tcp", "port": 80},
            "log_file": None, "status": "running",
            "pid": 12345, "crash_count": 0, "depends_on": [],
        }])
        calls: list[tuple] = []

        def fake_killpg(pgid, sig):
            calls.append((pgid, sig))

        monkeypatch.setattr(server.os, "getpgid", lambda pid: pid)
        monkeypatch.setattr(server.os, "killpg", fake_killpg)

        r = client.post("/api/services/dev/restart")
        assert r.status_code == 200
        assert calls == [(12345, signal.SIGTERM)]
        assert r.json()["pid"] == 12345

    def test_post_service_restart_returns_404_when_unknown_name(
        self, client, tmp_path,
    ):
        """Plan §D6.2: unknown service name → 404."""
        _write_tasks_yaml(tmp_path)
        _write_services_json(tmp_path, [{
            "name": "dev", "command": "x", "lifetime": "run",
            "ready_check": {"kind": "tcp", "port": 80},
            "log_file": None, "status": "running",
            "pid": 1, "crash_count": 0, "depends_on": [],
        }])
        r = client.post("/api/services/missing/restart")
        assert r.status_code == 404

    def test_post_service_restart_returns_409_when_pid_null(
        self, client, tmp_path,
    ):
        """Plan §D6.2: a service with no pid (stopped) → 409."""
        _write_tasks_yaml(tmp_path)
        _write_services_json(tmp_path, [{
            "name": "dev", "command": "x", "lifetime": "run",
            "ready_check": {"kind": "tcp", "port": 80},
            "log_file": None, "status": "stopped",
            "pid": None, "crash_count": 0, "depends_on": [],
        }])
        r = client.post("/api/services/dev/restart")
        assert r.status_code == 409

    def test_post_service_restart_rejects_invalid_name_chars(
        self, client, tmp_path,
    ):
        """Plan §D6.2: defense-in-depth name regex rejects bad chars
        (also blocked by the ServiceSpec.name pattern at load time)."""
        _write_tasks_yaml(tmp_path)
        # Path traversal attempt → 400 (per regex) or 404/422 from
        # FastAPI routing depending on the raw segment shape.
        r = client.post("/api/services/bad..name/restart")
        assert r.status_code in (400, 404, 422)


# ---------------------------------------------------------------------------
# v0.3 task 12: aggregations endpoints (Plan §2.14)
# ---------------------------------------------------------------------------


def _write_aggregations_yaml(work_dir: Path, body: str) -> Path:
    """Write aggregations.yaml at work_dir root."""
    p = work_dir / "aggregations.yaml"
    p.write_text(body)
    return p


def _write_gate_txt(work_dir: Path, task_id: str, gate: str,
                    content: str, plans_dir: str = "plans") -> Path:
    d = work_dir / plans_dir / f"task-{task_id}" / "gates"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{gate}.txt"
    p.write_text(content, encoding="utf-8")
    return p


def _write_gate_json(work_dir: Path, task_id: str, gate: str,
                     payload: dict, plans_dir: str = "plans") -> Path:
    d = work_dir / plans_dir / f"task-{task_id}" / "gates"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{gate}.json"
    p.write_text(_json.dumps(payload, indent=2) + "\n",
                 encoding="utf-8")
    return p


class TestAggregationsAPI:
    """T12: GET /api/aggregations and /api/aggregate/{name}."""

    # ── List endpoint (GET /api/aggregations) ──

    def test_list_returns_empty_when_no_yaml(self, client, tmp_path):
        """No aggregations.yaml → 200 with []. Dashboard hides panel."""
        _write_tasks_yaml(tmp_path)
        r = client.get("/api/aggregations")
        assert r.status_code == 200
        assert r.json() == {"aggregations": []}

    def test_list_returns_configured_entries(self, client, tmp_path):
        """Each entry surfaces name + gate + mode."""
        _write_tasks_yaml(tmp_path)
        _write_aggregations_yaml(tmp_path, """\
- name: untrans
  gate: untranslated-strings
  extract: "missing key: (\\\\S+)"
- name: sec-sev
  gate: security-auditor
  field: severity
- name: sec-cat-sev
  gate: security-auditor
  fields: [category, severity]
""")
        r = client.get("/api/aggregations")
        assert r.status_code == 200
        body = r.json()
        assert [a["name"] for a in body["aggregations"]] == \
            ["untrans", "sec-sev", "sec-cat-sev"]
        assert [a["mode"] for a in body["aggregations"]] == \
            ["regex", "field", "fields"]

    def test_list_finds_docs_subdirectory(self, client, tmp_path):
        """aggregations.yaml under docs/ is discovered."""
        _write_tasks_yaml(tmp_path)
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "aggregations.yaml").write_text(
            "- name: x\n  gate: tests\n  extract: pass\n"
        )
        r = client.get("/api/aggregations")
        assert r.status_code == 200
        assert len(r.json()["aggregations"]) == 1

    def test_list_returns_500_on_malformed_yaml(self, client, tmp_path):
        """Bad YAML → 500 with helpful error."""
        _write_tasks_yaml(tmp_path)
        _write_aggregations_yaml(tmp_path, "not: valid: yaml: [\n")
        r = client.get("/api/aggregations")
        assert r.status_code == 500
        assert "error" in r.json()

    def test_list_returns_400_on_top_level_dict(self, client, tmp_path):
        """Top-level dict (not list) → 400 (per §6.D4)."""
        _write_tasks_yaml(tmp_path)
        _write_aggregations_yaml(tmp_path,
            "aggregations:\n  - name: x\n    gate: tests\n    extract: 'foo'\n")
        r = client.get("/api/aggregations")
        assert r.status_code == 400

    def test_list_returns_400_on_duplicate_names(self, client, tmp_path):
        _write_tasks_yaml(tmp_path)
        _write_aggregations_yaml(tmp_path, """\
- name: dup
  gate: tests
  extract: foo
- name: dup
  gate: tests
  field: bar
""")
        r = client.get("/api/aggregations")
        assert r.status_code == 400
        assert "duplicate" in r.json()["error"].lower()

    def test_list_returns_400_when_neither_extract_nor_field_set(
        self, client, tmp_path,
    ):
        """'neither' case (per §6.D3 + advisor #8)."""
        _write_tasks_yaml(tmp_path)
        _write_aggregations_yaml(tmp_path,
            "- name: x\n  gate: tests\n")
        r = client.get("/api/aggregations")
        assert r.status_code == 400
        assert "exactly one" in r.json()["error"].lower()

    def test_list_returns_400_when_gate_has_path_traversal(
        self, client, tmp_path,
    ):
        """gate validated with safe-id regex (advisor #1)."""
        _write_tasks_yaml(tmp_path)
        _write_aggregations_yaml(tmp_path,
            "- name: x\n  gate: '../etc'\n  extract: 'foo'\n")
        r = client.get("/api/aggregations")
        assert r.status_code == 400

    def test_list_returns_empty_for_empty_yaml(self, client, tmp_path):
        """yaml.safe_load returns None for empty file."""
        _write_tasks_yaml(tmp_path)
        _write_aggregations_yaml(tmp_path, "")
        r = client.get("/api/aggregations")
        assert r.status_code == 200
        assert r.json() == {"aggregations": []}

    # ── Per-name endpoint (GET /api/aggregate/{name}) ──

    def test_aggregate_returns_404_when_no_yaml(self, client, tmp_path):
        """Missing aggregations.yaml → 404 on per-name fetch."""
        _write_tasks_yaml(tmp_path)
        r = client.get("/api/aggregate/anything")
        assert r.status_code == 404

    def test_aggregate_returns_404_when_name_unknown(self, client, tmp_path):
        _write_tasks_yaml(tmp_path)
        _write_aggregations_yaml(tmp_path,
            "- name: x\n  gate: tests\n  extract: 'foo'\n")
        r = client.get("/api/aggregate/missing")
        assert r.status_code == 404

    def test_aggregate_returns_400_on_invalid_name_chars(self, client, tmp_path):
        """Path-traversal in URL segment is blocked at the regex."""
        _write_tasks_yaml(tmp_path)
        r = client.get("/api/aggregate/foo.bar")
        assert r.status_code == 400

    # === SPEC TEST (a): endpoint walks task dirs correctly ===

    def test_endpoint_walks_task_dirs(self, client, tmp_path):
        """Tallies span every task-* directory under plans_dir."""
        _write_tasks_yaml(tmp_path)
        _write_aggregations_yaml(tmp_path,
            "- name: agg\n  gate: foo\n  extract: 'XXX'\n")
        for tid in ("1", "2", "3"):
            _write_gate_txt(tmp_path, tid, "foo", "XXX\nXXX\n")
        r = client.get("/api/aggregate/agg")
        assert r.status_code == 200
        rows = r.json()["rows"]
        assert len(rows) == 1
        assert rows[0]["finding"] == "XXX"
        assert rows[0]["total_count"] == 6
        assert rows[0]["per_task"] == {"1": 2, "2": 2, "3": 2}

    # === SPEC TEST (b): Mode A regex extracts from gate output text ===

    def test_mode_a_regex_extracts_capture_group(self, client, tmp_path):
        _write_tasks_yaml(tmp_path)
        # Single-quoted YAML preserves backslashes literally; the
        # Python source needs `\\S` so the YAML value is `(\S+)`,
        # which then compiles as the regex non-whitespace charclass.
        _write_aggregations_yaml(tmp_path,
            "- name: untrans\n  gate: untranslated-strings\n"
            "  extract: 'missing key: (\\S+)'\n")
        _write_gate_txt(tmp_path, "1", "untranslated-strings",
                        "missing key: foo\nmissing key: bar\nok\n")
        r = client.get("/api/aggregate/untrans")
        rows = r.json()["rows"]
        # Sorted by total_count desc; tiebreaker = finding asc
        assert [(row["finding"], row["total_count"]) for row in rows] == \
            [("bar", 1), ("foo", 1)]

    def test_mode_a_falls_back_to_full_match_when_no_capture_group(
        self, client, tmp_path,
    ):
        """Per §6.D6: zero-group regex returns m.group(0)."""
        _write_tasks_yaml(tmp_path)
        _write_aggregations_yaml(tmp_path,
            "- name: x\n  gate: foo\n  extract: 'WARN'\n")
        _write_gate_txt(tmp_path, "1", "foo",
                        "WARN: bad\nWARN: also bad\nok\n")
        r = client.get("/api/aggregate/x")
        rows = r.json()["rows"]
        assert rows == [{"finding": "WARN", "total_count": 2,
                         "per_task": {"1": 2}}]

    # === SPEC TEST (c): Mode B reads .json and iterates findings[] ===

    def test_mode_b_reads_json_and_iterates_findings(self, client, tmp_path):
        _write_tasks_yaml(tmp_path)
        _write_aggregations_yaml(tmp_path,
            "- name: sev\n  gate: security-auditor\n  field: severity\n")
        _write_gate_json(tmp_path, "1", "security-auditor", {
            "findings": [
                {"severity": "critical"},
                {"severity": "high"},
                {"severity": "critical"},
            ],
        })
        r = client.get("/api/aggregate/sev")
        rows = r.json()["rows"]
        assert rows == [
            {"finding": "critical", "total_count": 2, "per_task": {"1": 2}},
            {"finding": "high",     "total_count": 1, "per_task": {"1": 1}},
        ]

    # === SPEC TEST (d): Mode B with missing field buckets as "(unknown)" ===

    def test_mode_b_missing_field_buckets_as_unknown(self, client, tmp_path):
        _write_tasks_yaml(tmp_path)
        _write_aggregations_yaml(tmp_path,
            "- name: sev\n  gate: foo\n  field: severity\n")
        _write_gate_json(tmp_path, "1", "foo", {
            "findings": [
                {"severity": "critical"},
                {"category": "ACL"},   # severity missing
                {"severity": None},    # severity null → also (unknown)
            ],
        })
        r = client.get("/api/aggregate/sev")
        rows = r.json()["rows"]
        finds = {r_["finding"]: r_["total_count"] for r_ in rows}
        assert finds == {"(unknown)": 2, "critical": 1}

    # === SPEC TEST (e): composite `fields` joined with ":" ===

    def test_mode_b_composite_fields_joined_with_colon(self, client, tmp_path):
        _write_tasks_yaml(tmp_path)
        _write_aggregations_yaml(tmp_path,
            "- name: cat-sev\n  gate: sec\n"
            "  fields: [category, severity]\n")
        _write_gate_json(tmp_path, "1", "sec", {
            "findings": [
                {"category": "ACL", "severity": "critical"},
                {"category": "ACL", "severity": "critical"},
                {"category": "Secrets", "severity": "high"},
            ],
        })
        r = client.get("/api/aggregate/cat-sev")
        rows = r.json()["rows"]
        finds = {r_["finding"]: r_["total_count"] for r_ in rows}
        assert finds == {"ACL:critical": 2, "Secrets:high": 1}

    # === Additional: composite with one missing field → "ACL:(unknown)" ===
    # (per §2.B / §6.D5: per-slot replacement, NOT whole-bucket)

    def test_mode_b_composite_with_missing_field_uses_per_slot_unknown(
        self, client, tmp_path,
    ):
        _write_tasks_yaml(tmp_path)
        _write_aggregations_yaml(tmp_path,
            "- name: cat-sev\n  gate: sec\n"
            "  fields: [category, severity]\n")
        _write_gate_json(tmp_path, "1", "sec", {
            "findings": [
                {"category": "ACL"},  # severity missing
                {"severity": "critical"},  # category missing
            ],
        })
        r = client.get("/api/aggregate/cat-sev")
        rows = r.json()["rows"]
        finds = {r_["finding"]: r_["total_count"] for r_ in rows}
        assert finds == {"ACL:(unknown)": 1, "(unknown):critical": 1}

    # === SPEC TEST (f): per-task count and total align ===

    def test_per_task_count_and_total_align(self, client, tmp_path):
        _write_tasks_yaml(tmp_path)
        _write_aggregations_yaml(tmp_path,
            "- name: sev\n  gate: foo\n  field: severity\n")
        _write_gate_json(tmp_path, "1", "foo", {
            "findings": [{"severity": "critical"}, {"severity": "critical"}]})
        _write_gate_json(tmp_path, "2", "foo", {
            "findings": [{"severity": "critical"}]})
        r = client.get("/api/aggregate/sev")
        row = r.json()["rows"][0]
        assert row["finding"] == "critical"
        assert row["total_count"] == 3
        assert row["per_task"] == {"1": 2, "2": 1}
        assert row["total_count"] == sum(row["per_task"].values())

    # === SPEC TEST (g): missing gate dir silently skipped ===

    def test_missing_gate_dir_silently_skipped(self, client, tmp_path):
        _write_tasks_yaml(tmp_path)
        _write_aggregations_yaml(tmp_path,
            "- name: x\n  gate: foo\n  extract: 'BUG'\n")
        # task-1 has gates/foo.txt; task-2 has NO gates dir at all
        _write_gate_txt(tmp_path, "1", "foo", "BUG\n")
        (tmp_path / "plans" / "task-2").mkdir(parents=True)
        r = client.get("/api/aggregate/x")
        assert r.status_code == 200
        rows = r.json()["rows"]
        assert rows == [{"finding": "BUG", "total_count": 1,
                         "per_task": {"1": 1}}]

    def test_mode_b_missing_json_silently_skipped(self, client, tmp_path):
        """Mode B counterpart to test_missing_gate_dir_silently_skipped:
        a task whose gates dir lacks <gate>.json is skipped without
        crashing. The .txt path is unaffected (this aggregation is
        Mode B, so .txt isn't read at all)."""
        _write_tasks_yaml(tmp_path)
        _write_aggregations_yaml(tmp_path,
            "- name: sev\n  gate: foo\n  field: severity\n")
        _write_gate_json(tmp_path, "1", "foo", {
            "findings": [{"severity": "critical"}]})
        # task-2 has gates/ but no foo.json
        (tmp_path / "plans" / "task-2" / "gates").mkdir(parents=True)
        r = client.get("/api/aggregate/sev")
        rows = r.json()["rows"]
        assert rows == [{"finding": "critical", "total_count": 1,
                         "per_task": {"1": 1}}]

    # === SPEC TEST (h): bad regex returns 500 ===

    def test_bad_regex_returns_500(self, client, tmp_path):
        _write_tasks_yaml(tmp_path)
        _write_aggregations_yaml(tmp_path,
            "- name: x\n  gate: foo\n  extract: '(?P<unclosed'\n")
        # Need at least one task dir for the walk to attempt compile.
        (tmp_path / "plans" / "task-1" / "gates").mkdir(parents=True)
        r = client.get("/api/aggregate/x")
        assert r.status_code == 500
        assert "regex" in r.json()["error"].lower()

    # === SPEC TEST (i): both extract and field set returns 400 ===

    def test_both_extract_and_field_returns_400(self, client, tmp_path):
        _write_tasks_yaml(tmp_path)
        _write_aggregations_yaml(tmp_path, """\
- name: x
  gate: foo
  extract: 'bar'
  field: baz
""")
        r = client.get("/api/aggregate/x")
        assert r.status_code == 400
        assert "exactly one" in r.json()["error"].lower()

    # ── Additional invariant tests ──

    def test_rows_sorted_by_total_count_desc_then_finding_asc(
        self, client, tmp_path,
    ):
        """Per §6.D7: sort key = (-total_count, finding)."""
        _write_tasks_yaml(tmp_path)
        _write_aggregations_yaml(tmp_path,
            "- name: x\n  gate: foo\n  field: tag\n")
        _write_gate_json(tmp_path, "1", "foo", {
            "findings": [
                {"tag": "z"}, {"tag": "z"}, {"tag": "z"},
                {"tag": "a"}, {"tag": "a"},
                {"tag": "m"}, {"tag": "m"},
            ],
        })
        r = client.get("/api/aggregate/x")
        rows = r.json()["rows"]
        # z=3, then a=2 (asc tiebreak before m=2), then m=2.
        assert [(rw["finding"], rw["total_count"]) for rw in rows] == \
            [("z", 3), ("a", 2), ("m", 2)]

    def test_per_task_keys_are_naturally_sorted(self, client, tmp_path):
        """Per §6.D7 / advisor #4: numeric IDs sort as ints, not strings."""
        _write_tasks_yaml(tmp_path)
        _write_aggregations_yaml(tmp_path,
            "- name: x\n  gate: foo\n  field: tag\n")
        for tid in ("1", "2", "10"):
            _write_gate_json(tmp_path, tid, "foo",
                {"findings": [{"tag": "shared"}]})
        r = client.get("/api/aggregate/x")
        per_task = r.json()["rows"][0]["per_task"]
        # JSON preserves insertion order; the loader emits naturally-sorted.
        assert list(per_task.keys()) == ["1", "2", "10"]

    def test_mode_b_with_extra_keys_in_payload_is_ignored(
        self, client, tmp_path,
    ):
        """T11 case-4 sidecar (with `summary` key) is read transparently."""
        _write_tasks_yaml(tmp_path)
        _write_aggregations_yaml(tmp_path,
            "- name: sev\n  gate: foo\n  field: severity\n")
        _write_gate_json(tmp_path, "1", "foo", {
            "findings": [{"severity": "critical"}],
            "summary": "1 issue found",
            "_raw": "noise",
        })
        r = client.get("/api/aggregate/sev")
        assert r.json()["rows"][0]["finding"] == "critical"

    def test_mode_b_with_empty_findings_returns_empty_rows(
        self, client, tmp_path,
    ):
        """T11 case-3 (`{"findings": []}`) gives no contribution."""
        _write_tasks_yaml(tmp_path)
        _write_aggregations_yaml(tmp_path,
            "- name: sev\n  gate: foo\n  field: severity\n")
        _write_gate_json(tmp_path, "1", "foo", {"findings": []})
        r = client.get("/api/aggregate/sev")
        assert r.json()["rows"] == []

    def test_mode_b_with_corrupt_json_silently_skipped(self, client, tmp_path):
        """A hand-edited or interrupted .json shouldn't crash."""
        _write_tasks_yaml(tmp_path)
        _write_aggregations_yaml(tmp_path,
            "- name: sev\n  gate: foo\n  field: severity\n")
        d = tmp_path / "plans" / "task-1" / "gates"
        d.mkdir(parents=True)
        (d / "foo.json").write_text("not valid json {{{")
        # Plus a valid task to confirm the walk continues.
        _write_gate_json(tmp_path, "2", "foo", {
            "findings": [{"severity": "critical"}]})
        r = client.get("/api/aggregate/sev")
        rows = r.json()["rows"]
        assert rows == [{"finding": "critical", "total_count": 1,
                         "per_task": {"2": 1}}]

    def test_mode_b_with_non_dict_finding_skipped(self, client, tmp_path):
        """Per §6.D5: non-dict findings inside the list are skipped."""
        _write_tasks_yaml(tmp_path)
        _write_aggregations_yaml(tmp_path,
            "- name: sev\n  gate: foo\n  field: severity\n")
        _write_gate_json(tmp_path, "1", "foo", {
            "findings": [
                {"severity": "critical"},
                "this is a string, not a dict",
                None,
                {"severity": "high"},
            ],
        })
        r = client.get("/api/aggregate/sev")
        finds = {r_["finding"]: r_["total_count"]
                 for r_ in r.json()["rows"]}
        assert finds == {"critical": 1, "high": 1}

    def test_mode_b_non_string_field_value_uses_str(self, client, tmp_path):
        """Per §6.D5: integer fields round-trip through str()."""
        _write_tasks_yaml(tmp_path)
        _write_aggregations_yaml(tmp_path,
            "- name: line\n  gate: foo\n  field: line\n")
        _write_gate_json(tmp_path, "1", "foo", {
            "findings": [{"line": 42}, {"line": 42}, {"line": 100}],
        })
        r = client.get("/api/aggregate/line")
        finds = {r_["finding"]: r_["total_count"]
                 for r_ in r.json()["rows"]}
        assert finds == {"42": 2, "100": 1}

    def test_uses_custom_plans_dir(self, client, tmp_path):
        """plans_dir override (from defaults) is honored."""
        _write_tasks_yaml(tmp_path, """\
version: 1
defaults:
  plans_dir: artifacts
tasks:
  - id: 1
    title: T
    description: D
""")
        _write_aggregations_yaml(tmp_path,
            "- name: x\n  gate: foo\n  extract: 'BUG'\n")
        _write_gate_txt(tmp_path, "1", "foo", "BUG\n",
                        plans_dir="artifacts")
        r = client.get("/api/aggregate/x")
        assert r.status_code == 200
        assert r.json()["rows"][0]["total_count"] == 1


# ---------------------------------------------------------------------------
# T16: memory merge-suggestions endpoints + memory file + gate findings
# ---------------------------------------------------------------------------


@pytest.fixture
def _isolated_memory_dir(monkeypatch, tmp_path):
    """Redirect memory_dir_for_cwd to tmp_path/memdir so tests do not
    touch the operator's real ~/.claude/projects/.../memory directory.
    Pin §G.0: this only works because server.py imports
    memory_dir_for_cwd at module scope."""
    mem = tmp_path / "memdir"
    monkeypatch.setattr(
        "claude_plan_execute.server.memory_dir_for_cwd",
        lambda _cwd: mem,
    )
    return mem


def test_server_module_exposes_memory_dir_for_cwd_attribute():
    """Pin §G.0 / §N.20: server.py imports memory_dir_for_cwd,
    _drop_memory_file, and parse_memory_frontmatter at MODULE
    SCOPE so test fixtures can monkeypatch them."""
    import claude_plan_execute.server as s
    assert hasattr(s, "memory_dir_for_cwd")
    assert hasattr(s, "_drop_memory_file")
    assert hasattr(s, "parse_memory_frontmatter")


def _seed_suggestions(tmp_path: Path, suggestions: list[dict]) -> Path:
    plans = tmp_path / "plans"
    plans.mkdir(exist_ok=True)
    p = plans / "memory-merge-suggestions.json"
    p.write_text(_json.dumps(suggestions, indent=2))
    return p


def test_get_merge_suggestions_returns_empty_when_file_absent(
    tmp_path, client, _isolated_memory_dir,
):
    _write_tasks_yaml(tmp_path)
    r = client.get("/api/memory/merge-suggestions")
    assert r.status_code == 200
    assert r.json() == {"suggestions": []}


def test_get_merge_suggestions_returns_list_when_file_present(
    tmp_path, client, _isolated_memory_dir,
):
    _write_tasks_yaml(tmp_path)
    _seed_suggestions(tmp_path, [
        {"id": "a-b-12345678", "from": "a", "into": "b",
         "reason": "x", "status": "pending", "ts": "2026-01-01"},
    ])
    r = client.get("/api/memory/merge-suggestions")
    assert r.status_code == 200
    out = r.json()["suggestions"]
    assert len(out) == 1
    assert out[0]["from"] == "a"


def test_post_accept_merge_deletes_from_memory_and_marks_accepted(
    tmp_path, client, _isolated_memory_dir,
):
    _write_tasks_yaml(tmp_path)
    mem = _isolated_memory_dir
    mem.mkdir(parents=True, exist_ok=True)
    (mem / "a.md").write_text(
        "---\nname: a\nlifecycle: evergreen\ncitation_count: 0\n"
        "---\n\nbody\n", encoding="utf-8",
    )
    (mem / "b.md").write_text(
        "---\nname: b\nlifecycle: evergreen\ncitation_count: 0\n"
        "---\n\nbody\n", encoding="utf-8",
    )
    (mem / "MEMORY.md").write_text(
        "- [a](a.md) — desc\n- [b](b.md) — desc\n", encoding="utf-8",
    )
    sid = "a-b-12345678"
    _seed_suggestions(tmp_path, [
        {"id": sid, "from": "a", "into": "b",
         "reason": "x", "status": "pending", "ts": "2026-01-01"},
    ])
    r = client.post(f"/api/memory/merge-suggestions/{sid}/accept")
    assert r.status_code == 200
    assert r.json()["status"] == "accepted"
    assert not (mem / "a.md").exists()
    assert (mem / "b.md").exists()
    idx = (mem / "MEMORY.md").read_text()
    assert "(a.md)" not in idx
    assert "(b.md)" in idx
    # Suggestions file updated.
    data = _json.loads(
        (tmp_path / "plans" / "memory-merge-suggestions.json").read_text()
    )
    assert data[0]["status"] == "accepted"


def test_post_decline_merge_marks_declined_does_not_delete(
    tmp_path, client, _isolated_memory_dir,
):
    _write_tasks_yaml(tmp_path)
    mem = _isolated_memory_dir
    mem.mkdir(parents=True, exist_ok=True)
    (mem / "a.md").write_text("---\nname: a\n---\n", encoding="utf-8")
    sid = "a-b-12345678"
    _seed_suggestions(tmp_path, [
        {"id": sid, "from": "a", "into": "b",
         "reason": "x", "status": "pending", "ts": "2026-01-01"},
    ])
    r = client.post(f"/api/memory/merge-suggestions/{sid}/decline")
    assert r.status_code == 200
    assert r.json()["status"] == "declined"
    assert (mem / "a.md").exists()


def test_post_invalid_action_400(tmp_path, client, _isolated_memory_dir):
    _write_tasks_yaml(tmp_path)
    _seed_suggestions(tmp_path, [
        {"id": "x-y-12345678", "from": "x", "into": "y",
         "reason": "", "status": "pending", "ts": ""},
    ])
    r = client.post("/api/memory/merge-suggestions/x-y-12345678/explode")
    assert r.status_code == 400


def test_post_invalid_sid_400(tmp_path, client, _isolated_memory_dir):
    _write_tasks_yaml(tmp_path)
    r = client.post("/api/memory/merge-suggestions/..%2Fevil/accept")
    # path-traversal — the regex rejects it; either 400 from the
    # regex check OR 404 from FastAPI's path matcher.
    assert r.status_code in (400, 404)


def test_post_unknown_sid_404(tmp_path, client, _isolated_memory_dir):
    _write_tasks_yaml(tmp_path)
    _seed_suggestions(tmp_path, [
        {"id": "real-id-12345678", "from": "x", "into": "y",
         "reason": "", "status": "pending", "ts": ""},
    ])
    r = client.post("/api/memory/merge-suggestions/missing-id/accept")
    assert r.status_code == 404


def test_post_accept_already_accepted_409(
    tmp_path, client, _isolated_memory_dir,
):
    _write_tasks_yaml(tmp_path)
    sid = "x-y-12345678"
    _seed_suggestions(tmp_path, [
        {"id": sid, "from": "x", "into": "y",
         "reason": "", "status": "accepted", "ts": ""},
    ])
    r = client.post(f"/api/memory/merge-suggestions/{sid}/accept")
    assert r.status_code == 409


def test_get_memory_file_returns_body_and_frontmatter(
    tmp_path, client, _isolated_memory_dir,
):
    _write_tasks_yaml(tmp_path)
    mem = _isolated_memory_dir
    mem.mkdir(parents=True, exist_ok=True)
    (mem / "foo.md").write_text(
        "---\nname: foo\ntype: project\n---\n\nbody text\n",
        encoding="utf-8",
    )
    r = client.get("/api/memory/file/foo")
    assert r.status_code == 200
    out = r.json()
    assert "body text" in out["body"]
    assert out["frontmatter"]["name"] == "foo"


def test_get_memory_file_404_on_missing(tmp_path, client, _isolated_memory_dir):
    _write_tasks_yaml(tmp_path)
    r = client.get("/api/memory/file/ghost")
    assert r.status_code == 404


def test_get_memory_file_400_on_invalid_name(
    tmp_path, client, _isolated_memory_dir,
):
    _write_tasks_yaml(tmp_path)
    r = client.get("/api/memory/file/bad..name")
    assert r.status_code in (400, 404)


def test_get_memory_file_500_on_malformed_yaml_frontmatter(
    tmp_path, client, _isolated_memory_dir,
):
    """Pin §G.3 / §N.23: a malformed YAML frontmatter raises
    yaml.YAMLError; the except tuple includes it so the
    endpoint returns 500 with structured JSON, not a stack trace."""
    _write_tasks_yaml(tmp_path)
    mem = _isolated_memory_dir
    mem.mkdir(parents=True, exist_ok=True)
    (mem / "bad.md").write_text(
        "---\nname: bad\n  bad: indent: here\n---\nbody\n",
        encoding="utf-8",
    )
    r = client.get("/api/memory/file/bad")
    assert r.status_code == 500
    assert "error" in r.json()


def test_get_gate_findings_returns_sidecar_when_present(tmp_path, client):
    _write_tasks_yaml(tmp_path)
    gates_dir = tmp_path / "plans" / "task-1" / "gates"
    gates_dir.mkdir(parents=True, exist_ok=True)
    (gates_dir / "memory-curate.json").write_text(
        _json.dumps({"findings": [{"memory": "x", "verdict": "keep"}]}),
    )
    r = client.get("/api/gates/1/memory-curate/findings")
    assert r.status_code == 200
    out = r.json()
    assert out["findings"][0]["memory"] == "x"


def test_get_gate_findings_404_when_absent(tmp_path, client):
    _write_tasks_yaml(tmp_path)
    r = client.get("/api/gates/1/memory-curate/findings")
    assert r.status_code == 404
