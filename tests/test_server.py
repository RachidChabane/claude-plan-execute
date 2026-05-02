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
        "schema_version": 2,
        "tasks": {
            "1": {"status": "done", "attempts": [], "current_attempt": 1},
        },
    }


def test_get_state_returns_empty_tasks_dict_when_no_state_file(client, tmp_path):
    _write_tasks_yaml(tmp_path)
    r = client.get("/api/state")
    assert r.json() == {"schema_version": 2, "tasks": {}}


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
        "schema_version": 2,
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
        "schema_version": 2,
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
            "schema_version": 2,
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
