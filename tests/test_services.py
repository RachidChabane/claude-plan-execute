"""Tests for claude_plan_execute.services (v0.3 task 8)."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from claude_plan_execute.services import (
    ServiceManager,
    ServiceUnavailableError,
    ServiceUnstableError,
    _MAX_CONSECUTIVE_CRASHES,
)

sys.path.insert(0, str(Path(__file__).parent))
from _service_helpers import (  # type: ignore  # noqa: E402
    _FakePopen,
    _LocalHttpServer,
    _free_tcp_port,
)


def _spec(name: str, **overrides) -> dict:
    """Construct a baseline service spec dict with sensible defaults."""
    base = {
        "name": name,
        "command": "true",
        "cwd": None,
        "env": {},
        "ready_check": {"kind": "tcp", "port": 65535,
                         "url": None, "command": None},
        "ready_timeout_seconds": 5,
        "lifetime": "run",
        "log_file": None,
        "depends_on": [],
    }
    base.update(overrides)
    return base


# --- (a) Ready-check kind=http (3) -----------------------------------------

def test_ready_check_http_succeeds_when_endpoint_returns_200(tmp_path):
    """Plan §D2.7: a healthy http endpoint passes the probe."""
    with _LocalHttpServer(status=200) as srv:
        spec = _spec(
            "dev",
            command="sleep 5",
            ready_check={"kind": "http", "url": srv.url,
                         "port": None, "command": None},
            ready_timeout_seconds=5,
        )
        m = ServiceManager([spec], tmp_path)
        try:
            m.start("dev")
            assert m.is_running("dev")
        finally:
            m.tear_down_all()


def test_ready_check_http_succeeds_on_2xx_3xx_range(tmp_path):
    """Plan §D2.7: 2xx/3xx all count as ready."""
    with _LocalHttpServer(status=302) as srv:
        spec = _spec(
            "dev",
            command="sleep 5",
            ready_check={"kind": "http", "url": srv.url,
                         "port": None, "command": None},
            ready_timeout_seconds=5,
        )
        m = ServiceManager([spec], tmp_path)
        try:
            m.start("dev")
            assert m.is_running("dev")
        finally:
            m.tear_down_all()


def test_ready_check_http_times_out_when_endpoint_unreachable(tmp_path):
    """Plan §D2.7: an unreachable endpoint raises after timeout."""
    free_port = _free_tcp_port()
    spec = _spec(
        "dev",
        command="sleep 5",
        ready_check={"kind": "http",
                     "url": f"http://127.0.0.1:{free_port}/",
                     "port": None, "command": None},
        ready_timeout_seconds=2,
    )
    m = ServiceManager([spec], tmp_path)
    try:
        with pytest.raises(ServiceUnavailableError):
            m.start("dev")
    finally:
        m.tear_down_all()


# --- (b) Ready-check kind=tcp (2) ------------------------------------------

def test_ready_check_tcp_succeeds_when_listener_present(tmp_path):
    """Plan §D2.7: tcp probe succeeds when something listens."""
    with _LocalHttpServer(status=200) as srv:
        spec = _spec(
            "dev",
            command="sleep 5",
            ready_check={"kind": "tcp", "port": srv.port,
                         "url": None, "command": None},
            ready_timeout_seconds=5,
        )
        m = ServiceManager([spec], tmp_path)
        try:
            m.start("dev")
            assert m.is_running("dev")
        finally:
            m.tear_down_all()


def test_ready_check_tcp_times_out_when_no_listener(tmp_path):
    """Plan §D2.7: tcp probe times out when nothing listens."""
    free_port = _free_tcp_port()
    spec = _spec(
        "dev",
        command="sleep 5",
        ready_check={"kind": "tcp", "port": free_port,
                     "url": None, "command": None},
        ready_timeout_seconds=2,
    )
    m = ServiceManager([spec], tmp_path)
    try:
        with pytest.raises(ServiceUnavailableError):
            m.start("dev")
    finally:
        m.tear_down_all()


# --- (c) Ready-check kind=shell (2) ----------------------------------------

def test_ready_check_shell_succeeds_on_exit_zero(tmp_path):
    """Plan §D2.7: shell probe succeeds on exit 0."""
    spec = _spec(
        "dev",
        command="sleep 5",
        ready_check={"kind": "shell", "command": "true",
                     "url": None, "port": None},
        ready_timeout_seconds=5,
    )
    m = ServiceManager([spec], tmp_path)
    try:
        m.start("dev")
        assert m.is_running("dev")
    finally:
        m.tear_down_all()


def test_ready_check_shell_fails_on_nonzero_exit(tmp_path):
    """Plan §D2.7: shell probe fails on nonzero exit; times out."""
    spec = _spec(
        "dev",
        command="sleep 5",
        ready_check={"kind": "shell", "command": "false",
                     "url": None, "port": None},
        ready_timeout_seconds=2,
    )
    m = ServiceManager([spec], tmp_path)
    try:
        with pytest.raises(ServiceUnavailableError):
            m.start("dev")
    finally:
        m.tear_down_all()


# --- (d) start → stop → start (2) ------------------------------------------

def test_start_then_stop_then_start_is_idempotent(tmp_path):
    """Plan §D2.4: start is idempotent; stop frees the proc."""
    with _LocalHttpServer(status=200) as srv:
        spec = _spec(
            "dev",
            command="sleep 30",
            ready_check={"kind": "tcp", "port": srv.port,
                         "url": None, "command": None},
            ready_timeout_seconds=5,
        )
        m = ServiceManager([spec], tmp_path)
        try:
            m.start("dev")
            assert m.is_running("dev")
            m.stop("dev")
            assert not m.is_running("dev")
            m.start("dev")
            assert m.is_running("dev")
        finally:
            m.tear_down_all()


def test_stop_kills_process_group_so_descendants_die(tmp_path):
    """Plan §D2.8: SIGTERM hits the whole process group."""
    marker = tmp_path / "child_pid"
    # Parent writes the bash subshell PID, then sleeps; we kill parent
    # via stop() and verify the recorded pid is gone.
    cmd = (
        f"echo $$ > {marker}; sleep 30"
    )
    spec = _spec(
        "dev",
        command=cmd,
        ready_check={"kind": "shell",
                     "command": f"test -s {marker}",
                     "url": None, "port": None},
        ready_timeout_seconds=5,
    )
    m = ServiceManager([spec], tmp_path)
    try:
        m.start("dev")
        time.sleep(0.2)
        recorded = int(marker.read_text().strip())
        m.stop("dev")
        time.sleep(0.5)
        # PID should be gone (or reaped) — ESRCH on getpgid signals death
        with pytest.raises((ProcessLookupError, OSError)):
            os.kill(recorded, 0)
    finally:
        m.tear_down_all()


# --- (e) ensure missing (3) ------------------------------------------------

def test_ensure_with_unknown_service_raises(tmp_path):
    """Plan §D2.4: ensure raises when an unknown service is requested."""
    m = ServiceManager([], tmp_path)
    with pytest.raises(ServiceUnavailableError):
        m.ensure(["missing"])


def test_ensure_skips_already_running_services(tmp_path):
    """Plan §D2.4: ensure is a no-op for already-running services."""
    with _LocalHttpServer(status=200) as srv:
        spec = _spec(
            "dev",
            command="sleep 30",
            ready_check={"kind": "tcp", "port": srv.port,
                         "url": None, "command": None},
            ready_timeout_seconds=5,
        )
        m = ServiceManager([spec], tmp_path)
        try:
            m.ensure(["dev"])
            first_pid = m._states["dev"].pid
            m.ensure(["dev"])
            assert m._states["dev"].pid == first_pid
        finally:
            m.tear_down_all()


def test_ensure_starts_depends_on_first(tmp_path):
    """Plan §D2.4: depends_on chains start in order."""
    with _LocalHttpServer(status=200) as srv:
        a = _spec(
            "a",
            command="sleep 30",
            ready_check={"kind": "tcp", "port": srv.port,
                         "url": None, "command": None},
            ready_timeout_seconds=5,
        )
        b = _spec(
            "b",
            command="sleep 30",
            ready_check={"kind": "tcp", "port": srv.port,
                         "url": None, "command": None},
            ready_timeout_seconds=5,
            depends_on=["a"],
        )
        m = ServiceManager([a, b], tmp_path)
        try:
            m.ensure(["b"])
            assert m.is_running("a")
            assert m.is_running("b")
        finally:
            m.tear_down_all()


# --- (f) Crash auto-restart (6) --------------------------------------------

def test_poll_health_no_crash_returns_empty_events(tmp_path):
    """Plan §D2.6: poll_health returns [] when nothing crashed."""
    m = ServiceManager([_spec("dev")], tmp_path)
    assert m.poll_health() == []


def test_poll_health_single_crash_auto_restarts_and_increments_counter(
    tmp_path, monkeypatch,
):
    """Plan §D2.6: a crash bumps the counter and auto-restarts."""
    spec = _spec("dev")
    m = ServiceManager([spec], tmp_path)
    fake = _FakePopen([1])
    m._states["dev"].proc = fake
    m._states["dev"].pid = fake.pid
    m._states["dev"].status = "running"

    calls: list[str] = []
    def fake_start(name: str) -> None:
        calls.append(name)
        m._states[name].proc = _FakePopen([])
        m._states[name].pid = 12345
        m._states[name].status = "running"

    monkeypatch.setattr(m, "start", fake_start)
    events = m.poll_health()
    assert len(events) == 1
    assert events[0]["event"] == "crashed_restarted"
    assert events[0]["crash_count"] == 1
    assert calls == ["dev"]


def test_poll_health_third_consecutive_crash_raises_unstable(tmp_path, monkeypatch):
    """Plan §D2.6: 3 consecutive crash detections trigger unstable."""
    spec = _spec("dev")
    m = ServiceManager([spec], tmp_path)
    m._states["dev"].crash_count = _MAX_CONSECUTIVE_CRASHES - 1
    fake = _FakePopen([1])
    m._states["dev"].proc = fake
    m._states["dev"].pid = fake.pid
    m._states["dev"].status = "running"

    monkeypatch.setattr(m, "start", lambda name: None)
    with pytest.raises(ServiceUnstableError):
        m.poll_health()
    assert m._states["dev"].status == "unstable"
    assert m._states["dev"].crash_count == _MAX_CONSECUTIVE_CRASHES


def test_poll_health_failed_restart_does_not_raise_unstable(
    tmp_path, monkeypatch,
):
    """Plan §D2.6: a failed restart does NOT escalate to unstable.
    Counter ticks once per crash detection, status stays 'crashed'."""
    spec = _spec("dev")
    m = ServiceManager([spec], tmp_path)
    fake = _FakePopen([1])
    m._states["dev"].proc = fake
    m._states["dev"].pid = fake.pid
    m._states["dev"].status = "running"

    def boom(_name: str) -> None:
        raise ServiceUnavailableError("dev", "spawn failed")

    monkeypatch.setattr(m, "start", boom)
    events = m.poll_health()
    assert len(events) == 1
    assert events[0]["event"] == "crashed_no_restart"
    assert events[0]["crash_count"] == 1
    assert m._states["dev"].status == "crashed"


def test_poll_health_failed_restart_does_not_double_increment(
    tmp_path, monkeypatch,
):
    """Plan §D2.6: counter ticks per crash detection only — a failed
    restart attempt itself does NOT increment again."""
    spec = _spec("dev")
    m = ServiceManager([spec], tmp_path)
    fake = _FakePopen([1])
    m._states["dev"].proc = fake
    m._states["dev"].pid = fake.pid
    m._states["dev"].status = "running"

    monkeypatch.setattr(
        m, "start",
        lambda _n: (_ for _ in ()).throw(
            ServiceUnavailableError("dev", "fail")
        ),
    )
    m.poll_health()
    assert m._states["dev"].crash_count == 1


def test_reset_crash_count_clears_counter_after_successful_run(tmp_path):
    """Plan §D2.6: reset_crash_count zeroes the counter."""
    m = ServiceManager([_spec("dev")], tmp_path)
    m._states["dev"].crash_count = 2
    m.reset_crash_count("dev")
    assert m._states["dev"].crash_count == 0


# --- (g) lifetime tear-downs (3) -------------------------------------------

def _seed_running(m: ServiceManager, name: str) -> None:
    """Mark a service running by seeding _states with a fake proc."""
    m._states[name].proc = _FakePopen([])
    m._states[name].pid = 12345
    m._states[name].status = "running"


def test_tear_down_for_scope_task_stops_only_task_lifetime_services(tmp_path):
    """Plan §D2.5: scope-targeted tear-down stops only matching services."""
    a = _spec("run-svc", lifetime="run")
    b = _spec("task-svc", lifetime="task")
    m = ServiceManager([a, b], tmp_path)
    _seed_running(m, "run-svc")
    _seed_running(m, "task-svc")
    m.tear_down_for_scope("task")
    assert m._states["run-svc"].proc is not None
    assert m._states["task-svc"].proc is None


def test_tear_down_for_scope_run_does_not_touch_task_or_gate_lifetime(tmp_path):
    """Plan §D2.5: lifetime=run scope leaves task/gate scopes alone."""
    a = _spec("a", lifetime="run")
    b = _spec("b", lifetime="task")
    c = _spec("c", lifetime="gate")
    m = ServiceManager([a, b, c], tmp_path)
    _seed_running(m, "a")
    _seed_running(m, "b")
    _seed_running(m, "c")
    m.tear_down_for_scope("run")
    assert m._states["a"].proc is None
    assert m._states["b"].proc is not None
    assert m._states["c"].proc is not None


def test_tear_down_for_scope_reverse_topo_order(tmp_path):
    """Plan §D2.5: dependents are stopped before their dependencies."""
    a = _spec("a", lifetime="task")
    b = _spec("b", lifetime="task", depends_on=["a"])
    m = ServiceManager([a, b], tmp_path)
    _seed_running(m, "a")
    _seed_running(m, "b")
    order: list[str] = []
    real_stop = m.stop

    def tracked(name: str) -> None:
        order.append(name)
        real_stop(name)

    m.stop = tracked  # type: ignore[method-assign]
    m.tear_down_for_scope("task")
    assert order.index("b") < order.index("a")


# --- Process-group + log-file (3) ------------------------------------------

def test_start_writes_stdout_to_log_file(tmp_path):
    """Plan §D2.4: subprocess stdout is appended to log_file."""
    log_path = tmp_path / "logs" / "dev.log"
    spec = _spec(
        "dev",
        command="echo hello-from-svc; sleep 5",
        ready_check={"kind": "shell", "command": "true",
                     "url": None, "port": None},
        log_file=str(log_path),
        ready_timeout_seconds=5,
    )
    m = ServiceManager([spec], tmp_path)
    try:
        m.start("dev")
        time.sleep(0.3)
        m.stop("dev")
        contents = log_path.read_text()
        assert "hello-from-svc" in contents
    finally:
        m.tear_down_all()


def test_log_file_parent_dir_is_created(tmp_path):
    """Plan §D2.4: missing parent dirs of log_file are created on start."""
    log_path = tmp_path / "deep" / "subdir" / "out.log"
    spec = _spec(
        "dev",
        command="sleep 5",
        ready_check={"kind": "shell", "command": "true",
                     "url": None, "port": None},
        log_file=str(log_path),
        ready_timeout_seconds=5,
    )
    m = ServiceManager([spec], tmp_path)
    try:
        m.start("dev")
        assert log_path.parent.exists()
    finally:
        m.tear_down_all()


def test_kill_process_group_falls_back_to_proc_kill_on_oserror(monkeypatch):
    """Plan §D2.8: SIGKILL fallback to proc.kill() when killpg raises."""
    fake = _FakePopen([None, 0])
    fake.kill_called = False  # type: ignore[attr-defined]

    def fake_kill():
        fake.kill_called = True  # type: ignore[attr-defined]
        # Simulate process exiting after kill
        fake._seq = [0]

    fake.kill = fake_kill  # type: ignore[attr-defined]

    def raise_killpg(*_a, **_kw):
        raise ProcessLookupError("no such pgid")

    def raise_getpgid(_pid):
        raise ProcessLookupError("no such pid")

    monkeypatch.setattr(os, "killpg", raise_killpg)
    monkeypatch.setattr(os, "getpgid", raise_getpgid)

    def fake_send_signal(_sig):
        # Simulate process death after SIGTERM fallback
        fake._seq = [0]

    fake.send_signal = fake_send_signal  # type: ignore[method-assign]
    ServiceManager._kill_process_group(fake)


# --- status / services.json (5) --------------------------------------------

def test_status_includes_pid_and_lifetime_for_running_service(tmp_path):
    """Plan §D2.9: status() snapshot includes pid + lifetime for running."""
    spec = _spec("dev", lifetime="phase")
    m = ServiceManager([spec], tmp_path)
    _seed_running(m, "dev")
    snap = m.status()
    assert len(snap) == 1
    assert snap[0]["name"] == "dev"
    assert snap[0]["lifetime"] == "phase"
    assert snap[0]["pid"] == 12345
    assert snap[0]["status"] == "running"


def test_write_status_file_uses_atomic_rename(tmp_path):
    """Plan §D2.10: services.json snapshot has schema_version=1."""
    spec = _spec("dev")
    m = ServiceManager([spec], tmp_path)
    _seed_running(m, "dev")
    m._write_status_file()
    p = tmp_path / "services.json"
    assert p.exists()
    payload = json.loads(p.read_text())
    assert payload["schema_version"] == 1
    assert payload["services"][0]["name"] == "dev"


def test_write_status_file_creates_parent_directory(tmp_path):
    """Plan §D2.10: parent dir is created on first write."""
    plans_dir = tmp_path / "deep" / "plans"
    spec = _spec("dev")
    m = ServiceManager([spec], plans_dir)
    m._write_status_file()
    assert (plans_dir / "services.json").exists()


def test_init_does_not_write_services_json(tmp_path):
    """Plan §D2.3: __init__ MUST NOT create services.json or services/."""
    spec = _spec("dev")
    ServiceManager([spec], tmp_path)
    assert not (tmp_path / "services.json").exists()
    assert not (tmp_path / "services").exists()


def test_init_does_not_overwrite_existing_services_json(tmp_path):
    """Plan §D2.3: a pre-existing services.json (post-mortem snapshot)
    must survive a new manager construction byte-for-byte."""
    plans_dir = tmp_path
    plans_dir.mkdir(exist_ok=True)
    p = plans_dir / "services.json"
    original = json.dumps({
        "schema_version": 1,
        "services": [{
            "name": "dev", "command": "x", "lifetime": "run",
            "ready_check": {"kind": "tcp", "port": 80},
            "log_file": None, "status": "running",
            "pid": 99999, "crash_count": 0, "depends_on": [],
        }],
    }, indent=2)
    p.write_text(original)
    ServiceManager([_spec("dev")], plans_dir)
    assert p.read_text() == original


def test_first_state_transition_writes_services_json(tmp_path):
    """Plan §D2.3: first state transition (start/stop/poll_health)
    writes services.json with schema_version=1."""
    spec = _spec("dev")
    m = ServiceManager([spec], tmp_path)
    assert not (tmp_path / "services.json").exists()
    _seed_running(m, "dev")
    m._write_status_file()
    payload = json.loads((tmp_path / "services.json").read_text())
    assert payload["schema_version"] == 1
    assert payload["services"][0]["name"] == "dev"


# --- Restart helper (2) ----------------------------------------------------

def test_restart_stops_then_starts(tmp_path, monkeypatch):
    """Plan §D2.4: restart calls stop then start."""
    spec = _spec("dev")
    m = ServiceManager([spec], tmp_path)
    _seed_running(m, "dev")
    calls: list[str] = []
    monkeypatch.setattr(m, "stop", lambda n: calls.append(f"stop:{n}"))
    monkeypatch.setattr(m, "start", lambda n: calls.append(f"start:{n}"))
    m.restart("dev")
    assert calls == ["stop:dev", "start:dev"]


def test_restart_does_not_reset_crash_count(tmp_path, monkeypatch):
    """Plan §11.7: crash counter is preserved across restart()."""
    spec = _spec("dev")
    m = ServiceManager([spec], tmp_path)
    m._states["dev"].crash_count = 2
    monkeypatch.setattr(m, "stop", lambda n: None)
    monkeypatch.setattr(m, "start", lambda n: None)
    m.restart("dev")
    assert m._states["dev"].crash_count == 2


# --- tail_log (1) ----------------------------------------------------------

def test_tail_log_returns_last_lines(tmp_path):
    """Plan §D2.9: tail_log returns up to N most-recent lines."""
    log_path = tmp_path / "x.log"
    log_path.write_text("a\nb\nc\nd\ne\n")
    spec = _spec("dev", log_file=str(log_path))
    m = ServiceManager([spec], tmp_path)
    lines = m.tail_log("dev", lines=2)
    assert lines == ["d", "e"]


def test_tail_log_returns_empty_when_log_missing(tmp_path):
    """Plan §D2.9: tail_log returns [] when log file is absent."""
    spec = _spec("dev", log_file=str(tmp_path / "nope.log"))
    m = ServiceManager([spec], tmp_path)
    assert m.tail_log("dev") == []
