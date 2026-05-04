"""Declarative service lifecycle for unattended runs.

ServiceManager owns subprocess lifecycle for project-declared
services (dev-server, postgres, minio, etc). Designed for the
single-process runner; cross-process status is exported via
{plans_dir}/services.json (see plan §D8).

POSIX-only: uses os.setsid + os.killpg for process-group control.
"""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import IO


_SIGTERM_GRACE_SECONDS = 5
_HTTP_PROBE_TIMEOUT_SECONDS = 2
_TCP_PROBE_TIMEOUT_SECONDS = 2
_SHELL_PROBE_TIMEOUT_SECONDS = 5
_LOG_TAIL_BYTES = 64 * 1024  # bound the tail-read for /api/services
_MAX_CONSECUTIVE_CRASHES = 3


class ServiceError(Exception):
    """Base for all ServiceManager errors."""

    def __init__(self, name: str, reason: str) -> None:
        super().__init__(f"service {name!r}: {reason}")
        self.name = name
        self.reason = reason


class ServiceUnavailableError(ServiceError):
    """Service failed its ready check or could not be launched."""


class ServiceCrashedError(ServiceError):
    """Service was running and is now dead (poll() returned a code)."""


class ServiceUnstableError(ServiceError):
    """Service has crashed >= 3 times in a row; manager gives up."""


@dataclass
class _ServiceState:
    spec: dict                        # validated ServiceSpec dump
    proc: subprocess.Popen | None = None
    log_handle: IO | None = None
    crash_count: int = 0
    last_started_at: float = 0.0
    status: str = "stopped"           # stopped | starting | running | crashed | unstable
    pid: int | None = None
    log_file: str | None = None       # resolved abs path


class ServiceManager:
    """Owns service subprocesses for the lifetime of a CLI run.

    Thread-safety: a single re-entrant lock guards the state map.
    The runner loop is single-threaded; the only re-entrancy is
    `ensure()` recursing into `start()` for `depends_on` chains.

    Lifetime semantics (see plan §D5 for cli/runner call sites):
      - run:   started before the task loop; lives until run end.
      - phase: started at phase entry; killed at phase change.
      - task:  started at task entry; killed at task exit.
      - gate:  started before Phase 4; killed after Phase 4.

    Crash semantics:
      - poll_health() runs Popen.poll() on every running service.
      - dead service → record crash, increment counter, attempt
        start() once. Counter accumulates across consecutive
        crashes; reset_crash_count(name) (called by the runner
        at task end for services that survived the task) clears
        it back to 0.
      - 3rd consecutive crash → status='unstable',
        ServiceUnstableError raised.
    """

    def __init__(
        self,
        services: list[dict],
        plans_dir: Path,
    ) -> None:
        self._lock = threading.RLock()
        self._states: dict[str, _ServiceState] = {
            s["name"]: _ServiceState(spec=dict(s))
            for s in services
        }
        self._plans_dir = plans_dir
        # Resolve log paths eagerly (string-only — no mkdir, no file
        # writes) so /api/services can report them even before any
        # service has started. The actual log directory is created
        # lazily by start() (which already does
        # `log_path.parent.mkdir(parents=True, exist_ok=True)`); the
        # services.json file is created lazily by the first state
        # transition's call to _write_status_file().
        #
        # Why no init-time writes: cli.main constructs this manager
        # before the --list / --status / --reset early-returns
        # (Plan §D5.1). Those commands are historically read-only.
        # Writing services.json or mkdir'ing <plans_dir>/services/
        # during init would create files on every read-only invocation
        # AND would clobber the post-mortem snapshot from a prior
        # crashed run (status='running', pid=<orphan>) with a fresh
        # 'stopped' snapshot, losing operator-relevant state.
        # The dashboard's load_services() returns [] on missing file
        # (Plan §D6.1) so an absent services.json is well-handled.
        for st in self._states.values():
            if st.spec.get("log_file"):
                st.log_file = str(Path(st.spec["log_file"]).resolve())
            else:
                st.log_file = str(
                    plans_dir / "services" / f"{st.spec['name']}.log"
                )
        # NOTE: NO _write_status_file() here. See block-comment above.

    def names(self) -> list[str]:
        return list(self._states.keys())

    def spec_lifetime(self, name: str) -> str:
        st = self._states.get(name)
        if st is None:
            raise ServiceUnavailableError(name, "no such service")
        return st.spec.get("lifetime", "run")

    def is_running(self, name: str) -> bool:
        with self._lock:
            st = self._states.get(name)
            if st is None or st.proc is None:
                return False
            return st.proc.poll() is None

    def start(self, name: str) -> None:
        """Start a single service, run ready_check, raise on failure.

        Idempotent: returns immediately if already running.
        Resolves and starts depends_on (recursively) before this
        one — schema validation guarantees no cycles.
        """
        with self._lock:
            st = self._states.get(name)
            if st is None:
                raise ServiceUnavailableError(
                    name, "not declared in services[]",
                )
            if self.is_running(name):
                return
            for dep in st.spec.get("depends_on", []):
                self.start(dep)

            spec = st.spec
            cmd = spec["command"]
            cwd = spec.get("cwd")
            env_overlay = spec.get("env") or {}

            # NOTE: after the §D2.3 init, st.log_file is ALWAYS set
            # (the init resolves either spec['log_file'] or the
            # default `<plans_dir>/services/<name>.log` path string).
            # The `else` branch below is therefore defensively dead
            # code — it would only execute if a future change
            # introduced an alternative construction path for
            # _ServiceState. Kept defensive; do not rely on its
            # absence semantically.
            log_path = Path(st.log_file) if st.log_file else None
            if log_path is not None:
                # First-touch creation of <plans_dir>/services/ —
                # deliberately deferred from §D2.3 init so read-only
                # commands (--list / --status / --reset) don't create
                # the directory. mkdir is idempotent (exist_ok=True).
                log_path.parent.mkdir(parents=True, exist_ok=True)
                st.log_handle = open(log_path, "ab", buffering=0)
            else:
                st.log_handle = None

            child_env = dict(os.environ)
            child_env.update(env_overlay)

            try:
                proc = subprocess.Popen(
                    ["bash", "-c", cmd],
                    cwd=cwd,
                    env=child_env,
                    stdout=st.log_handle or subprocess.DEVNULL,
                    stderr=st.log_handle or subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL,
                    preexec_fn=os.setsid,
                )
            except (OSError, FileNotFoundError) as exc:
                if st.log_handle is not None:
                    st.log_handle.close()
                    st.log_handle = None
                raise ServiceUnavailableError(
                    name, f"could not spawn: {exc}",
                ) from exc

            st.proc = proc
            st.pid = proc.pid
            st.status = "starting"
            st.last_started_at = time.time()
            self._write_status_file()

            try:
                self._wait_ready(spec, proc)
            except ServiceUnavailableError:
                self._kill_process_group(proc)
                if st.log_handle is not None:
                    st.log_handle.close()
                    st.log_handle = None
                st.proc = None
                st.pid = None
                st.status = "stopped"
                self._write_status_file()
                raise
            st.status = "running"
            self._write_status_file()

    def stop(self, name: str) -> None:
        """SIGTERM, then SIGKILL after _SIGTERM_GRACE_SECONDS."""
        with self._lock:
            st = self._states.get(name)
            if st is None or st.proc is None:
                return
            self._kill_process_group(st.proc)
            if st.log_handle is not None:
                st.log_handle.close()
                st.log_handle = None
            st.proc = None
            st.pid = None
            st.status = "stopped"
            self._write_status_file()

    def restart(self, name: str) -> None:
        """Stop then start. Crash counter is preserved across the
        restart (see plan §11.7 for the trade-off rationale)."""
        self.stop(name)
        self.start(name)

    def ensure(self, required: list[str]) -> None:
        """Bring up listed services not already running.

        Order: starts in topological order over depends_on; the
        input list order is the tie-breaker for unrelated
        services. Raises ServiceUnavailableError on first failure
        (does NOT roll back partial starts — caller decides
        whether to tear-down).
        """
        with self._lock:
            for name in required:
                if name not in self._states:
                    raise ServiceUnavailableError(
                        name, "not declared in services[]",
                    )
                if not self.is_running(name):
                    self.start(name)

    def tear_down_for_scope(self, scope: str) -> None:
        """Stop services whose lifetime matches `scope`.

        Reverse-depends_on order: a service that depends on
        postgres dies before postgres dies. Errors on individual
        stops are swallowed (best-effort cleanup).
        """
        with self._lock:
            target = [
                name for name, st in self._states.items()
                if st.spec.get("lifetime", "run") == scope
                and st.proc is not None
            ]
            for name in self._reverse_topo_order(target):
                try:
                    self.stop(name)
                except Exception:
                    pass

    def tear_down_all(self) -> None:
        """Stop every running service, in reverse depends_on order."""
        with self._lock:
            running = [
                name for name, st in self._states.items()
                if st.proc is not None
            ]
            for name in self._reverse_topo_order(running):
                try:
                    self.stop(name)
                except Exception:
                    pass

    def _reverse_topo_order(self, names: list[str]) -> list[str]:
        """Return `names` in reverse depends_on order: dependents
        before their dependencies. Schema validation guarantees
        no cycles, so a simple DFS suffices."""
        order: list[str] = []
        seen: set[str] = set()

        def visit(n: str) -> None:
            if n in seen or n not in names:
                return
            seen.add(n)
            spec = self._states[n].spec
            for dep in spec.get("depends_on", []):
                visit(dep)
            order.append(n)

        for n in names:
            visit(n)
        return list(reversed(order))

    def poll_health(self) -> list[dict]:
        """Detect crashes; auto-restart once per detection.

        Returns a list of event dicts:
            {"name": str,
             "event": "crashed_restarted" | "crashed_no_restart"
                      | "unstable",
             "exit_code": int | None,
             "crash_count": int}

        - `crashed_restarted`: process was running, now dead;
          we tried start() and it succeeded. Counter +1.
        - `crashed_no_restart`: process was running, now dead;
          we tried start() and it failed. Counter +1, status
          stays 'crashed'. The next ensure()/restart() retries.
          NOT raised — a subsequent ensure() failure surfaces
          via ServiceUnavailableError on the dependent task.
        - `unstable`: 3rd consecutive *crash* (not 3rd failed
          restart). Status set to 'unstable'; raises
          ServiceUnstableError on the FIRST such service after
          the loop completes so the runner can record
          service_unstable on the dependent task.

        Counter ticks on each crash *detection*, never on
        restart-attempt failures. This matches the spec wording
        "3 consecutive crashes" exactly.

        Note: dashboard-initiated restarts (POST /api/services/
        <name>/restart sends SIGTERM to the recorded PID) appear
        here as crashes — the counter increments. Documented
        trade-off; differentiating intentional vs unintentional
        restart adds IPC complexity for marginal benefit.
        """
        events: list[dict] = []
        with self._lock:
            for name, st in list(self._states.items()):
                if st.proc is None:
                    continue
                rc = st.proc.poll()
                if rc is None:
                    continue
                # Process is dead — record the crash detection.
                if st.log_handle is not None:
                    st.log_handle.close()
                    st.log_handle = None
                st.proc = None
                st.pid = None
                st.crash_count += 1
                if st.crash_count >= _MAX_CONSECUTIVE_CRASHES:
                    st.status = "unstable"
                    self._write_status_file()
                    events.append({
                        "name": name, "event": "unstable",
                        "exit_code": rc,
                        "crash_count": st.crash_count,
                    })
                    continue
                st.status = "crashed"
                self._write_status_file()
                try:
                    self.start(name)
                    events.append({
                        "name": name, "event": "crashed_restarted",
                        "exit_code": rc,
                        "crash_count": st.crash_count,
                    })
                except ServiceUnavailableError:
                    # Restart failed; do NOT escalate to unstable
                    # here (the spec ties unstable to crash-count,
                    # not to restart-attempt failures). Status
                    # stays 'crashed'; counter stays at its current
                    # tick. Next ensure()/restart() will retry.
                    self._write_status_file()
                    events.append({
                        "name": name, "event": "crashed_no_restart",
                        "exit_code": rc,
                        "crash_count": st.crash_count,
                    })
        for ev in events:
            if ev["event"] == "unstable":
                raise ServiceUnstableError(
                    ev["name"],
                    f"{ev['crash_count']} consecutive crashes "
                    f"(last exit={ev['exit_code']})",
                )
        return events

    def reset_crash_count(self, name: str) -> None:
        """Reset counter once a service has held running through a
        full task. Called by the runner at task-end success-path
        (plan §D4.6). Only services that ARE running get reset
        (per §D4.6 success-path guard) — a service in 'crashed'
        state retains its counter into the next task.
        """
        with self._lock:
            st = self._states.get(name)
            if st is not None:
                st.crash_count = 0
                self._write_status_file()

    def _wait_ready(
        self, spec: dict, proc: subprocess.Popen,
    ) -> None:
        """Poll the configured ready_check every 1s up to
        ready_timeout_seconds. Raises ServiceUnavailableError on
        timeout OR if the subprocess dies before becoming ready.
        """
        rc = spec["ready_check"]
        timeout = int(spec.get("ready_timeout_seconds", 60))
        deadline = time.time() + timeout
        last_err = ""
        while time.time() < deadline:
            if proc.poll() is not None:
                raise ServiceUnavailableError(
                    spec["name"],
                    f"process exited before ready (exit "
                    f"{proc.returncode})",
                )
            ok, err = self._probe_once(rc)
            if ok:
                return
            last_err = err
            time.sleep(1.0)
        raise ServiceUnavailableError(
            spec["name"],
            f"ready_check timed out after {timeout}s "
            f"({rc['kind']}); last error: {last_err}",
        )

    @staticmethod
    def _probe_once(rc: dict) -> tuple[bool, str]:
        kind = rc["kind"]
        if kind == "http":
            url = rc["url"]
            try:
                req = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(
                    req, timeout=_HTTP_PROBE_TIMEOUT_SECONDS,
                ) as resp:
                    code = resp.getcode()
                    if 200 <= code < 400:
                        return True, ""
                    return False, f"http {code}"
            except urllib.error.HTTPError as exc:
                if 200 <= exc.code < 400:
                    return True, ""
                return False, f"http {exc.code}"
            except (urllib.error.URLError, OSError, ValueError) as exc:
                return False, f"{type(exc).__name__}: {exc}"
        if kind == "tcp":
            port = int(rc["port"])
            host = "127.0.0.1"
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(_TCP_PROBE_TIMEOUT_SECONDS)
            try:
                sock.connect((host, port))
                return True, ""
            except (OSError, socket.timeout) as exc:
                return False, f"{type(exc).__name__}: {exc}"
            finally:
                sock.close()
        if kind == "shell":
            cmd = rc["command"]
            try:
                proc = subprocess.run(
                    ["bash", "-c", cmd],
                    capture_output=True, text=True,
                    timeout=_SHELL_PROBE_TIMEOUT_SECONDS,
                )
                if proc.returncode == 0:
                    return True, ""
                return False, f"exit {proc.returncode}"
            except subprocess.TimeoutExpired:
                return False, "shell probe timeout"
            except (OSError, FileNotFoundError) as exc:
                return False, f"{type(exc).__name__}: {exc}"
        return False, f"unknown ready_check kind: {kind!r}"

    @staticmethod
    def _kill_process_group(proc: subprocess.Popen) -> None:
        """SIGTERM the process group; SIGKILL after grace.

        Mirrors the SIGTERM/SIGKILL ladder in
        server.py:_stop_execution and runner.handle_sigint.
        Best-effort: failures are swallowed at the bottom of the
        ladder.
        """
        if proc.poll() is not None:
            return
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, OSError):
            try:
                proc.send_signal(signal.SIGTERM)
            except (ProcessLookupError, OSError):
                return

        deadline = time.time() + _SIGTERM_GRACE_SECONDS
        while time.time() < deadline:
            if proc.poll() is not None:
                return
            time.sleep(0.1)

        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            try:
                proc.kill()
            except (ProcessLookupError, OSError):
                pass
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass

    def status(self) -> list[dict]:
        """Snapshot of every declared service. Used by /api/services
        and by `_write_status_file`."""
        with self._lock:
            return [
                {
                    "name": name,
                    "command": st.spec.get("command", ""),
                    "lifetime": st.spec.get("lifetime", "run"),
                    "ready_check": st.spec.get("ready_check", {}),
                    "log_file": st.log_file,
                    "status": st.status,
                    "pid": st.pid,
                    "crash_count": st.crash_count,
                    "depends_on": list(st.spec.get("depends_on", [])),
                }
                for name, st in self._states.items()
            ]

    def tail_log(self, name: str, lines: int = 200) -> list[str]:
        """Return the last `lines` lines of the service's log file,
        or [] if missing. Capped at _LOG_TAIL_BYTES (64 KB)."""
        with self._lock:
            st = self._states.get(name)
            if st is None or not st.log_file:
                return []
            log_path = st.log_file
        path = Path(log_path)
        if not path.exists():
            return []
        try:
            size = path.stat().st_size
            with open(path, "rb") as f:
                if size > _LOG_TAIL_BYTES:
                    f.seek(size - _LOG_TAIL_BYTES)
                tail = f.read().decode("utf-8", errors="replace")
        except OSError:
            return []
        return tail.splitlines()[-lines:]

    def _status_file_path(self) -> Path:
        return self._plans_dir / "services.json"

    def _write_status_file(self) -> None:
        """Atomically write the status snapshot to
        <plans_dir>/services.json. Mirrors state.State.save's
        tempfile dance.
        """
        path = self._status_file_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        snapshot = {
            "schema_version": 1,
            "services": self.status(),
        }
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(snapshot, f, indent=2)
            os.rename(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise


__all__ = [
    "ServiceManager",
    "ServiceError",
    "ServiceUnavailableError",
    "ServiceCrashedError",
    "ServiceUnstableError",
]
