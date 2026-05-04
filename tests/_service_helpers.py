"""Shared fixtures for test_services.py and test_runner.py.

Module-private; not exported through claude_plan_execute.
"""
from __future__ import annotations

import http.server
import socket
import threading
from typing import Any


def _free_tcp_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class _LocalHttpServer:
    """In-thread HTTP server returning configurable status.

    Usage:
        with _LocalHttpServer(status=200) as srv:
            assert srv.url == "http://127.0.0.1:<port>/"
    """

    def __init__(self, status: int = 200, body: bytes = b"ok") -> None:
        self._status = status
        self._body = body
        self._port = _free_tcp_port()
        self._httpd: http.server.HTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._port}/"

    @property
    def port(self) -> int:
        return self._port

    def __enter__(self) -> "_LocalHttpServer":
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(outer._status)
                self.send_header("Content-Length", str(len(outer._body)))
                self.end_headers()
                self.wfile.write(outer._body)

            def log_message(self, *_a, **_kw):
                pass

        # Skip the inherited server_bind's socket.getfqdn() reverse-DNS
        # lookup (it can take 30+ seconds on macOS when DNS is slow);
        # the server name is unused by tests here.
        class _FastBindHTTPServer(http.server.HTTPServer):
            def server_bind(self):
                from socketserver import TCPServer
                TCPServer.server_bind(self)
                self.server_name = "localhost"
                self.server_port = self.server_address[1]

        self._httpd = _FastBindHTTPServer(("127.0.0.1", self._port), H)
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()


class _FakePopen:
    """Minimal Popen stub for poll_health crash tests.

    poll_sequence — list of return values for successive poll()
    calls. None = still running, int = exited with that code.
    Once `_dead` is set (via send_signal/kill or terminate/wait), poll()
    returns 0 immediately so the SIGTERM grace loop in
    `ServiceManager._kill_process_group` exits fast.
    """

    def __init__(
        self, poll_sequence: list[int | None], pid: int = 99999,
    ) -> None:
        self._seq = list(poll_sequence)
        self.pid = pid
        self.returncode: int | None = None
        self._dead = False

    def poll(self):
        if self._dead:
            return self.returncode if self.returncode is not None else 0
        if not self._seq:
            return None
        v = self._seq.pop(0)
        self.returncode = v
        return v

    def send_signal(self, _sig):
        self._dead = True
        if self.returncode is None:
            self.returncode = 0

    def kill(self):
        self._dead = True
        if self.returncode is None:
            self.returncode = 0

    def wait(self, timeout=None):
        self._dead = True
        return self.returncode if self.returncode is not None else 0
