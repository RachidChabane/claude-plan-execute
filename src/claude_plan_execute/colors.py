"""ANSI color helpers."""
from __future__ import annotations

import sys

_use_color: bool = True


def set_use_color(enabled: bool) -> None:
    """Toggle global color output. Called once from the CLI's --no-color path."""
    global _use_color
    _use_color = enabled


def color(text: str, code: str) -> str:
    if _use_color and sys.stdout.isatty():
        return f"\033[{code}m{text}\033[0m"
    return text


def green(t: str) -> str: return color(t, "32")
def yellow(t: str) -> str: return color(t, "33")
def red(t: str) -> str: return color(t, "31")
def bold(t: str) -> str: return color(t, "1")
def dim(t: str) -> str: return color(t, "2")


def status_color(status: str) -> str:
    if status == "done":
        return green(status)
    if status == "pending":
        return dim(status)
    if status in ("planning", "reviewing", "revising", "implementing", "approved"):
        return yellow(status)
    return status
