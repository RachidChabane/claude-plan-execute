"""Loader, compiler, and registration helpers for `docs/invariants.yaml`.

Each entry in the file becomes a registered Gate on the default
registry at process startup (driven by `cli.main()`). The runner's
Phase-4 dispatch supports two access paths:

  - Direct lookup by name: `gates: [invariant-acl-on-chunk-read]`.
  - Meta-name expansion: `gates: [invariant-grep]` runs every loaded
    invariant.

The module is purposefully a sibling of `gates/builtin.py` rather
than folded into either it or `loader.py`: the loader/compiler is a
self-contained subsystem whose outputs are `Gate` instances. Adding
a fourth subsystem to `loader.py` (alongside tasks.yaml, persona,
decisions) would inflate it past 500 lines.
"""
from __future__ import annotations

import shlex
import sys
from pathlib import Path
from typing import Literal

try:
    import yaml
except ImportError:
    print("Error: PyYAML is required.", file=sys.stderr)
    sys.exit(1)

try:
    from pydantic import (
        BaseModel,
        ConfigDict,
        ValidationError,
        model_validator,
    )
except ImportError:
    print(
        "Error: pydantic>=2.0 is required.",
        file=sys.stderr,
    )
    sys.exit(1)

from claude_plan_execute.colors import yellow
from claude_plan_execute.gates import Gate, GateRegistry
from claude_plan_execute.schema import format_validation_error


# ---------------------------------------------------------------------------
# Pydantic model
# ---------------------------------------------------------------------------

class InvariantEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    kind: Literal["shell", "agent", "human"] = "shell"

    command: str | None = None
    regex_must_match: str | None = None
    regex_must_not_match: str | None = None
    files: list[str] | None = None

    prompt_template: str | None = None

    model: str | None = None
    effort: str | None = None
    # T10 (v0.3 §2.4) — flipped to `int | None = None` so
    # `compile_entry` can distinguish "user left it unset" (then
    # pick kind-specific default: 1440 for human, 10 for
    # shell/agent) from "user explicitly set N". See
    # plans/task-10/plan.md §D2.
    timeout_minutes: int | None = None
    parallel_safe: bool = True
    on_failure: Literal["block", "warn"] = "block"
    when: list[str] | None = None
    comment: str | None = None
    # T10 — optional override for the convention sentinel path.
    # Honored only by HumanGate; CLI / server use the convention
    # path verbatim (see plan §D8).
    sentinel_path: str | None = None

    @model_validator(mode="after")
    def _check_shell_specifiers(self) -> "InvariantEntry":
        if self.kind == "shell":
            provided = sum(
                1 for v in (
                    self.command,
                    self.regex_must_match,
                    self.regex_must_not_match,
                ) if v is not None
            )
            if provided > 1:
                raise ValueError(
                    "shell entry sets more than one of "
                    "{command, regex_must_match, regex_must_not_match}; "
                    "exactly one is required"
                )
        return self


# ---------------------------------------------------------------------------
# Compiler
# ---------------------------------------------------------------------------

def compile_entry(entry: InvariantEntry) -> Gate:
    """Compile a validated InvariantEntry into a runtime Gate.

    `regex_must_match` / `regex_must_not_match` are sugar for a
    `bash -c` grep one-liner over the configured starting paths
    (default ['.']). `files` entries are PATHS passed verbatim to
    `grep -R`, NOT bash globs (single-quoted tokens never glob).

    T10 (v0.3 §2.4) — resolves `timeout_minutes` kind-specific
    when the entry left it unset: 1440 for kind=human (24 h),
    10 for kind=shell / kind=agent. Explicit values pass through
    unchanged. See plans/task-10/plan.md §D2.
    """
    when = tuple(entry.when) if entry.when else ("always",)

    # Kind-specific timeout default: human waits 24 h by default,
    # shell/agent stay at 10 min. Resolved here (not at the Gate
    # dataclass layer) so the explicit `timeout_minutes: 30` YAML
    # form survives kind-agnostic.
    if entry.kind == "human":
        timeout = (
            entry.timeout_minutes
            if entry.timeout_minutes is not None
            else 1440
        )
    else:
        timeout = (
            entry.timeout_minutes
            if entry.timeout_minutes is not None
            else 10
        )

    if entry.kind == "shell":
        if entry.command is not None:
            cmd = entry.command
        elif entry.regex_must_match is not None:
            if entry.files is not None and len(entry.files) == 0:
                raise ValueError(
                    f"invariant {entry.name!r} files: [] is empty; "
                    f"omit the field to default to ['.']"
                )
            files = " ".join(
                shlex.quote(f) for f in (entry.files or ["."])
            )
            cmd = (
                f"grep -RIn -E -- "
                f"{shlex.quote(entry.regex_must_match)} {files}"
            )
        elif entry.regex_must_not_match is not None:
            if entry.files is not None and len(entry.files) == 0:
                raise ValueError(
                    f"invariant {entry.name!r} files: [] is empty; "
                    f"omit the field to default to ['.']"
                )
            files = " ".join(
                shlex.quote(f) for f in (entry.files or ["."])
            )
            cmd = (
                f"! grep -RIn -E -- "
                f"{shlex.quote(entry.regex_must_not_match)} {files}"
            )
        else:
            raise ValueError(
                f"invariant {entry.name!r} kind=shell requires one of: "
                f"command, regex_must_match, regex_must_not_match"
            )
        return Gate(
            kind="shell",
            command=("bash", "-c", cmd),
            timeout_minutes=timeout,
            parallel_safe=entry.parallel_safe,
            on_failure=entry.on_failure,
            when=when,
        )

    if entry.kind == "human":
        if entry.prompt_template is None:
            raise ValueError(
                f"invariant {entry.name!r} kind=human requires prompt_template"
            )
        # parallel_safe is HARDCODED to False for human gates — see
        # plan §C / §D2. A YAML-supplied `parallel_safe: true` is
        # silently ignored (a kind=human gate that ran in parallel
        # would parallelise human attention; that's a bug).
        return Gate(
            kind="human",
            prompt_template=entry.prompt_template,
            sentinel_path=entry.sentinel_path,
            timeout_minutes=timeout,
            parallel_safe=False,
            on_failure=entry.on_failure,
            when=when,
        )

    # kind == "agent" (existing branch)
    if entry.prompt_template is None:
        raise ValueError(
            f"invariant {entry.name!r} kind=agent requires prompt_template"
        )
    return Gate(
        kind="agent",
        prompt_template=entry.prompt_template,
        model=entry.model,
        effort=entry.effort,
        timeout_minutes=timeout,
        parallel_safe=entry.parallel_safe,
        on_failure=entry.on_failure,
        when=when,
    )


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_invariants(
    path: Path | str | None,
) -> list[tuple[str, Gate]]:
    """Read `path`, parse YAML, validate each entry, compile to Gates.

    Returns a list of (name, Gate) pairs in file order. Per-entry
    validation errors warn-skip; the rest of the file still loads.
    """
    if not path:
        return []
    p = Path(path) if not isinstance(path, Path) else path
    if not p.exists() or not p.is_file():
        return []

    with open(p) as f:
        raw = yaml.safe_load(f)

    if raw is None:
        return []
    if not isinstance(raw, list):
        print(yellow(
            f"Warning: {p} root must be a YAML list of invariant "
            f"entries; got {type(raw).__name__}. No invariants loaded."
        ))
        return []

    out: list[tuple[str, Gate]] = []
    seen: set[str] = set()
    reserved = {"tests", "lint", "invariant-grep"}

    for i, entry_raw in enumerate(raw):
        if not isinstance(entry_raw, dict):
            print(yellow(
                f"Warning: {p} entry [{i}] is not a mapping; skipping."
            ))
            continue
        name = entry_raw.get("name")
        if not name:
            print(yellow(
                f"Warning: {p} entry [{i}] missing 'name'; skipping."
            ))
            continue
        if name in reserved:
            print(yellow(
                f"Warning: {p} entry {name!r} uses a reserved gate "
                f"name; skipping."
            ))
            continue
        if name in seen:
            print(yellow(
                f"Warning: {p} entry {name!r} is a duplicate; "
                f"skipping the second occurrence."
            ))
            continue
        try:
            entry = InvariantEntry.model_validate(entry_raw)
        except ValidationError as exc:
            print(yellow(
                f"Warning: {p} entry {name!r} failed validation:\n"
                f"{format_validation_error(exc, p)}"
            ))
            continue
        try:
            gate = compile_entry(entry)
        except (ValueError, TypeError) as exc:
            print(yellow(
                f"Warning: {p} entry {name!r} failed to compile: {exc}"
            ))
            continue
        seen.add(name)
        out.append((name, gate))

    return out


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

_loaded_invariant_names: list[str] = []


def register_invariants_on(
    registry: GateRegistry,
    entries: list[tuple[str, Gate]],
) -> list[str]:
    """Register each (name, Gate) pair on `registry`. Returns the
    names actually registered. Skips collisions with a warning.

    Side effect: appends to the module-level `_loaded_invariant_names`
    only when `registry` is the module-level singleton — tests with a
    local registry get isolated meta-name expansion behavior.
    """
    registered: list[str] = []
    for name, gate in entries:
        try:
            registry.register(name, gate)
        except ValueError as exc:
            print(yellow(
                f"Warning: invariant {name!r} not registered: {exc}"
            ))
            continue
        registered.append(name)

    from claude_plan_execute.gates import default_registry as _dr
    if registry is _dr():
        _loaded_invariant_names.extend(registered)

    return registered


def loaded_invariant_names() -> list[str]:
    """Public accessor for the runner's `invariant-grep` meta-name
    expansion. Returns a copy of the names registered on the default
    registry, in file order.
    """
    return list(_loaded_invariant_names)


__all__ = [
    "InvariantEntry",
    "compile_entry",
    "load_invariants",
    "register_invariants_on",
    "loaded_invariant_names",
]
