"""Cross-task memory primitives (T15 v0.3).

Pure helpers used by the runner to keep Claude Code's auto-loaded
project memory directory honest:

  - `memory_dir_for_cwd`        — encode the project cwd to the
    Claude Code memory directory path.
  - `parse_memory_frontmatter`  — re-export of `loader.parse_frontmatter`
    so callers don't need a second import.
  - `write_memory_with_frontmatter` — atomic-write a memory file.
  - `increment_citations_for_plan` — bump `citation_count` for every
    `[MEM: <name>]` reference found in a plan body.
  - `prune_expired_memories`    — delete memory files whose lifecycle
    has elapsed (`task:<N>` once task N is done; `slate:<id>` once
    every task in the slate is done).

Import direction is one-way: `runner` imports `memory`. This module
must NOT import from `runner` or `state`.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import yaml

from claude_plan_execute.colors import yellow
from claude_plan_execute.loader import parse_frontmatter as parse_memory_frontmatter

# T13 forward-compat: parallel slate runner can drive citation +
# prune for two tasks concurrently. Serialize per-file mutation across
# threads so two writers can't race on the same memory file.
_memory_lock = threading.Lock()


_CITATION_RE = re.compile(r"\[MEM:\s*([A-Za-z0-9_\-]+)\s*\]")


def memory_dir_for_cwd(cwd: Path, *, home: Path | None = None) -> Path:
    """Compute the Claude Code memory dir for a given project cwd.

    Encoding scheme matches Claude Code: take the absolute cwd path
    and replace every `/` with `-`. The leading `/` becomes a
    leading `-`. Example: `/Users/me/proj` -> `-Users-me-proj`.

    `home` defaults to `Path.home()`; tests override with `tmp_path`
    so no real `~/.claude/projects/...` is touched.

    Pure: no filesystem access (the directory may not exist).
    """
    home = home if home is not None else Path.home()
    encoded = str(Path(cwd).resolve()).replace("/", "-")
    return home / ".claude" / "projects" / encoded / "memory"


def write_memory_with_frontmatter(path: Path, fm: dict, body: str) -> None:
    """Atomic-write a memory file with `fm` as YAML frontmatter and
    `body` as the post-frontmatter content.

    Uses tempfile + os.rename like state.State.save() so the file is
    never observed half-written. Idempotent.

    Body is preserved verbatim aside from one normalization:
    leading newlines are stripped before re-prefixing with a single
    `\\n` after the closing `---\\n`. Round-trip-safe.
    """
    fm_block = yaml.safe_dump(fm, sort_keys=False, default_flow_style=False)
    body_norm = body.lstrip("\n")
    text = f"---\n{fm_block}---\n\n{body_norm}"
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.rename(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def increment_citations_for_plan(
    memory_dir: Path, plan_text: str,
) -> dict[str, int]:
    """Find every [MEM: <name>] in plan_text; for each cited name,
    increment that memory's `citation_count` frontmatter field.

    Returns a dict mapping memory_name -> number of increments
    actually applied (0 for names that didn't resolve to a memory
    file). The dict is useful for tests and operator logs.

    Behavior:
      - Multiple occurrences of the same name in plan_text count
        multiple times.
      - References to nonexistent memory files are silently ignored
        (the reviewer caught hallucinated names; the runner does
        not double-fault).
      - `memory_dir` not existing -> returns a dict whose values are
        all 0 (every cited name is "unknown" because no files exist).
      - A memory file that lacks frontmatter, or whose frontmatter
        is malformed, is skipped with a yellow warning; counts as 0.
      - File-name resolution: <memory_dir>/<name>.md. Names that are
        not slug-shaped will not be matched by the regex and so are
        unreachable via this primitive — that is fine.

    Thread-safe: holds `_memory_lock` for the duration. T13's
    parallel slate runner reads/writes one memory file at a time.
    """
    names = _CITATION_RE.findall(plan_text)
    if not names:
        return {}
    result: dict[str, int] = {n: 0 for n in names}
    if not memory_dir.is_dir():
        return result

    with _memory_lock:
        for name in names:
            path = memory_dir / f"{name}.md"
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8")
                fm, body = parse_memory_frontmatter(text)
            except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
                print(yellow(
                    f"  [memory] skipping {path.name}: cannot parse "
                    f"frontmatter ({type(exc).__name__}: {exc})"
                ))
                continue
            if not fm:
                print(yellow(
                    f"  [memory] skipping {path.name}: no frontmatter."
                ))
                continue
            try:
                current = int(fm.get("citation_count", 0) or 0)
            except (TypeError, ValueError):
                current = 0
            fm["citation_count"] = current + 1
            try:
                write_memory_with_frontmatter(path, fm, body)
            except OSError as exc:
                print(yellow(
                    f"  [memory] failed to write {path.name}: {exc}"
                ))
                continue
            result[name] += 1
    return result


def prune_expired_memories(
    memory_dir: Path,
    *,
    is_task_done: Callable[[str | int], bool],
    slate_id: str,
    slate_complete: bool,
) -> list[str]:
    """Walk memory_dir, delete per-memory files whose lifecycle has
    expired, and remove the corresponding `MEMORY.md` index lines.

    Returns the list of deleted memory names (file stems, in
    deletion order) so the runner can log a one-line summary.

    Lifecycle rules:
      - `lifecycle: evergreen` -> never delete.
      - `lifecycle: task:<N>` -> delete iff `is_task_done(N)` is True.
        `<N>` is parsed as int when the substring is all digits;
        otherwise treated as a string id (forward-compat with
        non-int task ids).
        FOOTGUN: a memory scoped to the CURRENT task is pruned at
        this task's success boundary (the runner flips status to
        'done' before invoking prune). Intentional but documented.
      - `lifecycle: slate:<id>` -> delete iff `id == slate_id` AND
        `slate_complete` is True. A `slate:bar` memory is NEVER
        pruned by a `slate:foo` run (cross-slate isolation).
      - Any other lifecycle value (including missing) -> never delete.

    `MEMORY.md` is rewritten by removing every line whose target
    matches `(<basename>.md)` for any deleted file. The rewrite is
    idempotent. If the agent uses a different link format (absolute
    URL, raw filename) the per-memory file IS deleted but the index
    line survives — soft failure, leaves a dangling link. T16 will
    regenerate MEMORY.md as part of the curation gate.

    Defensive: missing `memory_dir`, missing `MEMORY.md`, malformed
    frontmatter, file-permission errors -> log yellow warning and
    skip the offending file. Never raise.

    Thread-safe: holds `_memory_lock` for the duration.
    """
    if not memory_dir.is_dir():
        return []

    deleted: list[str] = []
    with _memory_lock:
        candidates = sorted(
            p for p in memory_dir.glob("*.md") if p.name != "MEMORY.md"
        )
        for path in candidates:
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                print(yellow(
                    f"  [memory] skipping {path.name}: cannot read "
                    f"({type(exc).__name__}: {exc})"
                ))
                continue
            try:
                fm, _body = parse_memory_frontmatter(text)
            except yaml.YAMLError as exc:
                print(yellow(
                    f"  [memory] skipping {path.name}: malformed "
                    f"frontmatter ({exc})"
                ))
                continue

            lifecycle = (fm.get("lifecycle") or "").strip()
            should_delete = False
            m_task = re.match(r"^task:(.+)$", lifecycle)
            m_slate = re.match(r"^slate:(.+)$", lifecycle)
            if lifecycle == "evergreen":
                should_delete = False
            elif m_task:
                tok = m_task.group(1).strip()
                tid: str | int = int(tok) if tok.isdigit() else tok
                try:
                    should_delete = bool(is_task_done(tid))
                except Exception as exc:  # noqa: BLE001
                    print(yellow(
                        f"  [memory] is_task_done({tid!r}) raised; "
                        f"keeping {path.name}: {exc}"
                    ))
                    should_delete = False
            elif m_slate:
                slate = m_slate.group(1).strip()
                should_delete = (slate == slate_id) and bool(slate_complete)
            else:
                should_delete = False

            if should_delete:
                try:
                    path.unlink()
                except OSError as exc:
                    print(yellow(
                        f"  [memory] failed to unlink {path.name}: {exc}"
                    ))
                    continue
                deleted.append(path.stem)

        if deleted:
            index_path = memory_dir / "MEMORY.md"
            if index_path.is_file():
                try:
                    index_text = index_path.read_text(encoding="utf-8")
                except OSError as exc:
                    print(yellow(
                        f"  [memory] failed to read MEMORY.md: {exc}"
                    ))
                else:
                    needles = [f"({stem}.md)" for stem in deleted]
                    new_lines = [
                        line for line in index_text.splitlines(keepends=True)
                        if not any(n in line for n in needles)
                    ]
                    new_text = "".join(new_lines)
                    if new_text != index_text:
                        try:
                            _atomic_write_text(index_path, new_text)
                        except OSError as exc:
                            print(yellow(
                                f"  [memory] failed to rewrite "
                                f"MEMORY.md: {exc}"
                            ))
    return deleted


def _atomic_write_text(path: Path, text: str) -> None:
    """Atomic write of plain text (no frontmatter). Helper for the
    `MEMORY.md` rewrite in `prune_expired_memories`. Behaves like
    `state.State.save`: tempfile in the same directory, then rename.
    """
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.rename(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def count_memory_entries(memory_dir: Path) -> int:
    """Count per-memory files in `memory_dir`, excluding MEMORY.md.

    A "memory entry" is a `*.md` file whose name is not literally
    `MEMORY.md`. Symbolic links are followed (they're memory
    files too); non-`.md` files are ignored.

    Returns 0 when `memory_dir` is missing — typical first run on
    a fresh project. No I/O beyond the directory listing; no
    frontmatter parsing.
    """
    if not memory_dir.is_dir():
        return 0
    return sum(
        1 for p in memory_dir.glob("*.md")
        if p.name != "MEMORY.md"
    )


def _drop_memory_file(memory_dir: Path, name: str) -> bool:
    """Delete `<memory_dir>/<name>.md` and remove its line from
    MEMORY.md. Returns True if the file was unlinked, False if
    it was already missing or the directory was missing.

    Mirrors the line-removal machinery in `prune_expired_memories`
    (memory.py:267-273). The duplication is shallow (~10 lines)
    and deliberately not factored into a shared helper.

    Defensive: a missing target file is a silent no-op (the
    lifecycle prune may have deleted it earlier the same task,
    or the agent may have referenced a name that never existed).
    File-permission errors warn and return False.
    """
    if not memory_dir.is_dir():
        return False
    path = memory_dir / f"{name}.md"
    with _memory_lock:
        if not path.is_file():
            return False
        try:
            path.unlink()
        except OSError as exc:
            print(yellow(
                f"  [memory] failed to unlink {path.name}: {exc}"
            ))
            return False
        index_path = memory_dir / "MEMORY.md"
        if index_path.is_file():
            try:
                index_text = index_path.read_text(encoding="utf-8")
                needle = f"({name}.md)"
                new_text = "".join(
                    line for line in index_text.splitlines(keepends=True)
                    if needle not in line
                )
                if new_text != index_text:
                    _atomic_write_text(index_path, new_text)
            except OSError as exc:
                print(yellow(
                    f"  [memory] failed to rewrite MEMORY.md "
                    f"after dropping {name}: {exc}"
                ))
    return True


_MERGE_INTO_RE = re.compile(r"^merge_into:([A-Za-z0-9_\-]+)$")


def _suggestion_id(from_name: str, into_name: str) -> str:
    """Deterministic id: `<from>-<into>-<sha1[:8]>`. Stable across
    re-runs so re-applying the same agent output is idempotent."""
    import hashlib
    h = hashlib.sha1(
        f"{from_name}->{into_name}".encode("utf-8")
    ).hexdigest()[:8]
    return f"{from_name}-{into_name}-{h}"


def apply_memory_curate_findings(
    *,
    memory_dir: Path,
    findings: list[dict],
    suggestions_path: Path,
) -> dict[str, int]:
    """Apply the runner-actionable verdicts from a memory-curate
    findings list and accumulate merge suggestions.

    For each finding dict (the agent's per-memory verdict):
      - `verdict == "keep"` -> no action.
      - `verdict == "drop"` -> call `_drop_memory_file(memory_dir,
        name)`. A missing target file is a silent no-op.
      - `verdict matches "merge_into:<other>"` -> append a record
        to the suggestions JSON file (`suggestions_path`). The
        suggestion is appended only when BOTH the source and
        target memories exist on disk; a missing-target skips
        with a yellow warning.
      - any other shape -> silently skipped (fail-soft, mirrors
        T11's malformed-appendix handling).

    `suggestions_path` parent dir is created if missing. Atomic
    write via `_atomic_write_text`. Each suggestion gets a
    deterministic id (`<from>-<into>-<sha1[:8]>`) so a re-run with
    the same agent output produces the same id (no duplicates).

    Returns `{"kept": N, "dropped": N, "merge_suggested": N,
    "skipped": N}`.
    """
    counts = {"kept": 0, "dropped": 0,
              "merge_suggested": 0, "skipped": 0}
    if not isinstance(findings, list):
        return counts

    existing: list[dict] = []
    if suggestions_path.is_file():
        try:
            existing = json.loads(
                suggestions_path.read_text(encoding="utf-8")
            )
            if not isinstance(existing, list):
                existing = []
        except (OSError, json.JSONDecodeError):
            existing = []
    existing_ids = {s.get("id") for s in existing if isinstance(s, dict)}

    new_suggestions: list[dict] = []
    for finding in findings:
        if not isinstance(finding, dict):
            counts["skipped"] += 1
            continue
        name = finding.get("memory")
        verdict = finding.get("verdict")
        reason = finding.get("reason", "") or ""
        if not isinstance(name, str) or not isinstance(verdict, str):
            counts["skipped"] += 1
            continue

        if verdict == "keep":
            counts["kept"] += 1
            continue
        if verdict == "drop":
            if _drop_memory_file(memory_dir, name):
                counts["dropped"] += 1
            else:
                counts["skipped"] += 1
            continue
        m = _MERGE_INTO_RE.match(verdict)
        if m:
            target = m.group(1)
            src_path = memory_dir / f"{name}.md"
            tgt_path = memory_dir / f"{target}.md"
            if not src_path.is_file():
                counts["skipped"] += 1
                continue
            if not tgt_path.is_file():
                print(yellow(
                    f"  [memory] merge_into:{target} for {name} "
                    f"skipped — target file does not exist"
                ))
                counts["skipped"] += 1
                continue
            sid = _suggestion_id(name, target)
            if sid in existing_ids:
                continue
            new_suggestions.append({
                "id": sid,
                "from": name,
                "into": target,
                "reason": reason,
                "status": "pending",
                "ts": datetime.now(timezone.utc).isoformat(),
            })
            counts["merge_suggested"] += 1
            continue
        counts["skipped"] += 1

    if new_suggestions:
        with _memory_lock:
            merged = existing + new_suggestions
            suggestions_path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write_text(
                suggestions_path,
                json.dumps(merged, indent=2) + "\n",
            )
    return counts
