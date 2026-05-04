"""Auto-split blocked tasks via a Claude agent + tasks.yaml mutation.

The runner calls `run_splitter` from a single helper site after
recording certain terminal `FailureRecord`s. The splitter:

  1. Reads the parent task's plan.md, review-*.md, and state attempts.
  2. Asks a Claude agent (sonnet/medium) whether to split and into
     what children.
  3. If the agent returns `decision == "split"` with >= 2 children,
     mutates tasks.yaml under an `fcntl.flock` exclusive lock, using
     `ruamel.yaml` round-trip mode so user-authored comments survive.
  4. Flips the parent's state-entry status to `"split"` and records
     `split_children` for the dashboard.

Failures (subprocess crash, malformed JSON, file write OSError) log
a yellow warning and return — the parent stays `blocked`. No
`FailureRecord` is appended for the splitter itself.
"""
from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq

from claude_plan_execute.claude import run_claude
from claude_plan_execute.colors import dim, green, yellow
from claude_plan_execute.loader import (
    load_decisions_index,
    load_persona_fragment,
)
from claude_plan_execute.prompts import (
    build_decisions_section,
    build_failure_context_section,
    build_persona_section,
)
from claude_plan_execute.state import State

_SPLITTER_TEMPLATE_PATH = (
    Path(__file__).parent / "gates" / "templates" / "splitter.md"
)

# Mirrors gates.agent._FINDINGS_RE; intentional duplication — see
# plan §4 pin 5. The shape parsed inside the markers differs (single
# object vs list[dict]).
_FINDINGS_RE = re.compile(
    r"<!-- BEGIN_FINDINGS_JSON -->\s*(.*?)\s*<!-- END_FINDINGS_JSON -->",
    re.DOTALL,
)

# Splitter-agent run_claude knobs. Pinned constants per plan §4 pin 11.
_SPLITTER_MODEL = "sonnet"
_SPLITTER_EFFORT = "medium"
_SPLITTER_MAX_TURNS = 60
_SPLITTER_TIMEOUT_MINUTES = 30


_REVIEW_ROUND_RE = re.compile(r"review-(\d+)\.md$")


def _review_round(name: str) -> int:
    m = _REVIEW_ROUND_RE.search(name)
    return int(m.group(1)) if m else 0


def _log_dim(msg: str) -> None:
    print(dim(msg), file=sys.stdout)


def _log_yellow(msg: str) -> None:
    print(yellow(msg), file=sys.stdout)


def _log_green(msg: str) -> None:
    print(green(msg), file=sys.stdout)


def _build_prompt(
    *, task: dict, state: State, resolved: dict, plans_dir: Path,
) -> str:
    """Compose the splitter prompt.

    Composition order (mirrors prompts.build_plan_prompt):
        failure_context + persona + decisions + body
    """
    tid = task["id"]
    task_dir = plans_dir / f"task-{tid}"

    plan_path = task_dir / "plan.md"
    plan_body = (
        plan_path.read_text(encoding="utf-8")
        if plan_path.is_file() else ""
    )

    review_paths = sorted(
        task_dir.glob("review-*.md"),
        key=lambda p: _review_round(p.name),
    )
    review_blocks: list[str] = []
    for p in review_paths:
        try:
            text = p.read_text(encoding="utf-8")
        except OSError:
            continue
        review_blocks.append(f"### {p.name}\n\n{text}\n")
    reviews_section = (
        "\n\n".join(review_blocks) if review_blocks
        else "(no reviews on disk)"
    )

    attempts = list(state.get(tid).get("attempts", []))
    failure_context = build_failure_context_section(attempts)

    persona_path = resolved["persona_file"]["value"]
    persona_text = load_persona_fragment(persona_path)
    persona_section = build_persona_section(persona_text, "")

    decisions_dir = resolved["decisions_dir"]["value"]
    decisions_text = load_decisions_index(decisions_dir)
    decisions_section = build_decisions_section(decisions_text)

    template_body = _SPLITTER_TEMPLATE_PATH.read_text(encoding="utf-8")

    key_files_str = "\n".join(
        f"  - {kf}" for kf in (task.get("key_files") or [])
    ) or "  (none)"
    body = template_body
    body = body.replace("{task_id}", str(tid))
    body = body.replace("{task_title}", task["title"])
    body = body.replace("{task_description}", task.get("description", ""))
    body = body.replace("{key_files}", key_files_str)
    body = body.replace("{plan_body}", plan_body)
    body = body.replace("{reviews_section}", reviews_section)

    return failure_context + persona_section + decisions_section + body


def _parse_appendix(
    text: str, tid,
) -> tuple[str | None, str | None, list[dict]]:
    """Return (decision, reason, children).

    `decision` is None on any parse failure (the caller short-circuits).
    Logs a yellow warning on parse failure / shape mismatch.
    """
    m = _FINDINGS_RE.search(text)
    if not m:
        _log_yellow(
            f"  [splitter] task {tid}: no findings appendix in agent "
            f"response; leaving blocked."
        )
        return (None, None, [])
    body = m.group(1)
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        _log_yellow(
            f"  [splitter] task {tid}: malformed findings JSON "
            f"({exc}); leaving blocked."
        )
        return (None, None, [])
    if not isinstance(parsed, dict):
        _log_yellow(
            f"  [splitter] task {tid}: findings is not an object; "
            f"leaving blocked."
        )
        return (None, None, [])
    decision = parsed.get("decision")
    reason = parsed.get("reason") or ""
    children = parsed.get("children") or []
    if not isinstance(children, list) or not all(
        isinstance(c, dict) for c in children
    ):
        _log_yellow(
            f"  [splitter] task {tid}: 'children' is not a list of "
            f"objects; leaving blocked."
        )
        return (None, None, [])
    return (decision, reason, children)


def _render_child_entry(
    *, child: dict, parent_id, split_depth: int, parent_phase,
) -> "CommentedMap":
    """Render one child entry as a ruamel CommentedMap.

    Field order (matches existing tasks.yaml entries):
      id, phase, title, description, key_files, commit_message,
      depends_on, parent_task, split_depth.
    Any field absent from the agent payload is omitted.
    """
    cm = CommentedMap()
    cm["id"] = str(child["id"])
    phase = child.get("phase") or parent_phase
    if phase is not None:
        cm["phase"] = phase
    cm["title"] = str(child.get("title") or f"task {child['id']}")
    cm["description"] = str(child.get("description") or "")
    if child.get("key_files"):
        kf_seq = CommentedSeq(list(child["key_files"]))
        cm["key_files"] = kf_seq
    if child.get("commit_message"):
        cm["commit_message"] = str(child["commit_message"])
    deps = child.get("depends_on") or []
    if deps:
        cm["depends_on"] = CommentedSeq(list(deps))
    cm["parent_task"] = parent_id
    cm["split_depth"] = split_depth
    return cm


def _apply_split_to_yaml(
    *, tasks_yaml_path: Path, parent_task: dict,
    children: list[dict], current_depth: int,
) -> list[str] | None:
    """Mutate tasks.yaml in place under fcntl.flock(LOCK_EX).

    Returns the list of child ids on success; None on any failure
    (logged with a yellow warning).
    """
    parent_id = parent_task["id"]
    yaml_obj = YAML(typ="rt")
    yaml_obj.preserve_quotes = True
    yaml_obj.width = 4096  # avoid line-wrapping long titles
    yaml_obj.indent(mapping=2, sequence=4, offset=2)

    # Acquire exclusive lock on the original inode. The lock is
    # held across the read AND the os.replace below.
    #
    # Note on the inode-replace pattern: `os.replace(tmp_path,
    # tasks_yaml_path)` rebinds the path to a NEW inode at the
    # tail of this function, while we still hold a flock on the
    # OLD inode (the one we opened at the top of this block). A
    # subsequent worker that does `os.open(str(tasks_yaml_path), ...)`
    # opens the NEW inode and tries to flock on it independently —
    # the OLD-inode flock does not block them. That sounds
    # race-prone, but it is correct because:
    #   1. The first worker holds the lock until `os.replace`
    #      completes; concurrent open+flock attempts on the path
    #      either see the OLD inode (and queue on the same lock) or
    #      the NEW inode AFTER the replace (already-split body).
    #   2. The `already_split` re-check below operates on the
    #      content read INSIDE the locked critical section. A
    #      second worker that flocks on the NEW inode reads the
    #      already-split body and short-circuits via
    #      `already_split`.
    # The lock fd does NOT have to match the read fd — `flock`
    # locks the file description, not the path; the in-section
    # content check closes the gap.
    fd = os.open(str(tasks_yaml_path), os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)

        with open(tasks_yaml_path, encoding="utf-8") as f:
            data = yaml_obj.load(f)

        if not isinstance(data, dict) or "tasks" not in data:
            _log_yellow(
                f"  [splitter] tasks.yaml has no 'tasks' key; "
                f"cannot apply split."
            )
            return None
        tasks_seq = data["tasks"]
        if not isinstance(tasks_seq, list):
            _log_yellow(
                f"  [splitter] tasks.yaml 'tasks' is not a list; "
                f"cannot apply split."
            )
            return None

        parent_idx = next(
            (
                i for i, t in enumerate(tasks_seq)
                if t.get("id") == parent_id
            ),
            None,
        )
        if parent_idx is None:
            _log_yellow(
                f"  [splitter] task {parent_id} not found in "
                f"tasks.yaml; cannot apply split."
            )
            return None

        # Re-check status under lock — covers the race where a sibling
        # worker split this same parent between our state.get(...) at
        # entry and the lock acquisition here. We probe the in-file
        # signature of a successful split: any subsequent task entry
        # with `parent_task == parent_id`. The parent's own YAML
        # entry is preserved unchanged (see plan §1 item 4), so we
        # cannot key on a parent-side flag.
        already_split = any(
            t.get("parent_task") == parent_id
            for t in tasks_seq[parent_idx + 1:]
        )
        if already_split:
            _log_yellow(
                f"  [splitter] task {parent_id} already has "
                f"children in tasks.yaml; skipping (lock-race)."
            )
            return None

        new_depth = current_depth + 1
        parent_phase = parent_task.get("phase")
        new_child_ids: list[str] = []
        for offset, child in enumerate(children):
            entry = _render_child_entry(
                child=child,
                parent_id=parent_id,
                split_depth=new_depth,
                parent_phase=parent_phase,
            )
            tasks_seq.insert(parent_idx + 1 + offset, entry)
            new_child_ids.append(str(entry["id"]))

        # Rewire ALL tasks (except the parent itself and the freshly
        # inserted children) whose `depends_on` contains the parent's
        # id. The dependency graph in tasks.yaml is NOT required to be
        # listed in topological order — `validate_dependencies` is a
        # topological sort that ignores list-position. A task at
        # list-index 0 is allowed to forward-reference a parent at
        # list-index 1, so a position-based scan that only inspects
        # entries AFTER the inserted children would miss legal
        # forward-referring dependers and leave them stuck waiting on
        # `state.get(parent)["status"] == "done"` (which never happens
        # because the parent is `"split"`, not `"done"`).
        last_child_id = new_child_ids[-1]
        inserted_ids = set(new_child_ids)
        # `parent_idx` is invariant under our inserts: every
        # `tasks_seq.insert(parent_idx + 1 + offset, entry)` adds AFTER
        # the parent slot, so the parent stays at parent_idx.
        for i, t in enumerate(tasks_seq):
            if i == parent_idx:
                continue
            if str(t.get("id")) in inserted_ids:
                continue
            deps = t.get("depends_on")
            if not deps:
                continue
            new_deps = [
                last_child_id if str(d) == str(parent_id) else d
                for d in deps
            ]
            if new_deps != list(deps):
                t["depends_on"] = new_deps

        # Atomic write: tempfile + os.replace in same dir.
        # `tempfile.mkstemp` produces 0600 files; copy mode from the
        # original so `git diff` doesn't show a stat-bit change in
        # addition to the content edit. `shutil.copymode` is best-
        # effort: a missing original (impossible at this point — we
        # just read it) or filesystem without mode-bit support
        # (e.g., a future cifs/exfat mount) silently degrades to
        # 0600; acceptable.
        parent_dir = tasks_yaml_path.parent
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(parent_dir), suffix=".tmp",
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                yaml_obj.dump(data, f)
            try:
                shutil.copymode(str(tasks_yaml_path), tmp_path)
            except OSError:
                pass
            os.replace(tmp_path, tasks_yaml_path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        return new_child_ids
    except OSError as exc:
        _log_yellow(
            f"  [splitter] tasks.yaml mutation failed "
            f"({type(exc).__name__}: {exc}); leaving blocked."
        )
        return None
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            os.close(fd)
        except OSError:
            pass


def run_splitter(
    task: dict,
    state: State,
    file_defaults: dict,
    cli: dict,
    *,
    resolved: dict,
    plans_dir: Path,
    tasks_yaml_path: Path | None,
) -> None:
    """See module docstring. Never raises."""
    tid = task["id"]
    try:
        # 1. Idempotency. NOT a production-path early-out — see plan
        #    §4 pin 10. Production-path same-parent idempotency lives in
        #    `_apply_split_to_yaml`'s `already_split` re-check. This
        #    entry guard exists for direct callers (notably the (f)
        #    test and any future programmatic invocation that bypasses
        #    the runner). On every production trigger path the runner
        #    has mutated state to 'planning'/'reviewing'/.../'blocked'
        #    before we get here, so this `== "split"` check is
        #    effectively a no-op.
        if state.get(tid)["status"] == "split":
            _log_dim(f"  [splitter] task {tid} already split; skipping.")
            return

        # 2. tasks.yaml path required to mutate; absent in some test
        #    callers and any future programmatic invocation that
        #    didn't thread the path. Silent-yellow-warn on absent.
        if tasks_yaml_path is None:
            _log_yellow(
                f"  [splitter] tasks_yaml_path not set; cannot apply "
                f"split for task {tid}. (Pass via execute_task or "
                f"run_slate.)"
            )
            return

        # 3. Recursion guard (plan §4 pin 4).
        current_depth = int(task.get("split_depth") or 0)
        _max_depth_v = resolved["max_split_depth"]["value"]
        max_depth = int(_max_depth_v) if _max_depth_v is not None else 1
        if current_depth >= max_depth:
            _log_yellow(
                f"  [splitter] task {tid} already at split_depth="
                f"{current_depth} (max={max_depth}); skipping split."
            )
            return

        # 4. Build prompt (failure_context + persona + decisions + body).
        prompt = _build_prompt(
            task=task, state=state, resolved=resolved,
            plans_dir=plans_dir,
        )

        # 5. Resolve integer run_claude kwargs with explicit-None-check
        #    (preserves legitimate-zero semantics — see plan §4 pins 4
        #    and 11). `max_rate_limit_retries=0` is a documented
        #    rate-limit-awareness opt-out (claude.py:355-358); the
        #    `or N` shape would silently coerce that to N inside the
        #    splitter only.
        _mrr_v = resolved["max_rate_limit_retries"]["value"]
        max_rr = int(_mrr_v) if _mrr_v is not None else 3
        _bbf_v = resolved["rate_limit_base_backoff_seconds"]["value"]
        base_backoff = int(_bbf_v) if _bbf_v is not None else 60

        # 6. Invoke Claude.
        code, text = run_claude(
            prompt,
            model=_SPLITTER_MODEL,
            effort=_SPLITTER_EFFORT,
            max_turns=_SPLITTER_MAX_TURNS,
            timeout_minutes=_SPLITTER_TIMEOUT_MINUTES,
            skip_permissions=cli.get("skip_permissions", False),
            verbose=cli.get("verbose", False),
            max_rate_limit_retries=max_rr,
            rate_limit_base_backoff_seconds=base_backoff,
            usage_jsonl_path=str(
                plans_dir / f"task-{tid}" / "usage.jsonl"
            ),
            phase="splitter",
        )

        # 7. Parse appendix.
        decision, reason, children = _parse_appendix(text or "", tid)
        if decision is None:
            return  # _parse_appendix already logged.

        if decision == "keep":
            _log_green(
                f"  [splitter] task {tid}: agent judged the task "
                f"tractable; leaving blocked for human review. "
                f"Reason: {reason!r}"
            )
            return

        if decision != "split":
            _log_yellow(
                f"  [splitter] task {tid}: unknown decision "
                f"{decision!r}; skipping."
            )
            return

        # 8. Validate children.
        if len(children) < 2:
            _log_yellow(
                f"  [splitter] task {tid}: decision='split' but "
                f"only {len(children)} child(ren) — treating as keep."
            )
            return

        # 9. Mutate tasks.yaml under file lock.
        new_child_ids = _apply_split_to_yaml(
            tasks_yaml_path=tasks_yaml_path,
            parent_task=task,
            children=children,
            current_depth=current_depth,
        )
        if new_child_ids is None:
            return  # _apply_split_to_yaml already logged.

        # 10. Flip parent state.
        state.set_status(
            tid, "split",
            split_children=new_child_ids,
            block_reason=None,
        )
        _log_green(
            f"  [splitter] task {tid} -> split into {new_child_ids}. "
            f"Re-launch the slate to execute the children."
        )
    except Exception as exc:  # noqa: BLE001 — fail-soft contract.
        _log_yellow(
            f"  [splitter] task {tid} unexpected error: "
            f"{type(exc).__name__}: {exc}; leaving blocked."
        )
