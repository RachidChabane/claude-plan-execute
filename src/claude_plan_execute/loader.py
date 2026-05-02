"""tasks.yaml / tasks.json discovery, loading, and validation."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("Error: PyYAML is required. Install with: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

from claude_plan_execute.colors import red, yellow
from claude_plan_execute.schema import (
    TasksFile,
    ValidationError,
    format_validation_error,
)


def find_tasks_file(
    filename: str = "tasks.yaml",
    start_dir: Path | None = None,
) -> Path | None:
    base = start_dir if start_dir is not None else Path.cwd()
    for candidate in [filename, f"docs/{filename}"]:
        p = base / candidate
        if p.exists():
            return p
    return None


def load_tasks_data(path: Path) -> dict:
    """Raw read of a tasks file. No validation, no sys.exit. Returns the
    parsed dict (or {} for an empty YAML file)."""
    with open(path) as f:
        if path.suffix == ".json":
            return json.load(f)
        return yaml.safe_load(f) or {}


def load_tasks_file(path: Path) -> TasksFile:
    """Load and validate the tasks file. Returns the validated TasksFile.
    Calls sys.exit(1) on validation errors with a formatted message including
    file path, key path, and (for typos) suggested correction. Emits a
    deprecation warning on v2 files that use legacy `*_timeout_minutes` keys.
    """
    raw = load_tasks_data(path)

    if not isinstance(raw, dict) or "tasks" not in raw:
        print(red(f"Error: {path} must contain a 'tasks' key"), file=sys.stderr)
        sys.exit(1)

    if raw.get("version") is None:
        print(yellow(f"Warning: {path} has no 'version' field. Assuming version 1."))

    try:
        tasks_file = TasksFile.model_validate(raw)
    except ValidationError as exc:
        print(red(format_validation_error(exc, path)), file=sys.stderr)
        sys.exit(1)

    # Deprecation warnings: v2 files that still set legacy timeout/turns/rounds
    # keys at defaults level. v1 files are silent.
    if tasks_file.version == 2:
        legacy_pairs = {
            "plan_timeout_minutes": "caps.max_minutes_per_phase",
            "review_timeout_minutes": "caps.max_minutes_per_phase",
            "implement_timeout_minutes": "caps.max_minutes_per_phase",
            "max_review_rounds": "caps.max_review_rounds",
            "implement_max_turns": "caps.max_implement_turns",
        }
        defaults_set = tasks_file.defaults.model_dump(exclude_none=True)
        for legacy, replacement in legacy_pairs.items():
            if legacy in defaults_set:
                print(yellow(
                    f"Warning: {path}: 'defaults.{legacy}' is deprecated; "
                    f"use defaults.{replacement} instead."
                ))

    # Version-aware default flip warning (task 6 §D3 / §D4).
    # v1 / no-version files default to the legacy "auto_approve"
    # behavior; tell users so they can opt into the safer "fail" or
    # migrate to v2.
    if (
        (tasks_file.version is None or tasks_file.version == 1)
        and tasks_file.defaults.on_max_review_rounds is None
    ):
        print(yellow(
            f"Warning: {path}: version 1 defaults to "
            f"'on_max_review_rounds: auto_approve' (unsafe for autonomous "
            f"runs). The version 2 default is 'fail'. Set "
            f"'defaults.on_max_review_rounds: fail' explicitly or migrate "
            f"to 'version: 2'."
        ))

    return tasks_file


def effective_defaults(tasks_file: TasksFile) -> dict:
    """Flatten the validated defaults into the dict the resolver consumes as
    the project-defaults layer.

    - Drops nested `caps` and `task_class_models` (not resolver-consumable).
    - Expands `caps.max_minutes_per_phase` into three legacy resolver keys
      (`plan_timeout_minutes`, `review_timeout_minutes`,
      `implement_timeout_minutes`). Caps wins over a directly-set legacy
      value (S2). Deprecation warning was already emitted at load.
    - Expands `caps.max_review_rounds`, `caps.max_implement_turns`,
      `caps.max_attempts` into resolver-friendly keys.
    - Injects `on_max_review_rounds: "auto_approve"` for v1 / no-version
      files when the user hasn't set it explicitly (task 6 §D1).
    - Injects `gates: ["tests"]` for v1 / no-version files when the
      user hasn't set it explicitly (task 12 §D6).
    """
    out = tasks_file.defaults.model_dump(exclude_none=True)
    out.pop("caps", None)
    out.pop("task_class_models", None)

    caps = tasks_file.defaults.caps
    if caps is not None:
        if caps.max_minutes_per_phase is not None:
            out["plan_timeout_minutes"] = caps.max_minutes_per_phase
            out["review_timeout_minutes"] = caps.max_minutes_per_phase
            out["implement_timeout_minutes"] = caps.max_minutes_per_phase
        if caps.max_review_rounds is not None:
            out["max_review_rounds"] = caps.max_review_rounds
        if caps.max_implement_turns is not None:
            out["implement_max_turns"] = caps.max_implement_turns
        if caps.max_attempts is not None:
            out["max_attempts"] = caps.max_attempts

    if "on_max_review_rounds" not in out:
        if tasks_file.version is None or tasks_file.version == 1:
            out["on_max_review_rounds"] = "auto_approve"

    if "gates" not in out:
        if tasks_file.version is None or tasks_file.version == 1:
            out["gates"] = ["tests"]

    return out


def load_persona_fragment(path: Path | str | None) -> str:
    """Read a persona markdown fragment.

    Returns the file's UTF-8 contents when the path is set, the file
    exists, is a regular file, and has at most 800 lines. Returns
    "" silently when the path is None/empty, the file is missing, or
    the path is a directory. Returns "" with a stdout warning when
    the file is over 800 lines (long personas dilute every prompt;
    the cap is protective). Per-fragment cap, not aggregate — the
    caller may compose multiple fragments via prompts.build_persona_section.

    Path resolution is CWD-relative when `path` is a relative string;
    absolute paths and Path objects pass through unchanged. (See plan
    §D8.)
    """
    if not path:
        return ""
    p = Path(path) if not isinstance(path, Path) else path
    if not p.exists() or not p.is_file():
        return ""
    text = p.read_text(encoding="utf-8")
    line_count = len(text.splitlines())
    if line_count > 800:
        print(yellow(
            f"Warning: persona file {p} has {line_count} lines (>800); "
            f"skipping. Long personas dilute every prompt — split into "
            f"'phases[].defaults.persona_extra' overlays or trim."
        ))
        return ""
    return text


# ---------------------------------------------------------------------------
# Decisions log (task 8)
# ---------------------------------------------------------------------------

# Matches a leading YAML frontmatter block delimited by lines that
# contain only `---`. The body is whatever follows the closing `---`.
# Multiline (`re.S`) so `.` matches newlines inside the block.
_FRONTMATTER_RE = re.compile(
    r"\A---\s*\n(?P<fm>.*?)\n---\s*\n(?P<body>.*)\Z",
    re.S,
)

# Matches the first `# <title>` ATX heading on a line of its own.
_FIRST_H1_RE = re.compile(r"(?m)^#\s+(?P<title>.+?)\s*$")


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Split a markdown document into (frontmatter_dict, body).

    Detects the standard `---\\n…\\n---\\n` YAML frontmatter prefix.
    Returns `({}, text)` if no frontmatter delimiter is present at
    the start of the document. Raises `yaml.YAMLError` on malformed
    YAML inside a delimited block — the caller handles the error
    (the index-builder logs a warning and skips the file).
    """
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    fm_raw = m.group("fm")
    body = m.group("body")
    data = yaml.safe_load(fm_raw) or {}
    if not isinstance(data, dict):
        # Frontmatter that is a list / scalar / null is rejected as
        # malformed — same outcome as a YAML parse error.
        raise yaml.YAMLError(
            f"frontmatter is not a mapping: {type(data).__name__}"
        )
    return data, body


def load_decisions_index(decisions_dir: Path | str | None) -> str:
    """Read `<decisions_dir>/index.md` for prompt injection.

    Returns "" silently when:
      - decisions_dir is None or "".
      - decisions_dir does not exist or is not a directory.
      - <decisions_dir>/index.md does not exist or is not a regular file.

    Returns "" with a stdout warning when index.md exceeds the
    200-line cap (§D5).

    Returns the file's UTF-8 contents verbatim (no transformation,
    no frontmatter parsing) otherwise.
    """
    if not decisions_dir:
        return ""
    d = Path(decisions_dir) if not isinstance(decisions_dir, Path) else decisions_dir
    if not d.exists() or not d.is_dir():
        return ""
    index_path = d / "index.md"
    if not index_path.exists() or not index_path.is_file():
        return ""
    text = index_path.read_text(encoding="utf-8")
    line_count = len(text.splitlines())
    if line_count > 200:
        print(yellow(
            f"Warning: decisions index {index_path} has {line_count} "
            f"lines (>200); skipping. The index is a one-line-per-entry "
            f"summary — trim to under 200, or run "
            f"'claude-plan-execute decisions index' after archiving "
            f"older entries."
        ))
        return ""
    return text


def _extract_title(body: str, fallback_stem: str) -> str:
    """Pull a title from a decision file's body. Strategy:

    1. First H1 ATX heading (`# Foo`) → return its text.
    2. Otherwise, fall back to the filename stem with a leading
       `\\d+-` (or `\\d+_`) prefix stripped, dashes/underscores
       replaced with spaces. (`001-postgres-gin-bm25` → `postgres gin
       bm25`.)
    """
    m = _FIRST_H1_RE.search(body)
    if m:
        return m.group("title").strip()
    stem = re.sub(r"^\d+[-_]+", "", fallback_stem)
    return stem.replace("-", " ").replace("_", " ").strip() or fallback_stem


def scan_decisions(decisions_dir: Path) -> list[dict]:
    """Walk `<decisions_dir>/*.md` (excluding `index.md`), parse
    frontmatter, and return a list of normalized entry dicts.

    Each entry: {id: int, status: str, title: str, scope: list,
    filename: str}.

    Skips with a stdout warning:
      - files that fail YAML parse.
      - files without frontmatter at all.
      - files where `id` is missing or not int-coercible.
      - files where `status` is missing.

    Files where `status` is unknown (not accepted/superseded/
    deprecated) are kept (treated as accepted in the renderer) but
    a warning is printed.

    Sorted by `id` ascending. Stable for ties (input filesystem
    order; ties shouldn't happen in well-formed decision logs).
    """
    entries: list[dict] = []
    for path in sorted(decisions_dir.glob("*.md")):
        if path.name == "index.md":
            continue
        try:
            text = path.read_text(encoding="utf-8")
            fm, body = parse_frontmatter(text)
        except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
            # UnicodeDecodeError (a ValueError subclass — NOT caught by
            # OSError) covers binary or non-UTF-8 .md files dropped into
            # decisions_dir; without it, a single malformed file would
            # crash the entire `decisions index` command.
            print(yellow(
                f"Warning: skipping {path}: cannot parse frontmatter "
                f"({type(exc).__name__}: {exc})"
            ))
            continue
        if not fm:
            print(yellow(
                f"Warning: skipping {path}: no YAML frontmatter."
            ))
            continue
        if "id" not in fm:
            print(yellow(
                f"Warning: skipping {path}: frontmatter missing 'id'."
            ))
            continue
        try:
            entry_id = int(fm["id"])
        except (TypeError, ValueError):
            print(yellow(
                f"Warning: skipping {path}: 'id' is not an integer "
                f"(got {fm['id']!r})."
            ))
            continue
        status = fm.get("status")
        if not status:
            print(yellow(
                f"Warning: skipping {path}: frontmatter missing 'status'."
            ))
            continue
        if status not in ("accepted", "superseded", "deprecated"):
            print(yellow(
                f"Warning: {path} has unknown status {status!r}; "
                f"treating as 'accepted' for rendering."
            ))
        title = fm.get("title")
        if not title:
            title = _extract_title(body, path.stem)
        scope = fm.get("scope") or []
        if not isinstance(scope, list):
            scope = [scope]
        entries.append({
            "id": entry_id,
            "status": status,
            "title": title,
            "scope": scope,
            "filename": path.name,
        })
    entries.sort(key=lambda e: e["id"])
    return entries


_DECISIONS_INDEX_BANNER = (
    "<!-- Generated by `claude-plan-execute decisions index`. "
    "Edits will be overwritten. -->"
)


def render_decisions_index(entries: list[dict]) -> str:
    """Render entries as the markdown body of `index.md`.

    Format:
        <!-- Generated by ... -->

        - [001](filename.md) — title
        - ~~[002](filename.md) — title~~ (superseded)
        - [003](filename.md) — title

    Trailing newline. Empty entries → banner-only output (still
    trailing newline).
    """
    lines: list[str] = [_DECISIONS_INDEX_BANNER, ""]
    for e in entries:
        link = f"[{e['id']:03d}]({e['filename']})"
        body = f"{link} — {e['title']}"
        if e["status"] in ("superseded", "deprecated"):
            lines.append(f"- ~~{body}~~ ({e['status']})")
        else:
            lines.append(f"- {body}")
    return "\n".join(lines) + "\n"
