"""Cross-task gate-result aggregation (T12).

Reads project-supplied aggregations.yaml and walks
{plans_dir}/task-*/gates/ to produce read-only post-hoc tallies.
Two query modes: regex over <gate>.txt (Mode A; works on every
gate) and field extraction over <gate>.json (Mode B; requires
T11 structured-findings appendix).

aggregations.yaml shape (bare list at top level):

    - name: untranslated-by-locale
      gate: untranslated-strings
      extract: "(?im)missing key: (\\S+)"
    - name: security-by-severity
      gate: security-auditor
      field: severity
    - name: security-by-category-and-severity
      gate: security-auditor
      fields: [category, severity]

Public API:
    AggregationSpec
    AggregationConfigError
    AggregationRuntimeError
    find_aggregations_file(start_dir=None) -> Path | None
    load_aggregations(path) -> list[AggregationSpec]
    compute_aggregation(spec, plans_dir) -> list[dict]
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import yaml
except ImportError:
    print("Error: PyYAML required. pip install pyyaml", file=sys.stderr)
    sys.exit(1)


_SAFE_GATE_NAME_RE = re.compile(r"[A-Za-z0-9_-]+")
_SAFE_AGGREGATION_NAME_RE = re.compile(r"[A-Za-z0-9_-]+")


class AggregationConfigError(Exception):
    """Per-entry shape violation. Server returns 400."""


class AggregationRuntimeError(Exception):
    """YAML parse / regex compile / IO failure. Server returns 500."""


@dataclass(frozen=True)
class AggregationSpec:
    name: str
    gate: str
    extract: str | None = None
    field: str | None = None
    fields: tuple[str, ...] | None = None

    @property
    def mode(self) -> str:
        if self.extract is not None:
            return "regex"
        if self.field is not None:
            return "field"
        return "fields"


def find_aggregations_file(
    start_dir: Path | None = None,
) -> Path | None:
    base = start_dir if start_dir is not None else Path.cwd()
    for cand in ["aggregations.yaml", "docs/aggregations.yaml"]:
        p = base / cand
        if p.exists():
            return p
    return None


def load_aggregations(path: Path) -> list[AggregationSpec]:
    """Parse + per-entry validate. Raises:
      - AggregationRuntimeError: YAML parse / IO failure
      - AggregationConfigError: any per-entry shape violation
    """
    try:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as exc:
        raise AggregationRuntimeError(
            f"aggregations.yaml: cannot parse {path}: {exc}"
        ) from exc

    if raw is None:
        return []

    if not isinstance(raw, list):
        raise AggregationConfigError(
            f"aggregations.yaml: top-level must be a list of "
            f"entries; got {type(raw).__name__}"
        )

    specs: list[AggregationSpec] = []
    seen_names: set[str] = set()
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise AggregationConfigError(
                f"aggregations.yaml entry #{i}: must be a mapping; "
                f"got {type(entry).__name__}"
            )

        name = entry.get("name")
        if not isinstance(name, str) or not name:
            raise AggregationConfigError(
                f"aggregations.yaml entry #{i}: 'name' is required "
                f"(string)"
            )
        if not _SAFE_AGGREGATION_NAME_RE.fullmatch(name):
            raise AggregationConfigError(
                f"aggregations.yaml entry #{i}: 'name' {name!r} "
                f"must match [A-Za-z0-9_-]+"
            )
        if name in seen_names:
            raise AggregationConfigError(
                f"aggregations.yaml: duplicate aggregation name "
                f"{name!r}"
            )
        seen_names.add(name)

        gate = entry.get("gate")
        if not isinstance(gate, str) or not gate:
            raise AggregationConfigError(
                f"aggregations.yaml entry {name!r}: 'gate' is "
                f"required (string)"
            )
        if not _SAFE_GATE_NAME_RE.fullmatch(gate):
            raise AggregationConfigError(
                f"aggregations.yaml entry {name!r}: 'gate' {gate!r} "
                f"must match [A-Za-z0-9_-]+"
            )

        present = [k for k in ("extract", "field", "fields") if k in entry]
        if len(present) != 1:
            if not present:
                raise AggregationConfigError(
                    f"aggregations.yaml entry {name!r}: must set "
                    f"exactly one of 'extract', 'field', or 'fields'; "
                    f"got none"
                )
            raise AggregationConfigError(
                f"aggregations.yaml entry {name!r}: must set "
                f"exactly one of 'extract', 'field', or 'fields'; "
                f"got {present}"
            )

        extract = entry.get("extract")
        field = entry.get("field")
        fields = entry.get("fields")

        if extract is not None and not isinstance(extract, str):
            raise AggregationConfigError(
                f"aggregations.yaml entry {name!r}: 'extract' must "
                f"be a string"
            )
        if field is not None and not isinstance(field, str):
            raise AggregationConfigError(
                f"aggregations.yaml entry {name!r}: 'field' must "
                f"be a string"
            )
        if fields is not None:
            if (
                not isinstance(fields, list)
                or not fields
                or not all(isinstance(x, str) for x in fields)
            ):
                raise AggregationConfigError(
                    f"aggregations.yaml entry {name!r}: 'fields' "
                    f"must be a non-empty list of strings"
                )
            fields = tuple(fields)

        specs.append(AggregationSpec(
            name=name, gate=gate,
            extract=extract, field=field, fields=fields,
        ))
    return specs


def _natural_task_key(tid: str) -> tuple:
    """Natural sort: numeric IDs sort numerically; slug IDs sort
    lexically; all numeric IDs come before all slug IDs.
    """
    if tid.isdigit():
        return (0, int(tid))
    return (1, tid)


def _walk_task_dirs(plans_dir: Path) -> list[tuple[str, Path]]:
    """Return [(task_id, task_dir), ...] sorted naturally."""
    out: list[tuple[str, Path]] = []
    if not plans_dir.exists():
        return out
    for d in plans_dir.glob("task-*"):
        if not d.is_dir():
            continue
        tid = d.name[len("task-"):]
        if not tid:
            continue
        out.append((tid, d))
    out.sort(key=lambda pair: _natural_task_key(pair[0]))
    return out


def _bucket_increment(
    counts: dict[str, dict[str, int]], key: str, tid: str,
) -> None:
    bucket = counts.setdefault(key, {})
    bucket[tid] = bucket.get(tid, 0) + 1


def _extract_mode_a(
    spec: AggregationSpec, plans_dir: Path,
) -> dict[str, dict[str, int]]:
    try:
        pattern = re.compile(spec.extract)
    except re.error as exc:
        raise AggregationRuntimeError(
            f"aggregation {spec.name!r}: bad regex "
            f"{spec.extract!r}: {exc}"
        ) from exc
    counts: dict[str, dict[str, int]] = {}
    for tid, task_dir in _walk_task_dirs(plans_dir):
        txt_path = task_dir / "gates" / f"{spec.gate}.txt"
        if not txt_path.exists():
            continue
        try:
            text = txt_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for m in pattern.finditer(text):
            key = m.group(1) if m.groups() else m.group(0)
            _bucket_increment(counts, key, tid)
    return counts


def _extract_mode_b(
    spec: AggregationSpec, plans_dir: Path,
) -> dict[str, dict[str, int]]:
    keys: tuple[str, ...] = (
        (spec.field,) if spec.field is not None else spec.fields
    )
    counts: dict[str, dict[str, int]] = {}
    for tid, task_dir in _walk_task_dirs(plans_dir):
        json_path = task_dir / "gates" / f"{spec.gate}.json"
        if not json_path.exists():
            continue
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        findings = (
            payload.get("findings") if isinstance(payload, dict) else None
        )
        if not isinstance(findings, list):
            continue
        for finding in findings:
            if not isinstance(finding, dict):
                continue
            slots: list[str] = []
            for k in keys:
                v = finding.get(k)
                slots.append("(unknown)" if v is None else str(v))
            _bucket_increment(counts, ":".join(slots), tid)
    return counts


def compute_aggregation(
    spec: AggregationSpec, plans_dir: Path,
) -> list[dict]:
    """Walk plans_dir/task-*/gates/, extract findings per the
    spec's mode, return rows. Rows are dicts of shape:

        {"finding": str, "total_count": int,
         "per_task": {tid: count}}

    Sorted by (-total_count, finding). per_task keys are naturally
    sorted (numeric IDs before slug IDs; numeric IDs in numeric
    order).
    """
    if spec.extract is not None:
        counts = _extract_mode_a(spec, plans_dir)
    else:
        counts = _extract_mode_b(spec, plans_dir)

    rows: list[dict] = []
    for finding, per_task in counts.items():
        ordered = {
            tid: per_task[tid]
            for tid in sorted(per_task.keys(), key=_natural_task_key)
        }
        rows.append({
            "finding": finding,
            "total_count": sum(per_task.values()),
            "per_task": ordered,
        })
    rows.sort(key=lambda r: (-r["total_count"], r["finding"]))
    return rows


__all__ = [
    "AggregationSpec",
    "AggregationConfigError",
    "AggregationRuntimeError",
    "find_aggregations_file",
    "load_aggregations",
    "compute_aggregation",
]
