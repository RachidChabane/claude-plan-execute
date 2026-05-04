"""Tests for src/claude_plan_execute/splitter.py."""
from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest
from ruamel.yaml import YAML

from claude_plan_execute.splitter import run_splitter
from claude_plan_execute.state import State, make_failure_record


# --- Fixtures ---------------------------------------------------------------


@pytest.fixture(autouse=True)
def _stub_persona_and_decisions(monkeypatch):
    """Keep splitter side-effects contained: skip persona / decisions
    file reads in unit tests."""
    monkeypatch.setattr(
        "claude_plan_execute.splitter.load_persona_fragment",
        lambda _: "",
    )
    monkeypatch.setattr(
        "claude_plan_execute.splitter.load_decisions_index",
        lambda _: "",
    )


def _write_tasks_yaml(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")


def _make_state_with_failed_task(tmp_path: Path, tid, *, attempts=None):
    state = State(tmp_path / "state.json")
    state.set_status(tid, "blocked", block_reason="max_review_rounds_exhausted")
    for a in attempts or []:
        state.record_failure(tid, a)
    return state


def _stub_run_claude(monkeypatch, response_text: str):
    captured = []

    def stub(*args, **kwargs):
        captured.append((args, kwargs))
        return 0, response_text

    monkeypatch.setattr(
        "claude_plan_execute.splitter.run_claude", stub,
    )
    return captured


def _split_appendix(children, *, decision="split", reason="r"):
    body = {"decision": decision, "reason": reason, "children": children}
    return (
        f"## Decision\n\nfoo\n\n"
        f"<!-- BEGIN_FINDINGS_JSON -->\n"
        f"{json.dumps(body)}\n"
        f"<!-- END_FINDINGS_JSON -->\n"
    )


def _resolved(**overrides):
    base = {
        "auto_split_on_max_rounds": {"value": True, "from": "test"},
        "auto_split_failure_classes": {"value": [
            "review_rejected_after_max_rounds",
            "implementer_max_turns",
        ], "from": "test"},
        "max_split_depth": {"value": 1, "from": "test"},
        "persona_file": {"value": None, "from": "test"},
        "decisions_dir": {"value": None, "from": "test"},
        "max_rate_limit_retries": {"value": 0, "from": "test"},
        "rate_limit_base_backoff_seconds": {"value": 60, "from": "test"},
    }
    base.update({k: {"value": v, "from": "test"} for k, v in overrides.items()})
    return base


def _setup_split_scenario(tmp_path, *, tid=1, parent_yaml_id=1):
    """Create tasks.yaml + plans dir + state with attempt for a parent."""
    yaml_path = tmp_path / "tasks.yaml"
    _write_tasks_yaml(yaml_path,
        "version: 2\n"
        "tasks:\n"
        f"  - id: {parent_yaml_id}\n"
        "    title: T1\n"
        "    description: D\n",
    )
    plans_dir = tmp_path / "plans"
    (plans_dir / f"task-{tid}").mkdir(parents=True)
    (plans_dir / f"task-{tid}" / "plan.md").write_text("(plan body content)")
    (plans_dir / f"task-{tid}" / "review-1.md").write_text("nits round 1")
    state = _make_state_with_failed_task(plans_dir, tid, attempts=[
        make_failure_record(
            failure_class="review_rejected_after_max_rounds",
            phase="review", reason="...",
        ),
    ])
    return yaml_path, plans_dir, state


# --- Tests -----------------------------------------------------------------


def test_run_splitter_calls_run_claude_with_failure_context_and_plan_and_reviews(
    monkeypatch, tmp_path,
):
    """Description (a): the prompt carries failure-context, plan body,
    and review history. Asserted via structural inclusion (not exact
    substring) so this test is robust to template tweaks."""
    yaml_path, plans_dir, state = _setup_split_scenario(tmp_path)
    children = [
        {"id": "1a", "title": "X", "description": "DX", "depends_on": []},
        {"id": "1b", "title": "Y", "description": "DY",
         "depends_on": ["1a"]},
    ]
    captured = _stub_run_claude(monkeypatch, _split_appendix(children))
    task = {"id": 1, "title": "T1", "description": "D"}

    run_splitter(
        task=task, state=state, file_defaults={}, cli={},
        resolved=_resolved(), plans_dir=plans_dir,
        tasks_yaml_path=yaml_path,
    )

    assert len(captured) == 1
    prompt = captured[0][0][0]   # first call, first positional arg
    assert "[FAILURE CONTEXT]" in prompt
    assert "(plan body content)" in prompt
    assert "review-1.md" in prompt


def test_run_splitter_well_formed_split_appendix_applies_changes(
    monkeypatch, tmp_path,
):
    """Pin §6.3: a well-formed split appendix mutates tasks.yaml AND
    sets the parent's state status to 'split'."""
    yaml_path = tmp_path / "tasks.yaml"
    _write_tasks_yaml(yaml_path,
        "version: 2\n"
        "tasks:\n"
        "  - id: 1\n"
        "    title: T1\n"
        "    description: D\n"
        "  - id: 2\n"
        "    title: T2\n"
        "    description: D\n"
        "    depends_on: [1]\n",
    )
    plans_dir = tmp_path / "plans"
    (plans_dir / "task-1").mkdir(parents=True)
    (plans_dir / "task-1" / "plan.md").write_text("(plan)")
    (plans_dir / "task-1" / "review-1.md").write_text("nits")
    state = _make_state_with_failed_task(plans_dir, 1, attempts=[
        make_failure_record(
            failure_class="review_rejected_after_max_rounds",
            phase="review", reason="...",
        ),
    ])
    children = [
        {"id": "1a", "title": "X", "description": "DX",
         "depends_on": [], "phase": None},
        {"id": "1b", "title": "Y", "description": "DY",
         "depends_on": ["1a"], "phase": None},
    ]
    _stub_run_claude(monkeypatch, _split_appendix(children))
    task = {"id": 1, "title": "T1", "description": "D"}

    run_splitter(
        task=task, state=state, file_defaults={}, cli={},
        resolved=_resolved(), plans_dir=plans_dir,
        tasks_yaml_path=yaml_path,
    )

    yaml_obj = YAML(typ="rt")
    with open(yaml_path) as f:
        data = yaml_obj.load(f)
    ids = [t["id"] for t in data["tasks"]]
    assert ids == [1, "1a", "1b", 2]
    assert data["tasks"][1]["parent_task"] == 1
    assert data["tasks"][1]["split_depth"] == 1
    assert list(data["tasks"][3]["depends_on"]) == ["1b"]
    entry = state.get(1)
    assert entry["status"] == "split"
    assert entry["split_children"] == ["1a", "1b"]
    assert entry.get("block_reason") is None


def test_run_splitter_malformed_appendix_logs_warning_and_no_mutation(
    monkeypatch, tmp_path,
):
    """Description (b): malformed JSON inside the markers triggers a
    yellow warning and leaves the file untouched."""
    yaml_path, plans_dir, state = _setup_split_scenario(tmp_path)
    original = yaml_path.read_text()
    response = (
        "## Decision\n\nfoo\n\n"
        "<!-- BEGIN_FINDINGS_JSON -->\n"
        "{not-json,,,\n"
        "<!-- END_FINDINGS_JSON -->\n"
    )
    _stub_run_claude(monkeypatch, response)
    task = {"id": 1, "title": "T1", "description": "D"}

    run_splitter(
        task=task, state=state, file_defaults={}, cli={},
        resolved=_resolved(), plans_dir=plans_dir,
        tasks_yaml_path=yaml_path,
    )

    assert yaml_path.read_text() == original
    entry = state.get(1)
    assert entry["status"] == "blocked"


def test_run_splitter_missing_appendix_logs_warning_and_no_mutation(
    monkeypatch, tmp_path,
):
    """Description (b) extension: no marker pair → no mutation."""
    yaml_path, plans_dir, state = _setup_split_scenario(tmp_path)
    original = yaml_path.read_text()
    _stub_run_claude(monkeypatch, "no markers at all here")
    task = {"id": 1, "title": "T1", "description": "D"}

    run_splitter(
        task=task, state=state, file_defaults={}, cli={},
        resolved=_resolved(), plans_dir=plans_dir,
        tasks_yaml_path=yaml_path,
    )

    assert yaml_path.read_text() == original
    entry = state.get(1)
    assert entry["status"] == "blocked"


def test_run_splitter_keep_decision_does_not_mutate_yaml(
    monkeypatch, tmp_path,
):
    """Description (c): decision='keep' leaves tasks.yaml untouched
    and parent's status stays 'blocked'."""
    yaml_path, plans_dir, state = _setup_split_scenario(tmp_path)
    original = yaml_path.read_text()
    _stub_run_claude(monkeypatch, _split_appendix([], decision="keep"))
    task = {"id": 1, "title": "T1", "description": "D"}

    run_splitter(
        task=task, state=state, file_defaults={}, cli={},
        resolved=_resolved(), plans_dir=plans_dir,
        tasks_yaml_path=yaml_path,
    )

    assert yaml_path.read_text() == original
    entry = state.get(1)
    assert entry["status"] == "blocked"


def test_run_splitter_split_inserts_children_in_correct_slot(
    monkeypatch, tmp_path,
):
    """Description (d): children land immediately after the parent in
    `tasks` order, preserving topological order."""
    yaml_path = tmp_path / "tasks.yaml"
    _write_tasks_yaml(yaml_path,
        "version: 2\n"
        "tasks:\n"
        "  - id: 1\n    title: T1\n    description: D\n"
        "  - id: 2\n    title: T2\n    description: D\n"
        "    depends_on: [1]\n"
        "  - id: 3\n    title: T3\n    description: D\n"
        "    depends_on: [2]\n",
    )
    plans_dir = tmp_path / "plans"
    (plans_dir / "task-2").mkdir(parents=True)
    (plans_dir / "task-2" / "plan.md").write_text("(plan)")
    state = _make_state_with_failed_task(plans_dir, 2)
    children = [
        {"id": "2a", "title": "X", "description": "DX", "depends_on": []},
        {"id": "2b", "title": "Y", "description": "DY",
         "depends_on": ["2a"]},
    ]
    _stub_run_claude(monkeypatch, _split_appendix(children))
    task = {"id": 2, "title": "T2", "description": "D"}

    run_splitter(
        task=task, state=state, file_defaults={}, cli={},
        resolved=_resolved(), plans_dir=plans_dir,
        tasks_yaml_path=yaml_path,
    )

    yaml_obj = YAML(typ="rt")
    with open(yaml_path) as f:
        data = yaml_obj.load(f)
    ids = [t["id"] for t in data["tasks"]]
    assert ids == [1, 2, "2a", "2b", 3]


def test_run_splitter_preserves_yaml_comments(
    monkeypatch, tmp_path,
):
    """Description (d): user-authored comments survive a round-trip
    through the splitter (ruamel round-trip mode)."""
    yaml_path = tmp_path / "tasks.yaml"
    _write_tasks_yaml(yaml_path,
        "# Top-level comment that must survive.\n"
        "version: 2\n"
        "tasks:\n"
        "  # comment between tasks\n"
        "  - id: 1\n"
        "    title: T1\n"
        "    description: D\n",
    )
    plans_dir = tmp_path / "plans"
    (plans_dir / "task-1").mkdir(parents=True)
    (plans_dir / "task-1" / "plan.md").write_text("(plan)")
    state = _make_state_with_failed_task(plans_dir, 1)
    children = [
        {"id": "1a", "title": "X", "description": "DX", "depends_on": []},
        {"id": "1b", "title": "Y", "description": "DY",
         "depends_on": ["1a"]},
    ]
    _stub_run_claude(monkeypatch, _split_appendix(children))
    task = {"id": 1, "title": "T1", "description": "D"}

    run_splitter(
        task=task, state=state, file_defaults={}, cli={},
        resolved=_resolved(), plans_dir=plans_dir,
        tasks_yaml_path=yaml_path,
    )

    text = yaml_path.read_text()
    assert "# Top-level comment that must survive." in text


def test_run_splitter_rewires_simple_downstream_depends_on(
    monkeypatch, tmp_path,
):
    """Description (d): a downstream task that depended on the parent
    gets its dep rewired to the LAST child."""
    yaml_path = tmp_path / "tasks.yaml"
    _write_tasks_yaml(yaml_path,
        "version: 2\n"
        "tasks:\n"
        "  - id: 1\n    title: T1\n    description: D\n"
        "  - id: 2\n    title: T2\n    description: D\n"
        "    depends_on: [1]\n",
    )
    plans_dir = tmp_path / "plans"
    (plans_dir / "task-1").mkdir(parents=True)
    (plans_dir / "task-1" / "plan.md").write_text("(plan)")
    state = _make_state_with_failed_task(plans_dir, 1)
    children = [
        {"id": "1a", "title": "X", "description": "DX", "depends_on": []},
        {"id": "1b", "title": "Y", "description": "DY",
         "depends_on": ["1a"]},
    ]
    _stub_run_claude(monkeypatch, _split_appendix(children))
    task = {"id": 1, "title": "T1", "description": "D"}

    run_splitter(
        task=task, state=state, file_defaults={}, cli={},
        resolved=_resolved(), plans_dir=plans_dir,
        tasks_yaml_path=yaml_path,
    )

    yaml_obj = YAML(typ="rt")
    with open(yaml_path) as f:
        data = yaml_obj.load(f)
    by_id = {t["id"]: t for t in data["tasks"]}
    assert list(by_id[2]["depends_on"]) == ["1b"]


def test_run_splitter_compound_downstream_depends_on_only_replaces_parent(
    monkeypatch, tmp_path,
):
    """Description (d) edge case: a downstream task with multi-element
    depends_on `[1, 99]` becomes `[<last_child>, 99]` — only the
    parent reference is rewired."""
    yaml_path = tmp_path / "tasks.yaml"
    _write_tasks_yaml(yaml_path,
        "version: 2\n"
        "tasks:\n"
        "  - id: 1\n    title: T1\n    description: D\n"
        "  - id: 99\n    title: T99\n    description: D\n"
        "  - id: 2\n    title: T2\n    description: D\n"
        "    depends_on: [1, 99]\n",
    )
    plans_dir = tmp_path / "plans"
    (plans_dir / "task-1").mkdir(parents=True)
    (plans_dir / "task-1" / "plan.md").write_text("(plan)")
    state = _make_state_with_failed_task(plans_dir, 1)
    children = [
        {"id": "1a", "title": "X", "description": "DX", "depends_on": []},
        {"id": "1b", "title": "Y", "description": "DY",
         "depends_on": ["1a"]},
    ]
    _stub_run_claude(monkeypatch, _split_appendix(children))
    task = {"id": 1, "title": "T1", "description": "D"}

    run_splitter(
        task=task, state=state, file_defaults={}, cli={},
        resolved=_resolved(), plans_dir=plans_dir,
        tasks_yaml_path=yaml_path,
    )

    yaml_obj = YAML(typ="rt")
    with open(yaml_path) as f:
        data = yaml_obj.load(f)
    by_id = {t["id"]: t for t in data["tasks"]}
    assert list(by_id[2]["depends_on"]) == ["1b", 99]


def test_run_splitter_writes_split_status_with_split_children_and_clears_block_reason(
    monkeypatch, tmp_path,
):
    """Plan §4 pin 8: parent transitions to status='split', records
    `split_children`, and `block_reason` is cleared (set to None)."""
    yaml_path, plans_dir, state = _setup_split_scenario(tmp_path)
    children = [
        {"id": "1a", "title": "X", "description": "DX", "depends_on": []},
        {"id": "1b", "title": "Y", "description": "DY",
         "depends_on": ["1a"]},
    ]
    _stub_run_claude(monkeypatch, _split_appendix(children))
    task = {"id": 1, "title": "T1", "description": "D"}

    # Pre-condition: parent is currently blocked with a reason.
    assert state.get(1)["status"] == "blocked"
    assert state.get(1)["block_reason"] == "max_review_rounds_exhausted"

    run_splitter(
        task=task, state=state, file_defaults={}, cli={},
        resolved=_resolved(), plans_dir=plans_dir,
        tasks_yaml_path=yaml_path,
    )

    entry = state.get(1)
    assert entry["status"] == "split"
    assert entry["split_children"] == ["1a", "1b"]
    assert entry.get("block_reason") is None


def test_run_splitter_one_child_treated_as_keep(
    monkeypatch, tmp_path,
):
    """Plan §4 pin 12: a 1-child split is a no-op masquerading as
    work; treat as keep (no mutation, no status flip)."""
    yaml_path, plans_dir, state = _setup_split_scenario(tmp_path)
    original = yaml_path.read_text()
    children = [
        {"id": "1a", "title": "X", "description": "DX", "depends_on": []},
    ]
    _stub_run_claude(monkeypatch, _split_appendix(children))
    task = {"id": 1, "title": "T1", "description": "D"}

    run_splitter(
        task=task, state=state, file_defaults={}, cli={},
        resolved=_resolved(), plans_dir=plans_dir,
        tasks_yaml_path=yaml_path,
    )

    assert yaml_path.read_text() == original
    entry = state.get(1)
    assert entry["status"] == "blocked"


def test_run_splitter_rewires_upstream_listed_before_parent_depends_on(
    monkeypatch, tmp_path,
):
    """review-1 issue 1 regression: a task listed BEFORE the parent
    in tasks.yaml that has `depends_on: [<parent_id>]` must also have
    its `depends_on` rewired to point at the LAST child.

    `validate_dependencies` is a topological sort that ignores
    list-position, so forward references are legal. The rewire loop
    in `_apply_split_to_yaml` MUST scan all tasks (except parent and
    inserted children), not just tasks at YAML positions after the
    children — otherwise the depender silently waits forever for
    `state.get(parent)["status"] == "done"`, which never happens
    because the parent is `"split"`, not `"done"`.
    """
    yaml_path = tmp_path / "tasks.yaml"
    _write_tasks_yaml(yaml_path,
        "version: 2\n"
        "tasks:\n"
        # A is listed BEFORE its dependency P (a forward reference).
        "  - id: A\n"
        "    title: T-A\n"
        "    description: D\n"
        "    depends_on: [P]\n"
        "  - id: P\n"
        "    title: T-P\n"
        "    description: D\n",
    )
    plans_dir = tmp_path / "plans"
    (plans_dir / "task-P").mkdir(parents=True)
    (plans_dir / "task-P" / "plan.md").write_text("(plan)")
    state = _make_state_with_failed_task(plans_dir, "P")
    children = [
        {"id": "Pa", "title": "X", "description": "DX",
         "depends_on": [], "phase": None},
        {"id": "Pb", "title": "Y", "description": "DY",
         "depends_on": ["Pa"], "phase": None},
    ]
    _stub_run_claude(monkeypatch, _split_appendix(children))
    task = {"id": "P", "title": "T-P", "description": "D"}

    run_splitter(
        task=task, state=state, file_defaults={}, cli={},
        resolved=_resolved(), plans_dir=plans_dir,
        tasks_yaml_path=yaml_path,
    )

    yaml_obj = YAML(typ="rt")
    with open(yaml_path) as f:
        data = yaml_obj.load(f)
    by_id = {t["id"]: t for t in data["tasks"]}
    # The depender that was listed BEFORE the parent must now point
    # at the LAST child id, not at the (now-split) parent.
    assert list(by_id["A"]["depends_on"]) == ["Pb"]


def test_run_splitter_duplicate_within_children_lands_in_yaml_caught_by_next_load(
    monkeypatch, tmp_path,
):
    """review-1 comment (sibling of R6): the splitter performs no
    uniqueness check on the agent-returned children list. A duplicate
    child id in the SAME `children` array writes through to
    tasks.yaml as two entries with the same id. This test pins
    that the load-time validator on the next slate launch catches
    it (rather than silently corrupting state).

    The `with pytest.raises(SystemExit)` clause fires because of
    the new `TasksFile._check_cross_references` task-id uniqueness
    check added in plan §5.3.
    """
    yaml_path = tmp_path / "tasks.yaml"
    _write_tasks_yaml(yaml_path,
        "version: 2\n"
        "tasks:\n"
        "  - id: 1\n    title: T1\n    description: D\n",
    )
    plans_dir = tmp_path / "plans"
    (plans_dir / "task-1").mkdir(parents=True)
    (plans_dir / "task-1" / "plan.md").write_text("(plan)")
    state = _make_state_with_failed_task(plans_dir, 1)
    children = [
        {"id": "1a", "title": "X", "description": "DX",
         "depends_on": [], "phase": None},
        {"id": "1a", "title": "Y", "description": "DY",
         "depends_on": [], "phase": None},
    ]
    _stub_run_claude(monkeypatch, _split_appendix(children))
    task = {"id": 1, "title": "T1", "description": "D"}

    run_splitter(
        task=task, state=state, file_defaults={}, cli={},
        resolved=_resolved(), plans_dir=plans_dir,
        tasks_yaml_path=yaml_path,
    )

    yaml_obj = YAML(typ="rt")
    with open(yaml_path) as f:
        data = yaml_obj.load(f)
    ids = [t["id"] for t in data["tasks"]]
    assert ids.count("1a") == 2
    from claude_plan_execute import loader
    with pytest.raises(SystemExit):
        loader.load_tasks_file(yaml_path)


def test_run_splitter_max_split_depth_guard_blocks_recursion(
    monkeypatch, tmp_path,
):
    """Description (e): a child whose `split_depth` already equals
    `max_split_depth` is NOT re-split."""
    yaml_path, plans_dir, state = _setup_split_scenario(tmp_path)
    captured = _stub_run_claude(monkeypatch, _split_appendix([]))
    task = {
        "id": 1, "title": "T1", "description": "D",
        "split_depth": 1,  # already at max
    }

    run_splitter(
        task=task, state=state, file_defaults={}, cli={},
        resolved=_resolved(max_split_depth=1),
        plans_dir=plans_dir,
        tasks_yaml_path=yaml_path,
    )

    # run_claude must NOT have been called — the depth guard fires
    # before the agent invocation.
    assert captured == []
    entry = state.get(1)
    assert entry["status"] == "blocked"


def test_run_splitter_default_max_split_depth_is_1(
    monkeypatch, tmp_path,
):
    """Plan §5.4: HARDCODED_DEFAULTS['max_split_depth'] == 1; a
    parent at depth 0 IS allowed to split (child becomes depth 1
    and would itself be blocked from re-splitting)."""
    yaml_path, plans_dir, state = _setup_split_scenario(tmp_path)
    children = [
        {"id": "1a", "title": "X", "description": "DX", "depends_on": []},
        {"id": "1b", "title": "Y", "description": "DY",
         "depends_on": ["1a"]},
    ]
    _stub_run_claude(monkeypatch, _split_appendix(children))
    task = {"id": 1, "title": "T1", "description": "D"}
    # No split_depth set → treated as 0; max_split_depth=1 allows split.
    run_splitter(
        task=task, state=state, file_defaults={}, cli={},
        resolved=_resolved(max_split_depth=1),
        plans_dir=plans_dir,
        tasks_yaml_path=yaml_path,
    )
    yaml_obj = YAML(typ="rt")
    with open(yaml_path) as f:
        data = yaml_obj.load(f)
    by_id = {t["id"]: t for t in data["tasks"]}
    assert by_id["1a"]["split_depth"] == 1


def test_run_splitter_already_split_parent_idempotency_skip(
    monkeypatch, tmp_path,
):
    """Description (f) + plan §4 pin 10: a direct call into
    `run_splitter` on a parent whose state.status is already 'split'
    short-circuits before invoking run_claude. Documents the
    direct-callable contract; not a production-path early-out."""
    yaml_path, plans_dir, _ = _setup_split_scenario(tmp_path)
    # Build a state where status is already 'split'.
    state = State(plans_dir / "state.json")
    state.set_status(1, "split", split_children=["1a", "1b"])
    captured = _stub_run_claude(monkeypatch, _split_appendix([]))
    task = {"id": 1, "title": "T1", "description": "D"}

    run_splitter(
        task=task, state=state, file_defaults={}, cli={},
        resolved=_resolved(), plans_dir=plans_dir,
        tasks_yaml_path=yaml_path,
    )

    assert captured == []
    # State unchanged.
    entry = state.get(1)
    assert entry["status"] == "split"
    assert entry["split_children"] == ["1a", "1b"]


def test_run_splitter_concurrent_callers_serialize_via_lock(
    monkeypatch, tmp_path,
):
    """Pin §4.10 + description (g): two concurrent run_splitter calls
    must NOT both apply children. The second arrival sees status=='split'
    (set by the first under lock) and short-circuits.

    NB: this is a UNIT test of the lock primitive, not an integration
    test of a real concurrent flow. The runner topology (`_in_flight`
    + per-task `_attempted` set in `_SlateScheduler`) already
    guarantees one worker per task, so the two-thread scenario this
    test exercises is not reachable through the production runner.
    Keep the test as a regression canary against future changes that
    remove the file lock — if the in-runner one-worker-per-task
    invariant ever loosens, we want this safety net.
    """
    yaml_path = tmp_path / "tasks.yaml"
    _write_tasks_yaml(yaml_path,
        "version: 2\ntasks:\n"
        "  - id: 1\n    title: T1\n    description: D\n",
    )
    plans_dir = tmp_path / "plans"
    (plans_dir / "task-1").mkdir(parents=True)
    (plans_dir / "task-1" / "plan.md").write_text("(plan)")
    state = _make_state_with_failed_task(plans_dir, 1)
    children = [
        {"id": "1a", "title": "X", "description": "DX",
         "depends_on": []},
        {"id": "1b", "title": "Y", "description": "DY",
         "depends_on": ["1a"]},
    ]
    barrier = threading.Barrier(2)

    def stub(*args, **kwargs):
        barrier.wait(timeout=5)
        return 0, _split_appendix(children)

    monkeypatch.setattr(
        "claude_plan_execute.splitter.run_claude", stub,
    )
    task = {"id": 1, "title": "T1", "description": "D"}

    threads = [
        threading.Thread(target=run_splitter, kwargs=dict(
            task=task, state=state, file_defaults={}, cli={},
            resolved=_resolved(), plans_dir=plans_dir,
            tasks_yaml_path=yaml_path,
        ))
        for _ in range(2)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    yaml_obj = YAML(typ="rt")
    with open(yaml_path) as f:
        data = yaml_obj.load(f)
    ids = [t["id"] for t in data["tasks"]]
    # Children appear EXACTLY ONCE — idempotency wins, not parallel
    # application.
    assert ids == [1, "1a", "1b"]


def test_run_splitter_returns_silently_when_tasks_yaml_path_is_None(
    monkeypatch, tmp_path,
):
    """Plan §6.2 step 2: `tasks_yaml_path=None` (e.g. some test
    callers / programmatic invocations) short-circuits before
    invoking the agent."""
    yaml_path, plans_dir, state = _setup_split_scenario(tmp_path)
    captured = _stub_run_claude(monkeypatch, _split_appendix([]))
    task = {"id": 1, "title": "T1", "description": "D"}

    run_splitter(
        task=task, state=state, file_defaults={}, cli={},
        resolved=_resolved(), plans_dir=plans_dir,
        tasks_yaml_path=None,
    )

    assert captured == []
    entry = state.get(1)
    assert entry["status"] == "blocked"


def test_run_splitter_run_claude_phase_label_is_splitter(
    monkeypatch, tmp_path,
):
    """Plan §4 pin 11: run_claude is called with phase='splitter'
    so the JSONL row gets the new free-form label."""
    yaml_path, plans_dir, state = _setup_split_scenario(tmp_path)
    children = [
        {"id": "1a", "title": "X", "description": "DX", "depends_on": []},
        {"id": "1b", "title": "Y", "description": "DY",
         "depends_on": ["1a"]},
    ]
    captured = _stub_run_claude(monkeypatch, _split_appendix(children))
    task = {"id": 1, "title": "T1", "description": "D"}

    run_splitter(
        task=task, state=state, file_defaults={}, cli={},
        resolved=_resolved(), plans_dir=plans_dir,
        tasks_yaml_path=yaml_path,
    )

    assert len(captured) == 1
    kwargs = captured[0][1]
    assert kwargs["phase"] == "splitter"
    assert kwargs["model"] == "sonnet"
    assert kwargs["effort"] == "medium"


def test_run_splitter_subprocess_exception_logged_and_no_raise(
    monkeypatch, tmp_path,
):
    """Plan §6.2 step 1 catch-all: an unexpected exception inside
    run_splitter logs and returns; the parent stays blocked, no
    re-raise to the caller."""
    yaml_path, plans_dir, state = _setup_split_scenario(tmp_path)

    def boom(*args, **kwargs):
        raise RuntimeError("explosive subprocess")

    monkeypatch.setattr(
        "claude_plan_execute.splitter.run_claude", boom,
    )
    task = {"id": 1, "title": "T1", "description": "D"}

    # Must not raise.
    run_splitter(
        task=task, state=state, file_defaults={}, cli={},
        resolved=_resolved(), plans_dir=plans_dir,
        tasks_yaml_path=yaml_path,
    )

    entry = state.get(1)
    assert entry["status"] == "blocked"


