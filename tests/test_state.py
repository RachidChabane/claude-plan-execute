"""Tests for claude_plan_execute.state.State."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from claude_plan_execute.state import (
    FAILURE_CLASSES,
    SCHEMA_VERSION,
    State,
    STATUSES,
    make_failure_record,
)


def test_new_state_has_empty_tasks(tmp_path: Path):
    s = State(tmp_path / "state.json")
    assert s.data == {"schema_version": 3, "tasks": {}}


def test_get_creates_pending_entry(tmp_path: Path):
    s = State(tmp_path / "state.json")
    entry = s.get(1)
    assert entry == {"status": "pending", "attempts": [], "current_attempt": 1}
    # Returned reference is the same object stored in s.data
    assert s.data["tasks"]["1"] is entry


def test_get_accepts_int_or_str_task_id(tmp_path: Path):
    s = State(tmp_path / "state.json")
    s.get(1)
    assert s.get("1") is s.data["tasks"]["1"]


def test_set_status_persists_and_merges_extras(tmp_path: Path):
    path = tmp_path / "state.json"
    s = State(path)
    s.set_status(1, "planning", plan_path="plans/task-1/plan.md")
    s.set_status(1, "reviewing", review_round=2)

    entry = s.get(1)
    assert entry["status"] == "reviewing"
    assert entry["review_round"] == 2
    # plan_path from earlier set_status must be preserved (update merges).
    assert entry["plan_path"] == "plans/task-1/plan.md"


def test_save_writes_atomically(tmp_path: Path):
    path = tmp_path / "state.json"
    s = State(path)
    s.set_status(1, "done")

    on_disk = json.loads(path.read_text())
    assert on_disk == {
        "schema_version": 3,
        "tasks": {
            "1": {"status": "done", "attempts": [], "current_attempt": 1},
        },
    }

    # No leftover .tmp file
    assert list(tmp_path.glob("*.tmp")) == []


def test_save_creates_parent_directory(tmp_path: Path):
    nested = tmp_path / "plans" / "deep"
    s = State(nested / "state.json")
    s.set_status(1, "pending")
    assert (nested / "state.json").exists()


def test_load_existing_state(tmp_path: Path):
    """Schema-1 file: load injects defaults for attempts and current_attempt."""
    path = tmp_path / "state.json"
    path.write_text(json.dumps({
        "tasks": {"1": {"status": "done", "plan_path": "x"}, "2": {"status": "pending"}}
    }))
    s = State(path)
    assert s.get(1) == {
        "status": "done", "plan_path": "x",
        "attempts": [], "current_attempt": 1,
    }
    assert s.get(2) == {
        "status": "pending",
        "attempts": [], "current_attempt": 1,
    }


def test_reset_single_task_clears_all_keys(tmp_path: Path):
    path = tmp_path / "state.json"
    s = State(path)
    s.set_status(1, "implementing", plan_path="x", review_round=3)
    s.set_status(2, "done")

    s.reset(1)
    assert s.get(1) == {
        "status": "pending", "attempts": [], "current_attempt": 1,
    }
    assert s.get(2) == {
        "status": "done", "attempts": [], "current_attempt": 1,
    }

    # Persisted to disk
    on_disk = json.loads(path.read_text())
    assert on_disk["tasks"]["1"] == {
        "status": "pending", "attempts": [], "current_attempt": 1,
    }
    assert on_disk["tasks"]["2"] == {
        "status": "done", "attempts": [], "current_attempt": 1,
    }


def test_reset_all_tasks(tmp_path: Path):
    path = tmp_path / "state.json"
    s = State(path)
    s.set_status(1, "done")
    s.set_status(2, "implementing", plan_path="y")

    s.reset()
    assert s.get(1) == {
        "status": "pending", "attempts": [], "current_attempt": 1,
    }
    assert s.get(2) == {
        "status": "pending", "attempts": [], "current_attempt": 1,
    }


def test_reset_all_on_empty_state_is_noop(tmp_path: Path):
    s = State(tmp_path / "state.json")
    s.reset()  # no tasks; should not raise
    assert s.data == {"schema_version": 3, "tasks": {}}


def test_statuses_constant_includes_known_values():
    # Pin: STATUSES documents the known status names. Task 6 added "blocked"
    # for the on_max_review_rounds: fail path. T10 (v0.3 §2.4) added
    # "awaiting_human" for human-gate pause.
    assert "pending" in STATUSES
    assert "done" in STATUSES
    assert "implementing" in STATUSES
    assert "blocked" in STATUSES
    assert "awaiting_human" in STATUSES


def test_use_color_propagates_to_loader_and_deps_via_module_global(monkeypatch, capsys):
    """colors.set_use_color toggles a single module-level global; downstream
    modules (loader, deps) honor it because they import red/yellow as references
    that read colors._use_color at call time."""
    from claude_plan_execute import colors
    from claude_plan_execute.colors import red

    # Force isatty=True so the color path is reachable.
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)

    colors.set_use_color(True)
    assert "\033[" in red("hello")

    colors.set_use_color(False)
    assert red("hello") == "hello"

    # Restore default so other tests aren't affected.
    colors.set_use_color(True)


# ---------------------------------------------------------------------------
# Schema version + FailureRecord (task 9)
# ---------------------------------------------------------------------------


def test_schema_version_constant_is_3():
    """Pin: SCHEMA_VERSION is the load-bearing constant. Bumped to 3 in
    T2 v0.3 (§3.2) when per-task `usage_summary` was added."""
    assert SCHEMA_VERSION == 3


def test_new_state_has_schema_version_3(tmp_path: Path):
    s = State(tmp_path / "state.json")
    assert s.data["schema_version"] == 3


def test_save_writes_schema_version_to_disk(tmp_path: Path):
    path = tmp_path / "state.json"
    s = State(path)
    s.set_status(1, "pending")
    on_disk = json.loads(path.read_text())
    assert on_disk["schema_version"] == 3


def test_load_v1_state_self_heals_to_v3(tmp_path: Path):
    """Schema-1 file (no schema_version key) loads with the v3
    defaults injected and saves as schema 3 on next write."""
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"tasks": {"1": {"status": "done"}}}))
    s = State(path)
    assert s.data["schema_version"] == 3

    s.save()
    on_disk = json.loads(path.read_text())
    assert on_disk["schema_version"] == 3


def test_load_v2_state_self_heals_to_v3(tmp_path: Path):
    """Schema-2 file with attempts loads cleanly into a schema-3
    State; schema_version stamp is updated, attempts preserved."""
    fr = {
        "failure_class": "manual_stop", "phase": "implement",
        "timestamp": "2026-05-02T22:30:00+00:00",
        "reason": "x", "verbatim": "", "artifacts": {}, "hypothesis": None,
    }
    path = tmp_path / "state.json"
    path.write_text(json.dumps({
        "schema_version": 2,
        "tasks": {"1": {"status": "blocked", "attempts": [fr], "current_attempt": 2}},
    }))
    s = State(path)
    assert s.data["schema_version"] == 3
    assert s.get(1)["attempts"] == [fr]
    assert s.get(1)["current_attempt"] == 2

    s.save()
    on_disk = json.loads(path.read_text())
    assert on_disk["schema_version"] == 3


def test_load_v1_state_injects_attempts_default(tmp_path: Path):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({
        "tasks": {
            "1": {"status": "done"},
            "2": {"status": "pending", "review_round": 2},
        }
    }))
    s = State(path)
    assert s.get(1)["attempts"] == []
    assert s.get(2)["attempts"] == []


def test_load_v1_state_injects_current_attempt_default(tmp_path: Path):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"tasks": {"1": {"status": "done"}}}))
    s = State(path)
    assert s.get(1)["current_attempt"] == 1


def test_load_v2_state_preserves_attempts(tmp_path: Path):
    """Schema-2 file with non-empty attempts loads cleanly into v3
    (self-heal stamps schema_version=3); attempts preserved."""
    fr = {
        "failure_class": "manual_stop", "phase": "implement",
        "timestamp": "2026-05-02T22:30:00+00:00",
        "reason": "x", "verbatim": "", "artifacts": {}, "hypothesis": None,
    }
    path = tmp_path / "state.json"
    path.write_text(json.dumps({
        "schema_version": 2,
        "tasks": {"1": {"status": "blocked", "attempts": [fr], "current_attempt": 1}},
    }))
    s = State(path)
    assert s.data["schema_version"] == 3
    assert s.get(1)["attempts"] == [fr]
    assert s.get(1)["current_attempt"] == 1


def test_load_v2_state_does_not_overwrite_current_attempt(tmp_path: Path):
    """If a schema-2 file has current_attempt: 3, the load must not
    reset it to 1 (the v2 → v3 self-heal preserves it)."""
    path = tmp_path / "state.json"
    path.write_text(json.dumps({
        "schema_version": 2,
        "tasks": {"1": {"status": "blocked", "attempts": [], "current_attempt": 3}},
    }))
    s = State(path)
    assert s.get(1)["current_attempt"] == 3
    assert s.data["schema_version"] == 3


def test_get_creates_entry_with_attempts_default(tmp_path: Path):
    s = State(tmp_path / "state.json")
    assert s.get(1)["attempts"] == []


def test_get_creates_entry_with_current_attempt_default(tmp_path: Path):
    s = State(tmp_path / "state.json")
    assert s.get(1)["current_attempt"] == 1


def test_record_failure_appends_to_attempts(tmp_path: Path):
    s = State(tmp_path / "state.json")
    fr = make_failure_record("manual_stop", "implement", reason="x")
    s.record_failure(1, fr)
    assert s.get(1)["attempts"] == [fr]


def test_record_failure_appends_multiple(tmp_path: Path):
    s = State(tmp_path / "state.json")
    s.record_failure(1, make_failure_record("manual_stop", "implement", reason="a"))
    s.record_failure(1, make_failure_record("manual_stop", "implement", reason="b"))
    assert len(s.get(1)["attempts"]) == 2
    assert s.get(1)["attempts"][0]["reason"] == "a"
    assert s.get(1)["attempts"][1]["reason"] == "b"


def test_record_failure_persists_to_disk(tmp_path: Path):
    path = tmp_path / "state.json"
    s = State(path)
    s.record_failure(1, make_failure_record("manual_stop", "plan", reason="r"))
    on_disk = json.loads(path.read_text())
    assert len(on_disk["tasks"]["1"]["attempts"]) == 1
    assert on_disk["tasks"]["1"]["attempts"][0]["reason"] == "r"


def test_record_failure_does_not_change_status(tmp_path: Path):
    s = State(tmp_path / "state.json")
    s.set_status(1, "implementing")
    s.record_failure(1, make_failure_record("manual_stop", "implement", reason="x"))
    assert s.get(1)["status"] == "implementing"


def test_record_failure_does_not_change_current_attempt(tmp_path: Path):
    """Pin §D5: record_failure does NOT touch current_attempt.
    That bump is task 10's resume-prep responsibility."""
    s = State(tmp_path / "state.json")
    s.record_failure(1, make_failure_record("manual_stop", "plan", reason="x"))
    assert s.get(1)["current_attempt"] == 1
    s.record_failure(1, make_failure_record("manual_stop", "plan", reason="y"))
    assert s.get(1)["current_attempt"] == 1


def test_record_failure_preserves_existing_extras(tmp_path: Path):
    s = State(tmp_path / "state.json")
    s.set_status(1, "blocked", block_reason="implementer_timeout", plan_path="p.md")
    s.record_failure(1, make_failure_record("implementer_timeout", "implement", reason="x"))
    entry = s.get(1)
    assert entry["block_reason"] == "implementer_timeout"
    assert entry["plan_path"] == "p.md"


def test_reset_single_task_clears_attempts_and_current_attempt(tmp_path: Path):
    s = State(tmp_path / "state.json")
    s.record_failure(1, make_failure_record("manual_stop", "plan", reason="x"))
    s.get(1)["current_attempt"] = 5
    s.save()
    s.reset(1)
    assert s.get(1)["attempts"] == []
    assert s.get(1)["current_attempt"] == 1


def test_reset_all_clears_attempts_and_current_attempt(tmp_path: Path):
    s = State(tmp_path / "state.json")
    s.record_failure(1, make_failure_record("manual_stop", "plan", reason="x"))
    s.record_failure(2, make_failure_record("manual_stop", "plan", reason="y"))
    s.reset()
    assert s.get(1)["attempts"] == []
    assert s.get(2)["attempts"] == []
    assert s.get(1)["current_attempt"] == 1
    assert s.get(2)["current_attempt"] == 1


# --- make_failure_record (pure constructor) --------------------------------


def test_make_failure_record_required_keys_present():
    fr = make_failure_record("manual_stop", "plan", reason="r")
    expected_keys = {
        "failure_class", "phase", "timestamp", "reason",
        "verbatim", "artifacts", "hypothesis",
    }
    assert set(fr.keys()) == expected_keys


def test_make_failure_record_failure_class_passes_through():
    fr = make_failure_record("review_rejected_after_max_rounds", "review", reason="r")
    assert fr["failure_class"] == "review_rejected_after_max_rounds"


def test_make_failure_record_phase_passes_through():
    fr = make_failure_record("manual_stop", "implement", reason="r")
    assert fr["phase"] == "implement"


def test_make_failure_record_reason_passes_through():
    fr = make_failure_record("manual_stop", "plan", reason="cause text")
    assert fr["reason"] == "cause text"


def test_make_failure_record_default_verbatim_is_empty_string():
    fr = make_failure_record("manual_stop", "plan", reason="r")
    assert fr["verbatim"] == ""


def test_make_failure_record_default_artifacts_is_empty_dict():
    fr = make_failure_record("manual_stop", "plan", reason="r")
    assert fr["artifacts"] == {}
    assert isinstance(fr["artifacts"], dict)


def test_make_failure_record_default_hypothesis_is_None():
    fr = make_failure_record("manual_stop", "plan", reason="r")
    assert fr["hypothesis"] is None


def test_make_failure_record_artifacts_passes_through():
    fr = make_failure_record(
        "implementer_timeout", "implement",
        reason="r", artifacts={"plan": "p.md", "log": "l.txt"},
    )
    assert fr["artifacts"] == {"plan": "p.md", "log": "l.txt"}


def test_make_failure_record_hypothesis_passes_through():
    fr = make_failure_record(
        "manual_stop", "plan", reason="r", hypothesis="my guess",
    )
    assert fr["hypothesis"] == "my guess"


def test_make_failure_record_artifacts_is_copied_not_aliased():
    """Mutating the input dict after construction does not mutate the record."""
    inp = {"plan": "p.md"}
    fr = make_failure_record(
        "manual_stop", "plan", reason="r", artifacts=inp,
    )
    inp["other"] = "x"
    assert fr["artifacts"] == {"plan": "p.md"}


def test_make_failure_record_timestamp_is_iso_format():
    fr = make_failure_record("manual_stop", "plan", reason="r")
    # Round-trip via fromisoformat to confirm parseability.
    parsed = datetime.fromisoformat(fr["timestamp"])
    assert parsed.tzinfo is not None  # UTC offset present


def test_make_failure_record_timestamp_is_utc():
    fr = make_failure_record("manual_stop", "plan", reason="r")
    parsed = datetime.fromisoformat(fr["timestamp"])
    assert parsed.utcoffset().total_seconds() == 0


def test_make_failure_record_timestamp_recent():
    """Sanity: the timestamp is approximately 'now'."""
    before = datetime.now(timezone.utc)
    fr = make_failure_record("manual_stop", "plan", reason="r")
    after = datetime.now(timezone.utc)
    parsed = datetime.fromisoformat(fr["timestamp"])
    assert before <= parsed <= after


def test_make_failure_record_short_verbatim_passes_through():
    text = "short content"
    fr = make_failure_record("manual_stop", "plan", reason="r", verbatim=text)
    assert fr["verbatim"] == text


def test_make_failure_record_verbatim_at_4096_bytes_unchanged():
    """Boundary: cap is `> 4096`; exactly 4096 passes through unchanged."""
    text = "A" * 4096
    fr = make_failure_record("manual_stop", "plan", reason="r", verbatim=text)
    assert fr["verbatim"] == text
    assert "[truncated:" not in fr["verbatim"]


def test_make_failure_record_truncates_verbatim_over_4kb():
    text = "A" * 5000
    fr = make_failure_record("manual_stop", "plan", reason="r", verbatim=text)
    # Content portion is ≤ 4096 bytes (after marker stripped).
    assert fr["verbatim"].startswith("A" * 4096)
    assert "[truncated:" in fr["verbatim"]


def test_make_failure_record_truncation_marker_reports_omitted_bytes():
    text = "A" * 5000
    fr = make_failure_record("manual_stop", "plan", reason="r", verbatim=text)
    # 5000 - 4096 = 904 bytes omitted.
    assert "[truncated: 904 bytes omitted]" in fr["verbatim"]


def test_make_failure_record_verbatim_total_bounded_by_4kb_plus_marker():
    """Pin §D8: total string ≤ 4 KB content + marker overhead.
    Marker length is bounded — `[truncated: <int> bytes omitted]` is
    < 50 bytes for any reasonable input."""
    text = "A" * 100_000
    fr = make_failure_record("manual_stop", "plan", reason="r", verbatim=text)
    encoded_len = len(fr["verbatim"].encode("utf-8"))
    assert encoded_len <= 4096 + 50


def test_make_failure_record_handles_multibyte_utf8_safely():
    """A multi-byte UTF-8 sequence at the cap boundary must not raise."""
    # "أ" (Arabic letter alif) is 2 bytes in UTF-8. Build a string
    # that places one across the 4096-byte boundary.
    text = "A" * 4095 + "أ" * 100  # bytes: 4095 + (100 * 2) = 4295
    fr = make_failure_record("manual_stop", "plan", reason="r", verbatim=text)
    # Must not raise UnicodeDecodeError. Truncated content ≤ 4096 bytes.
    content_only = fr["verbatim"].split("\n[truncated:", 1)[0]
    assert len(content_only.encode("utf-8")) <= 4096


# --- FAILURE_CLASSES constant ----------------------------------------------


def test_failure_classes_includes_task_9_writes():
    """The four failure classes task 9 explicitly writes."""
    assert "review_rejected_after_max_rounds" in FAILURE_CLASSES
    assert "implementer_timeout" in FAILURE_CLASSES
    assert "implementer_max_turns" in FAILURE_CLASSES
    assert "manual_stop" in FAILURE_CLASSES


def test_failure_classes_includes_deferred_classes():
    """Forward-compat: classes deferred to task 12 / future."""
    assert "gate_failed" in FAILURE_CLASSES
    assert "plan_not_written" in FAILURE_CLASSES
    assert "test_failed" in FAILURE_CLASSES
    assert "budget_exceeded" in FAILURE_CLASSES


def test_failure_classes_includes_v0_3_writes():
    """v0.3 task 1 — Claude-API rate-limit retries exhausted."""
    assert "rate_limited" in FAILURE_CLASSES


def test_failure_classes_includes_v0_3_task_8_writes():
    """v0.3 task 8 — three service-related failure classes."""
    assert "service_unavailable" in FAILURE_CLASSES
    assert "service_crashed" in FAILURE_CLASSES
    assert "service_unstable" in FAILURE_CLASSES


def test_failure_classes_includes_v0_3_task_10_writes():
    """v0.3 task 10 — human gate timeout class (§2.4)."""
    assert "human_gate_timeout" in FAILURE_CLASSES


def test_make_failure_record_does_not_validate_failure_class():
    """Pin §D3: unknown failure_class strings are accepted (no raise)."""
    fr = make_failure_record(
        "completely_made_up_class", "plan", reason="r",
    )
    assert fr["failure_class"] == "completely_made_up_class"


# ---------------------------------------------------------------------------
# summarize_usage helper (task 2 v0.3)
# ---------------------------------------------------------------------------

import json as _json

from claude_plan_execute.state import summarize_usage


def test_summarize_usage_returns_zero_rollup_for_missing_file(tmp_path):
    p = tmp_path / "nope.jsonl"
    s = summarize_usage(p)
    assert s == {
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_cache_read": 0,
        "total_cache_create": 0,
        "total_runs": 0,
        "duration_seconds": 0.0,
    }


def test_summarize_usage_returns_zero_rollup_for_empty_file(tmp_path):
    p = tmp_path / "u.jsonl"
    p.write_text("")
    s = summarize_usage(p)
    assert s["total_runs"] == 0


def test_summarize_usage_sums_single_row(tmp_path):
    p = tmp_path / "u.jsonl"
    p.write_text(_json.dumps({
        "phase": "plan", "input": 100, "output": 50,
        "cache_read": 30, "cache_create": 5,
        "duration_seconds": 12.0,
    }) + "\n")
    s = summarize_usage(p)
    assert s["total_input_tokens"] == 100
    assert s["total_output_tokens"] == 50
    assert s["total_cache_read"] == 30
    assert s["total_cache_create"] == 5
    assert s["duration_seconds"] == 12.0
    assert s["total_runs"] == 1


def test_summarize_usage_sums_multiple_rows(tmp_path):
    p = tmp_path / "u.jsonl"
    rows = [
        {"phase": "plan", "input": 1, "output": 2,
         "cache_read": 3, "cache_create": 4,
         "duration_seconds": 5.0},
        {"phase": "implement", "input": 10, "output": 20,
         "cache_read": 30, "cache_create": 40,
         "duration_seconds": 50.0},
        {"phase": "gate:tests", "input": 0, "output": 0,
         "cache_read": 0, "cache_create": 0,
         "duration_seconds": 1.5},
    ]
    p.write_text("".join(_json.dumps(r) + "\n" for r in rows))
    s = summarize_usage(p)
    assert s["total_input_tokens"] == 11
    assert s["total_output_tokens"] == 22
    assert s["total_cache_read"] == 33
    assert s["total_cache_create"] == 44
    assert s["duration_seconds"] == 56.5
    assert s["total_runs"] == 3


def test_summarize_usage_skips_corrupt_lines(tmp_path):
    p = tmp_path / "u.jsonl"
    p.write_text(
        _json.dumps({"input": 1, "output": 1, "cache_read": 0,
                     "cache_create": 0, "duration_seconds": 0.0}) + "\n"
        + "not json\n"
        + "{incomplete\n"
        + _json.dumps({"input": 2, "output": 2, "cache_read": 0,
                       "cache_create": 0, "duration_seconds": 0.0}) + "\n"
    )
    s = summarize_usage(p)
    assert s["total_runs"] == 2
    assert s["total_input_tokens"] == 3


def test_summarize_usage_handles_missing_token_fields(tmp_path):
    """Pin §D9: rows missing token keys default to 0; row still counts."""
    p = tmp_path / "u.jsonl"
    p.write_text(_json.dumps({
        "phase": "plan", "duration_seconds": 5.0,
    }) + "\n")
    s = summarize_usage(p)
    assert s["total_input_tokens"] == 0
    assert s["total_runs"] == 1
    assert s["duration_seconds"] == 5.0


def test_summarize_usage_skips_non_dict_rows(tmp_path):
    p = tmp_path / "u.jsonl"
    p.write_text("[1, 2, 3]\n42\n\"a string\"\n")
    s = summarize_usage(p)
    assert s["total_runs"] == 0


def test_summarize_usage_skips_blank_lines(tmp_path):
    p = tmp_path / "u.jsonl"
    p.write_text(
        _json.dumps({"input": 1, "output": 0, "cache_read": 0,
                     "cache_create": 0, "duration_seconds": 0.0}) + "\n"
        + "\n"
        + "   \n"
        + _json.dumps({"input": 2, "output": 0, "cache_read": 0,
                       "cache_create": 0, "duration_seconds": 0.0}) + "\n"
    )
    s = summarize_usage(p)
    assert s["total_runs"] == 2


def test_summarize_usage_returns_correct_types(tmp_path):
    """Pin §D9: token totals are int; duration_seconds is float."""
    p = tmp_path / "u.jsonl"
    p.write_text(_json.dumps({
        "input": 1, "output": 2, "cache_read": 3, "cache_create": 4,
        "duration_seconds": 5.5,
    }) + "\n")
    s = summarize_usage(p)
    assert isinstance(s["total_input_tokens"], int)
    assert isinstance(s["total_output_tokens"], int)
    assert isinstance(s["total_cache_read"], int)
    assert isinstance(s["total_cache_create"], int)
    assert isinstance(s["total_runs"], int)
    assert isinstance(s["duration_seconds"], float)


# --- Thread-safety (T13 v0.3 §2.5) ----------------------------------------

import threading


def test_state_concurrent_set_status_no_lost_writes(tmp_path):
    """T13 §F2.3: 16 threads each call set_status N times; final
    state.json reflects ALL writes (no clobbered entries)."""
    s = State(tmp_path / "state.json")
    NUM_THREADS = 16
    NUM_WRITES = 50

    def worker(worker_id: int):
        for i in range(NUM_WRITES):
            s.set_status(
                f"task_{worker_id}_{i}", "implementing",
                round_num=i,
            )

    threads = [
        threading.Thread(target=worker, args=(w,))
        for w in range(NUM_THREADS)
    ]
    for t in threads: t.start()
    for t in threads: t.join()

    # On disk: every write landed (no clobber).
    on_disk = json.loads((tmp_path / "state.json").read_text())
    expected_keys = {
        f"task_{w}_{i}"
        for w in range(NUM_THREADS) for i in range(NUM_WRITES)
    }
    assert set(on_disk["tasks"].keys()) == expected_keys


def test_state_concurrent_record_failure_no_lost_attempts(tmp_path):
    """T13 §F2.3: record_failure under contention preserves all
    appends (the get-then-modify-then-save compound is atomic)."""
    s = State(tmp_path / "state.json")
    NUM_THREADS = 8
    NUM_FAILURES_PER = 25

    def worker(worker_id: int):
        for i in range(NUM_FAILURES_PER):
            s.record_failure(
                "shared_task",
                make_failure_record(
                    "manual_stop", "implement",
                    reason=f"w{worker_id}_i{i}",
                ),
            )

    threads = [
        threading.Thread(target=worker, args=(w,))
        for w in range(NUM_THREADS)
    ]
    for t in threads: t.start()
    for t in threads: t.join()

    attempts = s.get("shared_task")["attempts"]
    assert len(attempts) == NUM_THREADS * NUM_FAILURES_PER


def test_state_concurrent_save_atomic_no_partial_write(tmp_path):
    """T13 §F2.3: under heavy concurrent save(), the on-disk file is
    always parseable JSON (atomic-rename guarantee preserved)."""
    s = State(tmp_path / "state.json")

    def worker(worker_id: int):
        for i in range(50):
            s.set_status(f"t{worker_id}", "implementing", n=i)

    threads = [
        threading.Thread(target=worker, args=(w,))
        for w in range(8)
    ]
    for t in threads: t.start()
    for t in threads: t.join()

    on_disk = json.loads((tmp_path / "state.json").read_text())
    assert "tasks" in on_disk


def test_failure_classes_includes_worker_exception():
    """T13 §F2.5: worker_exception added to FAILURE_CLASSES."""
    assert "worker_exception" in FAILURE_CLASSES
