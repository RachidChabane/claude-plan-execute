"""Tests for persona-fragment loading and prompt section composition (task 7).

Covers:
- claude_plan_execute.loader.load_persona_fragment (file I/O, line cap)
- claude_plan_execute.prompts.build_persona_section (pure composition)
- persona kwarg on each of the four prompt builders (wiring)
"""
from __future__ import annotations

from pathlib import Path

import pytest

from claude_plan_execute.loader import load_persona_fragment
from claude_plan_execute.prompts import (
    build_implement_prompt,
    build_persona_section,
    build_plan_prompt,
    build_review_prompt,
    build_revise_prompt,
)


# ---------------------------------------------------------------------------
# A. load_persona_fragment
# ---------------------------------------------------------------------------


def test_load_persona_returns_empty_string_for_None():
    assert load_persona_fragment(None) == ""


def test_load_persona_returns_empty_string_for_empty_string_path():
    assert load_persona_fragment("") == ""


def test_load_persona_returns_empty_string_for_missing_file(tmp_path: Path):
    """Missing file is silent — common case before a project authors a persona."""
    assert load_persona_fragment(tmp_path / "missing.md") == ""


def test_load_persona_returns_empty_string_for_directory(tmp_path: Path):
    """Pointing persona_file at a directory is silent (Plan §D4)."""
    d = tmp_path / "personadir"
    d.mkdir()
    assert load_persona_fragment(d) == ""


def test_load_persona_reads_file_content_when_present(tmp_path: Path):
    p = tmp_path / "p.md"
    p.write_text("# Persona\nA short fragment.")
    assert load_persona_fragment(p) == "# Persona\nA short fragment."


def test_load_persona_accepts_string_path(tmp_path: Path):
    p = tmp_path / "p.md"
    p.write_text("hi")
    assert load_persona_fragment(str(p)) == "hi"


def test_load_persona_under_800_lines_returns_content(tmp_path: Path):
    p = tmp_path / "p.md"
    p.write_text("\n".join(f"line {i}" for i in range(500)))  # 500 lines
    out = load_persona_fragment(p)
    assert "line 0" in out
    assert "line 499" in out


def test_load_persona_at_exactly_800_lines_returns_content(tmp_path: Path):
    """Boundary: 800 lines is allowed (cap is '> 800')."""
    p = tmp_path / "p.md"
    p.write_text("\n".join(f"line {i}" for i in range(800)))  # exactly 800
    assert load_persona_fragment(p) != ""


def test_load_persona_over_800_lines_returns_empty(tmp_path: Path):
    p = tmp_path / "p.md"
    p.write_text("\n".join(f"line {i}" for i in range(801)))  # 801 lines
    assert load_persona_fragment(p) == ""


def test_load_persona_over_800_lines_emits_warning(tmp_path: Path, capsys):
    p = tmp_path / "p.md"
    p.write_text("\n".join(f"line {i}" for i in range(900)))
    load_persona_fragment(p)
    captured = capsys.readouterr().out
    assert "Warning" in captured
    assert "800" in captured                            # mentions the cap
    assert "900" in captured                            # mentions the actual count
    assert str(p) in captured                           # mentions file path
    assert "persona_extra" in captured                  # actionable hint


def test_load_persona_over_cap_warning_avoids_substring_deprecated(
    tmp_path: Path, capsys,
):
    """The warning must not collide with any 'deprecated' assertion that
    other warning tests rely on (cf. task 6 §D4 for the precedent)."""
    p = tmp_path / "p.md"
    p.write_text("\n".join(f"line {i}" for i in range(900)))
    load_persona_fragment(p)
    captured = capsys.readouterr().out
    assert "deprecated" not in captured


def test_load_persona_uses_utf8(tmp_path: Path):
    """Non-ASCII personas (Arabic, etc.) must round-trip cleanly."""
    p = tmp_path / "p.md"
    text = "# الشخصية\nبعض النص."
    p.write_text(text, encoding="utf-8")
    assert load_persona_fragment(p) == text


def test_load_persona_no_warning_for_under_cap(tmp_path: Path, capsys):
    p = tmp_path / "p.md"
    p.write_text("\n".join(["line"] * 50))
    load_persona_fragment(p)
    captured = capsys.readouterr().out
    assert "Warning" not in captured


def test_load_persona_no_warning_for_missing_file(tmp_path: Path, capsys):
    load_persona_fragment(tmp_path / "missing.md")
    captured = capsys.readouterr().out
    assert "Warning" not in captured


# ---------------------------------------------------------------------------
# B. build_persona_section
# ---------------------------------------------------------------------------


def test_build_persona_section_empty_inputs_returns_empty():
    assert build_persona_section("", "") == ""


def test_build_persona_section_whitespace_only_inputs_returns_empty():
    assert build_persona_section("  \n  ", "\t\n") == ""


def test_build_persona_section_global_only_emits_header():
    out = build_persona_section("Global text", "")
    assert "## Project conventions" in out
    assert "Global text" in out


def test_build_persona_section_extra_only_emits_header():
    out = build_persona_section("", "Extra text")
    assert "## Project conventions" in out
    assert "Extra text" in out


def test_build_persona_section_both_concatenates_global_first():
    out = build_persona_section("GLOBAL_MARKER", "EXTRA_MARKER")
    g_idx = out.index("GLOBAL_MARKER")
    e_idx = out.index("EXTRA_MARKER")
    assert g_idx < e_idx, "Global persona must precede per-phase persona_extra"


def test_build_persona_section_starts_with_header_line():
    out = build_persona_section("body", "")
    assert out.startswith("## Project conventions")


def test_build_persona_section_ends_with_double_newline_when_present():
    """Caller does `persona_section + body`; the trailing \\n\\n separates."""
    out = build_persona_section("body", "")
    assert out.endswith("\n\n")


def test_build_persona_section_strips_trailing_whitespace_in_inputs():
    out = build_persona_section("body\n\n\n", "extra\n\n\n")
    # The composer normalizes trailing whitespace so the block doesn't
    # accumulate runs of blank lines.
    assert "body\n\n\n" not in out
    assert "extra\n\n\n" not in out
    assert "body" in out
    assert "extra" in out


# ---------------------------------------------------------------------------
# C. Prompt builders accept and prepend persona
# ---------------------------------------------------------------------------


def test_plan_prompt_default_persona_is_empty():
    out = build_plan_prompt({"id": 1, "title": "T", "description": "D"}, "p.md")
    assert "## Project conventions" not in out


def test_plan_prompt_prepends_persona_when_supplied():
    persona = "## Project conventions\n\nDo X.\n\n"
    out = build_plan_prompt(
        {"id": 1, "title": "T", "description": "D"}, "p.md",
        persona=persona,
    )
    assert out.startswith("## Project conventions")
    assert "Do X." in out
    # Original body still present
    assert "TASK 1: T" in out


def test_plan_prompt_persona_appears_before_body():
    persona = "## Project conventions\n\nMARKER_PERSONA\n\n"
    out = build_plan_prompt(
        {"id": 1, "title": "T", "description": "D"}, "p.md",
        persona=persona,
    )
    persona_idx = out.index("MARKER_PERSONA")
    body_idx = out.index("TASK 1: T")
    assert persona_idx < body_idx


def test_plan_prompt_byte_length_unchanged_when_persona_empty():
    """Pin (D7): with persona='', output is byte-identical to pre-task-7.
    test_runner.py::test_plan_prompt_byte_length_matches_baseline asserts 734."""
    out = build_plan_prompt({"id": 1, "title": "T", "description": "D"}, "p.md")
    assert len(out) == 734


def test_review_prompt_default_persona_is_empty():
    out = build_review_prompt(
        {"id": 1, "title": "T", "description": "D"}, "p.md", "r.md",
    )
    assert "## Project conventions" not in out


def test_review_prompt_prepends_persona_when_supplied():
    persona = "## Project conventions\n\nREVIEW_PERSONA\n\n"
    out = build_review_prompt(
        {"id": 1, "title": "T", "description": "D"}, "p.md", "r.md",
        persona=persona,
    )
    assert out.startswith("## Project conventions")
    assert "REVIEW_PERSONA" in out
    # Original body still present
    assert "You are reviewing" in out


def test_revise_prompt_default_persona_is_empty():
    out = build_revise_prompt(
        {"id": 1, "title": "T", "description": "D"}, "p.md", "r.md",
    )
    assert "## Project conventions" not in out


def test_revise_prompt_prepends_persona_when_supplied():
    persona = "## Project conventions\n\nREVISE_PERSONA\n\n"
    out = build_revise_prompt(
        {"id": 1, "title": "T", "description": "D"}, "p.md", "r.md",
        persona=persona,
    )
    assert out.startswith("## Project conventions")
    assert "REVISE_PERSONA" in out
    assert "You are revising" in out


def test_implement_prompt_default_persona_is_empty():
    out = build_implement_prompt(
        {"id": 1, "title": "T", "description": "D"}, "p.md", "msg",
    )
    assert "## Project conventions" not in out


def test_implement_prompt_prepends_persona_when_supplied():
    persona = "## Project conventions\n\nIMPL_PERSONA\n\n"
    out = build_implement_prompt(
        {"id": 1, "title": "T", "description": "D"}, "p.md", "msg",
        persona=persona,
    )
    assert out.startswith("## Project conventions")
    assert "IMPL_PERSONA" in out
    assert "You are implementing" in out


def test_implement_prompt_byte_length_unchanged_when_persona_empty():
    """Pin: post-task-6-v0.3 baseline preserved when persona defaults to ''.

    Task 6 v0.3 §D1 added the WIP-commit / progress.md instruction block.
    """
    out = build_implement_prompt(
        {"id": 1, "title": "T", "description": "D"}, "p.md", "msg",
    )
    assert len(out) == 1129


def test_implement_prompt_no_longer_mentions_tests():
    """Task 12 §D4: the implement prompt no longer instructs the agent
    to run tests. The `tests` gate runs them after implement."""
    out = build_implement_prompt(
        {"id": 1, "title": "T", "description": "D"}, "p.md", "msg",
    )
    assert "Run tests" not in out
    assert "test failures" not in out
    assert "pytest" not in out


def test_implement_prompt_final_commit_is_step_6():
    """Renumbered: task 6 v0.3 §D1 added WIP-commit + progress.md
    instructions. Step 3 is now WIP commit chunks; step 4 is checklist
    maintenance; step 5 is clean-stop on low turn budget; step 6 is
    the final commit. Pin the new shape so a future format change is
    intentional."""
    out = build_implement_prompt(
        {"id": 1, "title": "T", "description": "D"}, "p.md", "msg",
    )
    assert "6. After every chunk is committed and the checklist reads all" in out
    # Step 7 must NOT exist (the prompt ends with step 6).
    assert "7. " not in out


def test_implement_prompt_includes_wip_commit_instruction():
    """Task 6 v0.3 §D1: the WIP commit message format is
    `task-{id} WIP: <chunk>`."""
    out = build_implement_prompt(
        {"id": 7, "title": "T", "description": "D"}, "p.md", "msg",
    )
    assert "task-7 WIP:" in out
    assert "git add -A && git commit -m" in out


def test_implement_prompt_includes_progress_md_path():
    """Task 6 v0.3 §D1: the prompt references the per-task progress.md
    sidecar path under plans/task-{id}/."""
    out = build_implement_prompt(
        {"id": 42, "title": "T", "description": "D"}, "p.md", "msg",
    )
    assert "plans/task-42/progress.md" in out


def test_implement_prompt_includes_low_turn_budget_clean_stop():
    """Task 6 v0.3 §D1: the prompt instructs the agent to stop cleanly
    when the turn budget is running low (so the runner can resume)."""
    out = build_implement_prompt(
        {"id": 1, "title": "T", "description": "D"}, "p.md", "msg",
    )
    assert "turn budget" in out
    assert "finish the session cleanly" in out


def test_implement_prompt_includes_progress_marker_legend():
    """Task 6 v0.3 §D1: the three states the runner relies on for
    routing are explicitly enumerated."""
    out = build_implement_prompt(
        {"id": 1, "title": "T", "description": "D"}, "p.md", "msg",
    )
    assert "[ ] not started" in out
    assert "[~] in progress" in out
    assert "[x] done" in out


def test_implement_prompt_signature_does_not_accept_test_command():
    """Pin §D4: removing the parameter is a clean break. A caller
    passing a 4th positional arg (the old test_command shape) raises
    TypeError because the next param is keyword-only."""
    with pytest.raises(TypeError):
        build_implement_prompt(
            {"id": 1, "title": "T", "description": "D"}, "p.md", "msg",
            "would-be-test-command-positional",
        )


def test_persona_kwarg_is_keyword_only_on_plan_prompt():
    """The * separator forces persona to be passed by keyword. Positional
    callers (existing runner + tests) keep working; new callers must be
    explicit."""
    with pytest.raises(TypeError):
        # Trying to pass persona positionally — must fail.
        build_plan_prompt(
            {"id": 1, "title": "T", "description": "D"}, "p.md",
            "## Project conventions\n\n",  # would-be positional persona
        )


def test_review_prompt_accepts_positional_prev_feedback_and_keyword_persona():
    """Regression: the existing runner.py call site passes prev_feedback
    positionally — `build_review_prompt(task, plan, review, prev_feedback)`
    — and the new signature `(task, plan, review, prev_feedback="", *,
    persona="")` keeps prev_feedback as positional-or-keyword. Confirm
    both forms still work and persona arrives via kwarg only."""
    task = {"id": 1, "title": "T", "description": "D"}
    # Positional prev_feedback (existing runner.py form), keyword persona.
    out = build_review_prompt(
        task, "p.md", "r.md", "previous notes",
        persona="## Project conventions\n\nP\n\n",
    )
    assert "## Project conventions" in out
    # Body is still produced when prev_feedback is non-empty (the original
    # behavior — feedback_section is added but prev_feedback content is
    # NOT inlined; same as test_review_prompt_does_NOT_inline_prev_feedback_body).
    assert "previous notes" not in out
    assert "revised based on previous feedback" in out


# ---------------------------------------------------------------------------
# D. Prompt builders — decisions kwarg (task 8)
# ---------------------------------------------------------------------------


def test_plan_prompt_default_decisions_is_empty():
    out = build_plan_prompt({"id": 1, "title": "T", "description": "D"}, "p.md")
    assert "## Architectural decisions on file" not in out


def test_plan_prompt_accepts_decisions_kwarg():
    decisions = "## Architectural decisions on file\n\n- [001](x.md) — Foo\n\n"
    out = build_plan_prompt(
        {"id": 1, "title": "T", "description": "D"}, "p.md",
        decisions=decisions,
    )
    assert "Foo" in out
    assert "TASK 1: T" in out


def test_plan_prompt_byte_length_unchanged_when_decisions_empty():
    """Pin (D9): with decisions='' (and persona='' default), output is
    byte-identical to pre-task-7. Same fixed input as
    test_plan_prompt_byte_length_matches_baseline in test_runner.py."""
    out = build_plan_prompt({"id": 1, "title": "T", "description": "D"}, "p.md")
    assert len(out) == 734


# ---------------------------------------------------------------------------
# E. build_failure_context_section + kwarg wiring (task 10)
# ---------------------------------------------------------------------------

import re

from claude_plan_execute.prompts import build_failure_context_section
from claude_plan_execute.state import make_failure_record


# E1. Empty / null inputs


def test_failure_context_empty_list_returns_empty_string():
    assert build_failure_context_section([]) == ""


def test_failure_context_None_returns_empty_string():
    assert build_failure_context_section(None) == ""


# E2. Header references most-recent attempt's phase


def test_failure_context_header_uses_last_attempt_phase():
    attempts = [
        make_failure_record("manual_stop", "plan", reason="x"),
        make_failure_record("implementer_timeout", "implement", reason="y"),
    ]
    out = build_failure_context_section(attempts)
    assert "[FAILURE CONTEXT — previous attempt failed in phase: implement]" in out


def test_failure_context_header_falls_back_when_phase_missing():
    """Defensive: a corrupt attempt with no 'phase' key renders 'unknown'."""
    out = build_failure_context_section([{
        "failure_class": "x", "reason": "r", "verbatim": "",
        "artifacts": {}, "hypothesis": None,
    }])
    assert "phase: unknown" in out


# E3. Per-attempt body — required fields


def test_failure_context_renders_attempt_failure_class():
    out = build_failure_context_section([
        make_failure_record("review_rejected_after_max_rounds", "review", reason="r"),
    ])
    assert "Attempt 1: review_rejected_after_max_rounds" in out


def test_failure_context_always_renders_reason():
    out = build_failure_context_section([
        make_failure_record("manual_stop", "plan", reason="MARKER_REASON"),
    ])
    assert "- Reason: MARKER_REASON" in out


def test_failure_context_numbers_attempts_starting_at_1():
    attempts = [
        make_failure_record("manual_stop", "plan", reason="a"),
        make_failure_record("manual_stop", "plan", reason="b"),
        make_failure_record("manual_stop", "plan", reason="c"),
    ]
    out = build_failure_context_section(attempts)
    assert "Attempt 1: " in out
    assert "Attempt 2: " in out
    assert "Attempt 3: " in out


# E4. Conditional fields — verbatim


def test_failure_context_renders_verbatim_when_present():
    attempts = [make_failure_record(
        "review_rejected_after_max_rounds", "review",
        reason="r", verbatim="MY_VERBATIM_TEXT",
    )]
    out = build_failure_context_section(attempts)
    assert "MY_VERBATIM_TEXT" in out


def test_failure_context_skips_verbatim_line_when_empty():
    attempts = [make_failure_record(
        "manual_stop", "plan", reason="r", verbatim="",
    )]
    out = build_failure_context_section(attempts)
    assert "Verbatim" not in out


def test_failure_context_uses_class_specific_label_for_review():
    attempts = [make_failure_record(
        "review_rejected_after_max_rounds", "review",
        reason="r", verbatim="x",
    )]
    out = build_failure_context_section(attempts)
    assert "Verbatim (last review): x" in out


def test_failure_context_uses_class_specific_label_for_timeout():
    attempts = [make_failure_record(
        "implementer_timeout", "implement",
        reason="r", verbatim="x",
    )]
    out = build_failure_context_section(attempts)
    assert "Verbatim (stderr): x" in out


def test_failure_context_uses_bare_label_for_unknown_class():
    attempts = [{
        "failure_class": "completely_made_up", "phase": "plan",
        "reason": "r", "verbatim": "x", "artifacts": {}, "hypothesis": None,
    }]
    out = build_failure_context_section(attempts)
    assert "Verbatim: x" in out
    assert "Verbatim (" not in out


# E5. Conditional fields — hypothesis


def test_failure_context_renders_hypothesis_when_set():
    attempts = [make_failure_record(
        "manual_stop", "plan", reason="r", hypothesis="MY_HYPOTHESIS",
    )]
    out = build_failure_context_section(attempts)
    assert "- Hypothesis: MY_HYPOTHESIS" in out


def test_failure_context_skips_hypothesis_when_None():
    attempts = [make_failure_record("manual_stop", "plan", reason="r")]
    out = build_failure_context_section(attempts)
    assert "Hypothesis" not in out


def test_failure_context_skips_hypothesis_when_empty_string():
    attempts = [make_failure_record(
        "manual_stop", "plan", reason="r", hypothesis="",
    )]
    out = build_failure_context_section(attempts)
    assert "Hypothesis" not in out


def test_failure_context_skips_hypothesis_when_whitespace_only():
    attempts = [make_failure_record(
        "manual_stop", "plan", reason="r", hypothesis="   \n  ",
    )]
    out = build_failure_context_section(attempts)
    assert "Hypothesis" not in out


# E6. Conditional fields — artifacts


def test_failure_context_renders_artifacts_paths_only():
    attempts = [make_failure_record(
        "review_rejected_after_max_rounds", "review",
        reason="r", artifacts={"plan": "p.md", "last_review": "r.md"},
    )]
    out = build_failure_context_section(attempts)
    assert "- Artifacts: p.md, r.md" in out
    assert "plan:" not in out
    assert "last_review:" not in out


def test_failure_context_skips_artifacts_when_empty():
    attempts = [make_failure_record(
        "manual_stop", "plan", reason="r", artifacts={},
    )]
    out = build_failure_context_section(attempts)
    assert "Artifacts" not in out


# E7. Footer — instruction list


def test_failure_context_footer_always_includes_address_prior_failure():
    attempts = [make_failure_record("manual_stop", "plan", reason="r")]
    out = build_failure_context_section(attempts)
    assert "1. Address the prior failure verbatim before adding new work" in out


def test_failure_context_footer_includes_hypothesis_check_when_any_set():
    attempts = [
        make_failure_record("manual_stop", "plan", reason="r"),
        make_failure_record("manual_stop", "plan", reason="r2", hypothesis="H"),
    ]
    out = build_failure_context_section(attempts)
    assert "2. Specifically check the hypothesis above" in out


def test_failure_context_footer_omits_hypothesis_check_when_none_set():
    attempts = [
        make_failure_record("manual_stop", "plan", reason="r"),
        make_failure_record("manual_stop", "plan", reason="r2"),
    ]
    out = build_failure_context_section(attempts)
    assert "check the hypothesis" not in out


def test_failure_context_ends_with_END_marker():
    attempts = [make_failure_record("manual_stop", "plan", reason="r")]
    out = build_failure_context_section(attempts)
    assert "[END FAILURE CONTEXT]\n\n" in out


def test_failure_context_ends_with_double_newline_for_caller_concat():
    """Caller does `failure_context + persona + ...`; the trailing
    '\\n\\n' supplies the separator with no manual padding."""
    out = build_failure_context_section([
        make_failure_record("manual_stop", "plan", reason="r"),
    ])
    assert out.endswith("\n\n")


# E8. 8 KB cap — drop oldest


def test_failure_context_drops_oldest_when_block_exceeds_8kb():
    """Three attempts with 4-KB verbatims each will overflow 8 KB.
    The render keeps the most recent ones and prepends an omission marker."""
    big = "X" * 4096
    attempts = [
        make_failure_record("manual_stop", "plan", reason=f"a{i}", verbatim=big)
        for i in range(3)
    ]
    out = build_failure_context_section(attempts)
    assert len(out.encode("utf-8")) <= 8192
    assert "earlier attempts omitted for length" in out


def test_failure_context_omission_marker_uses_correct_count():
    """If 2 of 3 attempts are dropped, the marker reads '[2 earlier attempts...]'."""
    huge = "X" * 4096
    attempts = [
        make_failure_record("manual_stop", "plan", reason=f"a{i}", verbatim=huge)
        for i in range(3)
    ]
    out = build_failure_context_section(attempts)
    assert "Attempt 1:" not in out
    marker = re.search(r"\[(\d+) earlier attempts omitted for length\]", out)
    assert marker is not None
    n_dropped = int(marker.group(1))
    first_kept = re.search(r"Attempt (\d+):", out)
    assert int(first_kept.group(1)) == n_dropped + 1


def test_failure_context_keeps_most_recent_attempt():
    """When dropping for budget, the most recent attempt (carrying
    the freshest evidence) is always retained."""
    huge = "X" * 4096
    attempts = [
        make_failure_record("manual_stop", "plan", reason="OLDEST_R", verbatim=huge),
        make_failure_record("manual_stop", "plan", reason="MIDDLE_R", verbatim=huge),
        make_failure_record("manual_stop", "plan", reason="NEWEST_R", verbatim=huge),
    ]
    out = build_failure_context_section(attempts)
    assert "NEWEST_R" in out
    assert "OLDEST_R" not in out


def test_failure_context_no_marker_when_all_attempts_fit():
    """A single small attempt fits comfortably; no omission marker."""
    attempts = [make_failure_record(
        "manual_stop", "plan", reason="short", verbatim="tiny",
    )]
    out = build_failure_context_section(attempts)
    assert "earlier attempts omitted" not in out


def test_failure_context_block_size_capped_for_pathological_single_attempt():
    """If a single attempt's render still exceeds 8 KB (e.g., huge
    artifacts dict), the rendered string is hard-truncated at the cap
    with a '[truncated]' marker."""
    huge_paths = {f"k{i}": "X" * 200 for i in range(50)}
    attempts = [{
        "failure_class": "manual_stop", "phase": "plan", "reason": "r",
        "verbatim": "", "artifacts": huge_paths, "hypothesis": None,
    }]
    out = build_failure_context_section(attempts)
    assert len(out.encode("utf-8")) <= 8192 + 20


# E8b. Defensive — non-dict entries skipped


def test_failure_context_skips_non_dict_entries():
    """Hand-edited state.json could in principle contain non-dict
    elements in `attempts`. The render filters them silently rather
    than raising AttributeError mid-prompt-build."""
    attempts = [
        "this is a string, not a dict",
        make_failure_record("manual_stop", "plan", reason="REAL_REASON"),
        12345,
    ]
    out = build_failure_context_section(attempts)
    assert "REAL_REASON" in out
    assert "Attempt 1: manual_stop" in out
    assert "this is a string" not in out
    assert "12345" not in out


def test_failure_context_returns_empty_when_all_entries_non_dict():
    """If filtering leaves nothing, behave as if the list were empty."""
    out = build_failure_context_section(["junk", 123, None])
    assert out == ""


# E9. Verbatim trusted as-pre-truncated


def test_failure_context_does_not_double_truncate_verbatim():
    """make_failure_record already caps verbatim at 4 KB with its own
    marker. The block render must NOT truncate again or add a second marker."""
    big_text = "Y" * 5000
    fr = make_failure_record(
        "review_rejected_after_max_rounds", "review",
        reason="r", verbatim=big_text,
    )
    assert "[truncated:" in fr["verbatim"]
    out = build_failure_context_section([fr])
    assert out.count("[truncated:") == 1


# E10. Plan/Implement kwarg wiring


def test_plan_prompt_default_failure_context_is_empty():
    out = build_plan_prompt({"id": 1, "title": "T", "description": "D"}, "p.md")
    assert "[FAILURE CONTEXT" not in out


def test_plan_prompt_prepends_failure_context_when_supplied():
    fc = "[FAILURE CONTEXT — previous attempt failed in phase: plan]\nFAKE_BLOCK\n[END FAILURE CONTEXT]\n\n"
    out = build_plan_prompt(
        {"id": 1, "title": "T", "description": "D"}, "p.md",
        failure_context=fc,
    )
    assert out.startswith("[FAILURE CONTEXT")
    assert "FAKE_BLOCK" in out
    assert "TASK 1: T" in out


def test_plan_prompt_failure_context_appears_before_persona():
    """§D1 ordering: failure_context, then persona, then decisions, then body."""
    fc = "FCMARKER\n\n"
    persona = "PERSONAMARKER\n\n"
    out = build_plan_prompt(
        {"id": 1, "title": "T", "description": "D"}, "p.md",
        persona=persona, failure_context=fc,
    )
    fc_idx = out.index("FCMARKER")
    p_idx = out.index("PERSONAMARKER")
    assert fc_idx < p_idx


def test_plan_prompt_failure_context_appears_before_decisions():
    fc = "FCMARKER\n\n"
    decisions = "DECISIONSMARKER\n\n"
    out = build_plan_prompt(
        {"id": 1, "title": "T", "description": "D"}, "p.md",
        decisions=decisions, failure_context=fc,
    )
    fc_idx = out.index("FCMARKER")
    d_idx = out.index("DECISIONSMARKER")
    assert fc_idx < d_idx


def test_plan_prompt_failure_context_keyword_only():
    """The * separator forces failure_context to be keyword-only."""
    with pytest.raises(TypeError):
        build_plan_prompt(
            {"id": 1, "title": "T", "description": "D"}, "p.md",
            "would-be-positional-failure-context",
        )


def test_plan_prompt_byte_length_unchanged_when_failure_context_empty():
    """Pin (D10): with failure_context='' (and persona/decisions defaults),
    the byte count matches the existing 734 baseline."""
    out = build_plan_prompt({"id": 1, "title": "T", "description": "D"}, "p.md")
    assert len(out) == 734


def test_implement_prompt_default_failure_context_is_empty():
    out = build_implement_prompt(
        {"id": 1, "title": "T", "description": "D"}, "p.md", "msg",
    )
    assert "[FAILURE CONTEXT" not in out


def test_implement_prompt_prepends_failure_context_when_supplied():
    fc = "FCMARKER\n\n"
    out = build_implement_prompt(
        {"id": 1, "title": "T", "description": "D"}, "p.md", "msg",
        failure_context=fc,
    )
    assert out.startswith("FCMARKER")
    assert "You are implementing" in out


def test_implement_prompt_failure_context_appears_before_persona():
    fc = "FCMARKER\n\n"
    persona = "PERSONAMARKER\n\n"
    out = build_implement_prompt(
        {"id": 1, "title": "T", "description": "D"}, "p.md", "msg",
        persona=persona, failure_context=fc,
    )
    fc_idx = out.index("FCMARKER")
    p_idx = out.index("PERSONAMARKER")
    assert fc_idx < p_idx


def test_implement_prompt_failure_context_keyword_only():
    with pytest.raises(TypeError):
        build_implement_prompt(
            {"id": 1, "title": "T", "description": "D"}, "p.md", "msg",
            "would-be-positional",
        )


def test_implement_prompt_byte_length_unchanged_when_failure_context_empty():
    """Pin: post-task-6-v0.3 baseline preserved when failure_context defaults to ''.

    Task 6 v0.3 §D1 added the WIP-commit / progress.md instruction block.
    """
    out = build_implement_prompt(
        {"id": 1, "title": "T", "description": "D"}, "p.md", "msg",
    )
    assert len(out) == 1129


# ---------------------------------------------------------------------------
# F. build_resume_prompt + parse_progress_status (task 6 v0.3)
# ---------------------------------------------------------------------------

from claude_plan_execute.prompts import (
    build_resume_prompt,
    parse_progress_status,
)


# F1. parse_progress_status

def test_parse_progress_status_empty_string_returns_zeros():
    assert parse_progress_status("") == (0, 0, 0)


def test_parse_progress_status_no_list_items_returns_zeros():
    assert parse_progress_status("# Heading\n\nA paragraph.") == (0, 0, 0)


def test_parse_progress_status_counts_done():
    assert parse_progress_status("- [x] one\n- [x] two") == (2, 0, 0)


def test_parse_progress_status_counts_in_progress():
    assert parse_progress_status("- [~] one\n- [~] two") == (0, 2, 0)


def test_parse_progress_status_counts_pending():
    assert parse_progress_status("- [ ] one\n- [ ] two") == (0, 0, 2)


def test_parse_progress_status_mixed_buckets():
    text = "- [x] done\n- [~] mid\n- [ ] todo\n- [x] also-done"
    assert parse_progress_status(text) == (2, 1, 1)


def test_parse_progress_status_accepts_indentation():
    assert parse_progress_status("    - [x] indented") == (1, 0, 0)


def test_parse_progress_status_accepts_star_and_plus_bullets():
    text = "* [x] one\n+ [ ] two"
    assert parse_progress_status(text) == (1, 0, 1)


def test_parse_progress_status_capital_X_is_NOT_counted():
    """Pin §D3: capital `X` is not recognized; the prompt mandates lower-case."""
    assert parse_progress_status("- [X] not counted") == (0, 0, 0)


def test_parse_progress_status_unknown_marker_not_counted():
    assert parse_progress_status("- [?] huh\n- [-] partial") == (0, 0, 0)


# F2. build_resume_prompt — basic shape

def test_resume_prompt_includes_task_id_and_title():
    out = build_resume_prompt(
        {"id": 7, "title": "Refactor", "description": "D"},
        "p.md", "prog.md", "- [~] item\n- [ ] item2", "", "msg",
    )
    assert "TASK 7: Refactor" in out


def test_resume_prompt_embeds_checklist_verbatim():
    progress = "- [x] step 1\n- [~] step 2\n- [ ] step 3"
    out = build_resume_prompt(
        {"id": 1, "title": "T", "description": "D"},
        "p.md", "prog.md", progress, "", "msg",
    )
    assert progress in out


def test_resume_prompt_embeds_progress_path():
    out = build_resume_prompt(
        {"id": 1, "title": "T", "description": "D"},
        "plans/task-1/plan.md", "plans/task-1/progress.md",
        "- [~] x", "", "msg",
    )
    assert "plans/task-1/progress.md" in out


def test_resume_prompt_includes_commit_message():
    out = build_resume_prompt(
        {"id": 1, "title": "T", "description": "D"},
        "p.md", "prog.md", "- [ ] x", "", "feat: my message",
    )
    assert "feat: my message" in out


# F3. build_resume_prompt — dirty-tree branch

def test_resume_prompt_dirty_tree_renders_porcelain_block():
    porcelain = " M src/foo.py\n?? new_file.py"
    out = build_resume_prompt(
        {"id": 1, "title": "T", "description": "D"},
        "p.md", "prog.md", "- [~] x", porcelain, "msg",
    )
    assert "Uncommitted hunks detected" in out
    assert "src/foo.py" in out
    assert "new_file.py" in out
    assert "task-1 WIP:" in out          # the (a) commit-as-chunk hint
    assert "git restore" in out          # the (b) revert hint


def test_resume_prompt_clean_tree_renders_clean_block():
    out = build_resume_prompt(
        {"id": 1, "title": "T", "description": "D"},
        "p.md", "prog.md", "- [~] x", "", "msg",
    )
    assert "Working tree is clean" in out
    assert "Uncommitted hunks detected" not in out


def test_resume_prompt_whitespace_only_porcelain_treated_as_clean():
    out = build_resume_prompt(
        {"id": 1, "title": "T", "description": "D"},
        "p.md", "prog.md", "- [~] x", "  \n  ", "msg",
    )
    assert "Working tree is clean" in out


# F4. build_resume_prompt — composition (persona + failure_context)

def test_resume_prompt_default_persona_is_empty():
    out = build_resume_prompt(
        {"id": 1, "title": "T", "description": "D"},
        "p.md", "prog.md", "- [~] x", "", "msg",
    )
    assert "## Project conventions" not in out


def test_resume_prompt_prepends_persona_when_supplied():
    persona = "## Project conventions\n\nP_MARKER\n\n"
    out = build_resume_prompt(
        {"id": 1, "title": "T", "description": "D"},
        "p.md", "prog.md", "- [~] x", "", "msg",
        persona=persona,
    )
    assert out.startswith("## Project conventions")
    assert "P_MARKER" in out
    assert "RESUMING" in out


def test_resume_prompt_default_failure_context_is_empty():
    out = build_resume_prompt(
        {"id": 1, "title": "T", "description": "D"},
        "p.md", "prog.md", "- [~] x", "", "msg",
    )
    assert "[FAILURE CONTEXT" not in out


def test_resume_prompt_prepends_failure_context_when_supplied():
    fc = "FCMARKER\n\n"
    out = build_resume_prompt(
        {"id": 1, "title": "T", "description": "D"},
        "p.md", "prog.md", "- [~] x", "", "msg",
        failure_context=fc,
    )
    assert out.startswith("FCMARKER")


def test_resume_prompt_failure_context_appears_before_persona():
    fc = "FCMARKER\n\n"
    persona = "PMARKER\n\n"
    out = build_resume_prompt(
        {"id": 1, "title": "T", "description": "D"},
        "p.md", "prog.md", "- [~] x", "", "msg",
        persona=persona, failure_context=fc,
    )
    assert out.index("FCMARKER") < out.index("PMARKER")


def test_resume_prompt_persona_and_failure_context_are_keyword_only():
    """The * separator forces both kwargs to keyword-only."""
    with pytest.raises(TypeError):
        build_resume_prompt(
            {"id": 1, "title": "T", "description": "D"},
            "p.md", "prog.md", "- [~] x", "", "msg",
            "would-be-positional-persona",
        )


def test_resume_prompt_does_not_accept_decisions_kwarg():
    """Pin §D9: resume builder does NOT accept `decisions=`. Same
    decision path as build_implement_prompt; mirrors the
    'decisions are Plan-only' regression."""
    with pytest.raises(TypeError):
        build_resume_prompt(
            {"id": 1, "title": "T", "description": "D"},
            "p.md", "prog.md", "- [~] x", "", "msg",
            decisions="should not be accepted",
        )
