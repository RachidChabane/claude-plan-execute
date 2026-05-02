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
        {"id": 1, "title": "T", "description": "D"}, "p.md", "pytest", "msg",
    )
    assert "## Project conventions" not in out


def test_implement_prompt_prepends_persona_when_supplied():
    persona = "## Project conventions\n\nIMPL_PERSONA\n\n"
    out = build_implement_prompt(
        {"id": 1, "title": "T", "description": "D"}, "p.md", "pytest", "msg",
        persona=persona,
    )
    assert out.startswith("## Project conventions")
    assert "IMPL_PERSONA" in out
    assert "You are implementing" in out


def test_implement_prompt_byte_length_unchanged_when_persona_empty():
    """Pin (D7): with persona='', output is byte-identical to pre-task-7.
    test_runner.py::test_implement_prompt_byte_length_matches_baseline asserts 415."""
    out = build_implement_prompt(
        {"id": 1, "title": "T", "description": "D"}, "p.md", "pytest", "msg",
    )
    assert len(out) == 415


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
