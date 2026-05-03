"""Tests for decisions log loader, index builder, and CLI handler (task 8).

Covers:
- claude_plan_execute.loader.parse_frontmatter (pure parser)
- claude_plan_execute.loader.load_decisions_index (file I/O, line cap)
- claude_plan_execute.loader.scan_decisions (directory walk, frontmatter validation)
- claude_plan_execute.loader.render_decisions_index (pure rendering)
- claude_plan_execute.prompts.build_decisions_section (pure composition)
- claude_plan_execute.prompts.build_plan_prompt with decisions kwarg (wiring)
- claude_plan_execute.cli.cmd_decisions_index (handler dispatch)
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from claude_plan_execute.cli import cmd_decisions_index
from claude_plan_execute.loader import (
    load_decisions_index,
    parse_frontmatter,
    render_decisions_index,
    scan_decisions,
)
from claude_plan_execute.prompts import (
    build_decisions_section,
    build_plan_prompt,
    build_implement_prompt,
    build_review_prompt,
    build_revise_prompt,
)


# ---------------------------------------------------------------------------
# A. parse_frontmatter
# ---------------------------------------------------------------------------


def test_parse_frontmatter_returns_empty_dict_and_full_text_when_no_delimiters():
    fm, body = parse_frontmatter("# Title\n\nBody.")
    assert fm == {}
    assert body == "# Title\n\nBody."


def test_parse_frontmatter_returns_dict_when_well_formed():
    text = "---\nid: 1\nstatus: accepted\n---\n# Title\n\nBody."
    fm, body = parse_frontmatter(text)
    assert fm == {"id": 1, "status": "accepted"}
    assert body == "# Title\n\nBody."


def test_parse_frontmatter_handles_list_scope():
    text = "---\nid: 7\nstatus: accepted\nscope:\n  - retrieval\n  - storage\n---\n"
    fm, _ = parse_frontmatter(text)
    assert fm["scope"] == ["retrieval", "storage"]


def test_parse_frontmatter_raises_on_malformed_yaml():
    # Unclosed flow sequence — reliably broken across PyYAML versions.
    text = "---\nfoo: [unterminated\n---\nBody"
    with pytest.raises(yaml.YAMLError):
        parse_frontmatter(text)


def test_parse_frontmatter_raises_on_non_mapping_frontmatter():
    """Frontmatter that's a scalar/list/null is rejected."""
    text = "---\n- a\n- b\n---\nBody"
    with pytest.raises(yaml.YAMLError):
        parse_frontmatter(text)


def test_parse_frontmatter_no_leading_delimiter_returns_empty_fm():
    """Frontmatter must start at line 0 — a `---` later in the file
    is not frontmatter."""
    text = "Some intro.\n\n---\nid: 1\n---\n"
    fm, body = parse_frontmatter(text)
    assert fm == {}
    assert body == text


# ---------------------------------------------------------------------------
# B. load_decisions_index
# ---------------------------------------------------------------------------


def test_load_decisions_index_returns_empty_for_None():
    assert load_decisions_index(None) == ""


def test_load_decisions_index_returns_empty_for_empty_string_path():
    assert load_decisions_index("") == ""


def test_load_decisions_index_returns_empty_for_missing_dir(tmp_path: Path):
    assert load_decisions_index(tmp_path / "nope") == ""


def test_load_decisions_index_returns_empty_when_decisions_dir_is_a_file(tmp_path: Path):
    f = tmp_path / "decisions"
    f.write_text("not a dir")
    assert load_decisions_index(f) == ""


def test_load_decisions_index_returns_empty_when_index_md_missing(tmp_path: Path):
    """Common case: directory exists but no index has been built yet."""
    d = tmp_path / "decisions"
    d.mkdir()
    assert load_decisions_index(d) == ""


def test_load_decisions_index_returns_empty_when_index_is_a_directory(tmp_path: Path):
    d = tmp_path / "decisions"
    d.mkdir()
    (d / "index.md").mkdir()
    assert load_decisions_index(d) == ""


def test_load_decisions_index_returns_content_when_present(tmp_path: Path):
    d = tmp_path / "decisions"
    d.mkdir()
    (d / "index.md").write_text("- [001](one.md) — One")
    assert load_decisions_index(d) == "- [001](one.md) — One"


def test_load_decisions_index_accepts_string_path(tmp_path: Path):
    d = tmp_path / "decisions"
    d.mkdir()
    (d / "index.md").write_text("hi")
    assert load_decisions_index(str(d)) == "hi"


def test_load_decisions_index_at_exactly_200_lines_returns_content(tmp_path: Path):
    """Boundary: 200 lines is allowed (cap is `> 200`)."""
    d = tmp_path / "decisions"
    d.mkdir()
    (d / "index.md").write_text("\n".join(f"- entry {i}" for i in range(200)))
    assert load_decisions_index(d) != ""


def test_load_decisions_index_over_200_lines_returns_empty(tmp_path: Path):
    d = tmp_path / "decisions"
    d.mkdir()
    (d / "index.md").write_text("\n".join(f"- entry {i}" for i in range(201)))
    assert load_decisions_index(d) == ""


def test_load_decisions_index_over_200_lines_emits_warning(tmp_path: Path, capsys):
    d = tmp_path / "decisions"
    d.mkdir()
    idx = d / "index.md"
    idx.write_text("\n".join(f"- entry {i}" for i in range(250)))
    load_decisions_index(d)
    out = capsys.readouterr().out
    assert "Warning" in out
    assert "200" in out
    assert "250" in out
    assert str(idx) in out


def test_load_decisions_index_no_warning_for_under_cap(tmp_path: Path, capsys):
    d = tmp_path / "decisions"
    d.mkdir()
    (d / "index.md").write_text("- one\n- two")
    load_decisions_index(d)
    out = capsys.readouterr().out
    assert "Warning" not in out


def test_load_decisions_index_no_warning_for_missing_dir(tmp_path: Path, capsys):
    load_decisions_index(tmp_path / "nope")
    out = capsys.readouterr().out
    assert "Warning" not in out


def test_load_decisions_index_uses_utf8(tmp_path: Path):
    d = tmp_path / "decisions"
    d.mkdir()
    text = "- [001](one.md) — أول قرار"
    (d / "index.md").write_text(text, encoding="utf-8")
    assert load_decisions_index(d) == text


# ---------------------------------------------------------------------------
# C. scan_decisions
# ---------------------------------------------------------------------------


def _write_decision(d: Path, filename: str, frontmatter: dict, body: str = ""):
    fm_yaml = yaml.safe_dump(frontmatter, sort_keys=False).strip()
    text = f"---\n{fm_yaml}\n---\n{body}"
    (d / filename).write_text(text)


def test_scan_decisions_returns_empty_for_empty_dir(tmp_path: Path):
    d = tmp_path / "decisions"
    d.mkdir()
    assert scan_decisions(d) == []


def test_scan_decisions_skips_index_md(tmp_path: Path):
    d = tmp_path / "decisions"
    d.mkdir()
    _write_decision(d, "001-real.md", {"id": 1, "status": "accepted"}, "# A")
    (d / "index.md").write_text("- existing index, must be ignored")
    entries = scan_decisions(d)
    assert len(entries) == 1
    assert entries[0]["id"] == 1


def test_scan_decisions_sorts_by_id_ascending(tmp_path: Path):
    d = tmp_path / "decisions"
    d.mkdir()
    _write_decision(d, "003-c.md", {"id": 3, "status": "accepted"}, "# C")
    _write_decision(d, "001-a.md", {"id": 1, "status": "accepted"}, "# A")
    _write_decision(d, "002-b.md", {"id": 2, "status": "accepted"}, "# B")
    entries = scan_decisions(d)
    assert [e["id"] for e in entries] == [1, 2, 3]


def test_scan_decisions_extracts_h1_when_no_title_field(tmp_path: Path):
    d = tmp_path / "decisions"
    d.mkdir()
    _write_decision(d, "001-x.md", {"id": 1, "status": "accepted"},
                    "# Postgres GIN with stemmer\n\nBody.")
    [entry] = scan_decisions(d)
    assert entry["title"] == "Postgres GIN with stemmer"


def test_scan_decisions_prefers_frontmatter_title_over_h1(tmp_path: Path):
    d = tmp_path / "decisions"
    d.mkdir()
    _write_decision(d, "001-x.md",
                    {"id": 1, "status": "accepted", "title": "FM Title"},
                    "# H1 Title\n")
    [entry] = scan_decisions(d)
    assert entry["title"] == "FM Title"


def test_scan_decisions_falls_back_to_filename_stem_when_no_title(tmp_path: Path):
    d = tmp_path / "decisions"
    d.mkdir()
    _write_decision(d, "001-citation-table-shape.md",
                    {"id": 1, "status": "accepted"}, "Body without H1.")
    [entry] = scan_decisions(d)
    assert entry["title"] == "citation table shape"


def test_scan_decisions_skips_file_without_frontmatter(tmp_path: Path, capsys):
    d = tmp_path / "decisions"
    d.mkdir()
    (d / "001-no-fm.md").write_text("# No frontmatter\n\nBody")
    entries = scan_decisions(d)
    assert entries == []
    out = capsys.readouterr().out
    assert "Warning" in out
    assert "no YAML frontmatter" in out


def test_scan_decisions_skips_file_with_malformed_yaml(tmp_path: Path, capsys):
    d = tmp_path / "decisions"
    d.mkdir()
    # Unclosed flow sequence — reliably raises across PyYAML versions.
    (d / "001-bad.md").write_text("---\nfoo: [unterminated\n---\n")
    entries = scan_decisions(d)
    assert entries == []
    out = capsys.readouterr().out
    assert "Warning" in out


def test_scan_decisions_skips_file_missing_id(tmp_path: Path, capsys):
    d = tmp_path / "decisions"
    d.mkdir()
    _write_decision(d, "noid.md", {"status": "accepted"})
    assert scan_decisions(d) == []
    out = capsys.readouterr().out
    assert "missing 'id'" in out


def test_scan_decisions_skips_file_with_non_int_id(tmp_path: Path, capsys):
    d = tmp_path / "decisions"
    d.mkdir()
    _write_decision(d, "x.md", {"id": "abc", "status": "accepted"})
    assert scan_decisions(d) == []
    out = capsys.readouterr().out
    assert "not an integer" in out


def test_scan_decisions_skips_file_missing_status(tmp_path: Path, capsys):
    d = tmp_path / "decisions"
    d.mkdir()
    _write_decision(d, "x.md", {"id": 1})
    assert scan_decisions(d) == []
    out = capsys.readouterr().out
    assert "missing 'status'" in out


def test_scan_decisions_warns_on_unknown_status_but_keeps_entry(tmp_path: Path, capsys):
    d = tmp_path / "decisions"
    d.mkdir()
    _write_decision(d, "001-x.md", {"id": 1, "status": "weird"}, "# T")
    entries = scan_decisions(d)
    assert len(entries) == 1
    out = capsys.readouterr().out
    assert "unknown status" in out


def test_scan_decisions_collects_scope_field(tmp_path: Path):
    d = tmp_path / "decisions"
    d.mkdir()
    _write_decision(d, "001-x.md",
                    {"id": 1, "status": "accepted", "scope": ["a", "b"]}, "# T")
    [entry] = scan_decisions(d)
    assert entry["scope"] == ["a", "b"]


def test_scan_decisions_normalizes_scalar_scope_to_list(tmp_path: Path):
    """A scalar scope value (`scope: retrieval`) becomes `['retrieval']`."""
    d = tmp_path / "decisions"
    d.mkdir()
    _write_decision(d, "001-x.md",
                    {"id": 1, "status": "accepted", "scope": "retrieval"}, "# T")
    [entry] = scan_decisions(d)
    assert entry["scope"] == ["retrieval"]


# ---------------------------------------------------------------------------
# D. render_decisions_index
# ---------------------------------------------------------------------------


def test_render_starts_with_generated_banner():
    out = render_decisions_index([])
    assert out.startswith("<!-- Generated by")


def test_render_empty_entries_still_writes_banner():
    out = render_decisions_index([])
    # Banner + blank line + trailing newline → 3 lines total when split.
    assert out.count("\n") >= 1
    assert "Edits will be overwritten" in out


def test_render_zero_pads_id_in_link_text():
    out = render_decisions_index([
        {"id": 7, "status": "accepted", "title": "Foo", "scope": [], "filename": "7.md"},
    ])
    assert "[007](7.md)" in out


def test_render_uses_filename_verbatim():
    out = render_decisions_index([
        {"id": 1, "status": "accepted", "title": "T", "scope": [],
         "filename": "001-postgres-gin.md"},
    ])
    assert "(001-postgres-gin.md)" in out


def test_render_renders_accepted_as_plain_bullet():
    out = render_decisions_index([
        {"id": 1, "status": "accepted", "title": "Accepted thing",
         "scope": [], "filename": "x.md"},
    ])
    assert "- [001](x.md) — Accepted thing" in out
    assert "~~" not in out


def test_render_renders_superseded_with_strikethrough_and_label():
    out = render_decisions_index([
        {"id": 2, "status": "superseded", "title": "Old",
         "scope": [], "filename": "old.md"},
    ])
    assert "~~[002](old.md) — Old~~" in out
    assert "(superseded)" in out


def test_render_renders_deprecated_with_strikethrough_and_label():
    out = render_decisions_index([
        {"id": 3, "status": "deprecated", "title": "Stale",
         "scope": [], "filename": "stale.md"},
    ])
    assert "~~[003](stale.md) — Stale~~" in out
    assert "(deprecated)" in out


def test_render_unknown_status_renders_as_plain_bullet():
    out = render_decisions_index([
        {"id": 4, "status": "weird", "title": "Whatever",
         "scope": [], "filename": "w.md"},
    ])
    # The scanner already warned; the renderer just doesn't strikethrough.
    assert "~~" not in out
    assert "- [004](w.md) — Whatever" in out


def test_render_emits_one_line_per_entry():
    entries = [
        {"id": i, "status": "accepted", "title": f"T{i}",
         "scope": [], "filename": f"{i}.md"}
        for i in (1, 2, 3)
    ]
    out = render_decisions_index(entries)
    bullet_count = out.count("- [")
    assert bullet_count == 3


# ---------------------------------------------------------------------------
# E. build_decisions_section (pure composer)
# ---------------------------------------------------------------------------


def test_build_decisions_section_empty_input_returns_empty():
    assert build_decisions_section("") == ""


def test_build_decisions_section_whitespace_only_returns_empty():
    assert build_decisions_section("\n\n  \t\n") == ""


def test_build_decisions_section_emits_header_when_content():
    out = build_decisions_section("- [001](x.md) — Foo")
    assert out.startswith("## Architectural decisions on file")
    assert "Foo" in out


def test_build_decisions_section_ends_with_double_newline():
    """Caller does `persona + decisions + body`; trailing \\n\\n separates."""
    out = build_decisions_section("- entry")
    assert out.endswith("\n\n")


def test_build_decisions_section_strips_trailing_whitespace():
    out = build_decisions_section("- entry\n\n\n")
    assert "entry\n\n\n" not in out
    assert "entry" in out


# ---------------------------------------------------------------------------
# F. build_plan_prompt accepts decisions kwarg (wiring)
# ---------------------------------------------------------------------------


def test_plan_prompt_default_decisions_is_empty():
    out = build_plan_prompt({"id": 1, "title": "T", "description": "D"}, "p.md")
    assert "## Architectural decisions on file" not in out


def test_plan_prompt_prepends_decisions_when_supplied():
    decisions = "## Architectural decisions on file\n\n- [001](x.md) — Foo\n\n"
    out = build_plan_prompt(
        {"id": 1, "title": "T", "description": "D"}, "p.md",
        decisions=decisions,
    )
    assert out.startswith("## Architectural decisions on file")
    assert "Foo" in out
    assert "TASK 1: T" in out  # body still present


def test_plan_prompt_decisions_appears_after_persona_and_before_body():
    persona = "## Project conventions\n\nPERSONA_MARKER\n\n"
    decisions = "## Architectural decisions on file\n\nDECISIONS_MARKER\n\n"
    out = build_plan_prompt(
        {"id": 1, "title": "T", "description": "D"}, "p.md",
        persona=persona, decisions=decisions,
    )
    p_idx = out.index("PERSONA_MARKER")
    d_idx = out.index("DECISIONS_MARKER")
    body_idx = out.index("TASK 1: T")
    assert p_idx < d_idx < body_idx


def test_plan_prompt_byte_length_unchanged_when_persona_and_decisions_empty():
    """Pin (D9): with persona='' and decisions='', output is byte-identical
    to pre-task-7 (734 bytes). Same fixed input as
    test_plan_prompt_byte_length_matches_baseline in test_runner.py."""
    out = build_plan_prompt({"id": 1, "title": "T", "description": "D"}, "p.md")
    assert len(out) == 734


def test_plan_prompt_decisions_kwarg_is_keyword_only():
    """The * separator forces decisions to be passed by keyword."""
    with pytest.raises(TypeError):
        # Trying to pass decisions positionally — must fail.
        build_plan_prompt(
            {"id": 1, "title": "T", "description": "D"}, "p.md",
            "",  # would-be positional persona
            "## Architectural decisions on file\n\nX\n\n",  # would-be positional decisions
        )


def test_review_prompt_does_NOT_accept_decisions_kwarg():
    """Pin (D1): decisions injection is Plan-only. Other builders reject it."""
    with pytest.raises(TypeError):
        build_review_prompt(
            {"id": 1, "title": "T", "description": "D"}, "p.md", "r.md",
            decisions="x",
        )


def test_revise_prompt_does_NOT_accept_decisions_kwarg():
    with pytest.raises(TypeError):
        build_revise_prompt(
            {"id": 1, "title": "T", "description": "D"}, "p.md", "r.md",
            decisions="x",
        )


def test_implement_prompt_does_NOT_accept_decisions_kwarg():
    with pytest.raises(TypeError):
        build_implement_prompt(
            {"id": 1, "title": "T", "description": "D"}, "p.md", "msg",
            decisions="x",
        )


# ---------------------------------------------------------------------------
# G. cmd_decisions_index (CLI handler)
# ---------------------------------------------------------------------------


def _make_args(**overrides) -> SimpleNamespace:
    defaults = dict(
        command="decisions", action="index",
        tasks=None, dir=None, decisions_dir=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_cmd_decisions_index_returns_1_when_dir_missing(tmp_path: Path):
    args = _make_args(decisions_dir=str(tmp_path / "nope"))
    code = cmd_decisions_index(args)
    assert code == 1


def test_cmd_decisions_index_writes_index_file(tmp_path: Path):
    d = tmp_path / "decisions"
    d.mkdir()
    _write_decision(d, "001-x.md", {"id": 1, "status": "accepted"}, "# Title")
    code = cmd_decisions_index(_make_args(decisions_dir=str(d)))
    assert code == 0
    assert (d / "index.md").exists()
    text = (d / "index.md").read_text()
    assert "<!-- Generated by" in text
    assert "[001](001-x.md) — Title" in text


def test_cmd_decisions_index_overwrites_existing_index(tmp_path: Path):
    d = tmp_path / "decisions"
    d.mkdir()
    (d / "index.md").write_text("OLD CONTENT")
    _write_decision(d, "001-x.md", {"id": 1, "status": "accepted"}, "# T")
    cmd_decisions_index(_make_args(decisions_dir=str(d)))
    text = (d / "index.md").read_text()
    assert "OLD CONTENT" not in text
    assert "[001]" in text


def test_cmd_decisions_index_writes_banner_only_for_empty_dir(tmp_path: Path):
    d = tmp_path / "decisions"
    d.mkdir()
    code = cmd_decisions_index(_make_args(decisions_dir=str(d)))
    assert code == 0
    text = (d / "index.md").read_text()
    assert "<!-- Generated by" in text
    assert "- [" not in text  # no entries


def test_cmd_decisions_index_resolves_decisions_dir_from_tasks_file(tmp_path: Path):
    """No --decisions-dir flag → reads defaults.decisions_dir from tasks.yaml."""
    d = tmp_path / "my-decisions"
    d.mkdir()
    _write_decision(d, "001-x.md", {"id": 1, "status": "accepted"}, "# T")

    tasks_yaml = tmp_path / "tasks.yaml"
    tasks_yaml.write_text(
        f"version: 2\n"
        f"defaults:\n"
        f"  decisions_dir: {d}\n"
        f"tasks:\n"
        f"  - id: 1\n"
        f"    title: T\n"
        f"    description: D\n"
    )

    args = _make_args(tasks=str(tasks_yaml), decisions_dir=None)
    code = cmd_decisions_index(args)
    assert code == 0
    assert (d / "index.md").exists()


def test_cmd_decisions_index_falls_back_to_hardcoded_when_no_tasks_or_flag(
    tmp_path: Path, monkeypatch,
):
    """No --decisions-dir, no tasks file → use hardcoded `docs/decisions`.
    The dir doesn't exist in the empty tmp_path → exit 1.

    Why the empty tmp_path matters: `find_tasks_file()` checks
    `cwd/tasks.yaml` and `cwd/docs/tasks.yaml`. An empty cwd means it
    returns None → handler falls through to HARDCODED_DEFAULTS
    (`docs/decisions`) → `Path("docs/decisions").exists()` is False
    (no `docs/` exists in tmp_path either) → exit 1. Future
    contributors: do NOT add tasks.yaml fixture creation to
    `_make_args` — it would silently invalidate this test by
    short-circuiting the fallback."""
    monkeypatch.chdir(tmp_path)
    args = _make_args()  # no tasks, no decisions_dir
    code = cmd_decisions_index(args)
    assert code == 1


def test_cmd_decisions_index_prints_count_summary(tmp_path: Path, capsys):
    d = tmp_path / "decisions"
    d.mkdir()
    _write_decision(d, "001-a.md", {"id": 1, "status": "accepted"}, "# A")
    _write_decision(d, "002-b.md", {"id": 2, "status": "accepted"}, "# B")
    cmd_decisions_index(_make_args(decisions_dir=str(d)))
    out = capsys.readouterr().out
    assert "Wrote 2 entries" in out


def test_cmd_decisions_index_honours_dir_flag(tmp_path: Path, monkeypatch):
    """The --dir flag chdirs before resolving relative paths."""
    project = tmp_path / "project"
    project.mkdir()
    decisions = project / "docs" / "decisions"
    decisions.mkdir(parents=True)
    _write_decision(decisions, "001-x.md",
                    {"id": 1, "status": "accepted"}, "# T")
    tasks_yaml = project / "tasks.yaml"
    tasks_yaml.write_text(
        f"version: 2\n"
        f"defaults:\n"
        f"  decisions_dir: docs/decisions\n"
        f"tasks:\n"
        f"  - id: 1\n"
        f"    title: T\n"
        f"    description: D\n"
    )
    # Run from tmp_path, chdir into project via --dir.
    monkeypatch.chdir(tmp_path)
    args = _make_args(dir=str(project))
    code = cmd_decisions_index(args)
    assert code == 0
    assert (decisions / "index.md").exists()


def test_cmd_decisions_index_warns_and_falls_back_on_malformed_tasks_yaml(
    tmp_path: Path, capsys,
):
    """Regression: when --tasks points at a malformed YAML file and
    --decisions-dir is not supplied, the handler must (a) catch the
    parse error, (b) print a yellow warning naming the path, and
    (c) fall through to the hardcoded `docs/decisions` fallback.

    Pins the `yellow(...)` import in cli.py: without it, this path
    raises NameError before printing the warning."""
    # A decisions directory must exist so we can prove the fallback
    # path actually completed. Tasks file is intentionally garbage.
    decisions = tmp_path / "docs" / "decisions"
    decisions.mkdir(parents=True)

    bad_tasks = tmp_path / "tasks.yaml"
    # Unclosed flow sequence — pyyaml raises yaml.YAMLError reliably.
    bad_tasks.write_text("defaults:\n  foo: [unterminated\n")

    # CWD must be tmp_path so the fallback's `docs/decisions` resolves
    # to the directory we just created.
    import os
    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        args = _make_args(tasks=str(bad_tasks), decisions_dir=None)
        code = cmd_decisions_index(args)
    finally:
        os.chdir(cwd)

    assert code == 0
    out = capsys.readouterr().out
    assert "Warning" in out
    assert "could not read decisions_dir" in out
    assert str(bad_tasks) in out
    assert (decisions / "index.md").exists()


def test_main_propagates_handler_exit_1_via_sys_exit(
    tmp_path: Path, monkeypatch,
):
    """End-to-end exit-code pin: the
    `claude-plan-execute decisions index` subcommand's process exit
    code must reflect cmd_decisions_index's return value.

    Validates `sys.exit(cmd_decisions_index(args))` in main() rather
    than a bare `return cmd_decisions_index(args)` (which is silently
    discarded by the root entry-point script's `main()` call). The
    handler returns 1 on missing-dir; SystemExit.code must be 1."""
    from claude_plan_execute import cli

    # Avoid picking up the repo's own tasks.yaml via find_tasks_file().
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "claude-plan-execute", "decisions", "index",
            "--decisions-dir", str(tmp_path / "definitely-missing"),
        ],
    )
    with pytest.raises(SystemExit) as exc_info:
        cli.main()
    # SystemExit's `code` is the int passed to sys.exit().
    assert exc_info.value.code == 1


def test_main_propagates_handler_exit_0_via_sys_exit(
    tmp_path: Path, monkeypatch,
):
    """Companion: success path must propagate exit 0 (not None or some
    other falsy non-int) through main()'s sys.exit(...) call."""
    from claude_plan_execute import cli

    decisions = tmp_path / "decisions"
    decisions.mkdir()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "claude-plan-execute", "decisions", "index",
            "--decisions-dir", str(decisions),
        ],
    )
    with pytest.raises(SystemExit) as exc_info:
        cli.main()
    assert exc_info.value.code == 0
    assert (decisions / "index.md").exists()
