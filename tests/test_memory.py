"""Unit tests for src/claude_plan_execute/memory.py (T15)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from claude_plan_execute.memory import (
    _drop_memory_file,
    apply_memory_curate_findings,
    count_memory_entries,
    increment_citations_for_plan,
    memory_dir_for_cwd,
    parse_memory_frontmatter,
    prune_expired_memories,
    write_memory_with_frontmatter,
)


# ---------------------------------------------------------------------------
# A. memory_dir_for_cwd path encoding
# ---------------------------------------------------------------------------


def test_memory_dir_replaces_slashes_with_dashes(tmp_path):
    fake_home = tmp_path / "home"
    out = memory_dir_for_cwd(Path("/Users/me/proj"), home=fake_home)
    assert out == fake_home / ".claude" / "projects" / "-Users-me-proj" / "memory"


def test_memory_dir_uses_actual_home_when_omitted():
    """Default home is Path.home(); ensure the call doesn't crash and
    produces a path under ~/.claude/projects."""
    out = memory_dir_for_cwd(Path("/x/y"))
    assert ".claude/projects/-x-y/memory" in str(out)


def test_memory_dir_handles_root_cwd(tmp_path):
    out = memory_dir_for_cwd(Path("/"), home=tmp_path)
    # `/`.replace('/', '-') == '-'; encoded path is '-'.
    assert out == tmp_path / ".claude" / "projects" / "-" / "memory"


def test_memory_dir_resolves_relative_input(tmp_path, monkeypatch):
    """A relative cwd is resolved against the actual cwd before
    encoding (Path.resolve())."""
    monkeypatch.chdir(tmp_path)
    out = memory_dir_for_cwd(Path("."), home=tmp_path)
    # tmp_path/.claude/projects/<encoded-tmp_path>/memory
    encoded = str(tmp_path.resolve()).replace("/", "-")
    assert out == tmp_path / ".claude" / "projects" / encoded / "memory"


# ---------------------------------------------------------------------------
# B. write_memory_with_frontmatter round-trip
# ---------------------------------------------------------------------------


def test_write_memory_round_trip(tmp_path):
    p = tmp_path / "x.md"
    write_memory_with_frontmatter(
        p,
        {"name": "x", "type": "feedback", "lifecycle": "evergreen",
         "citation_count": 3},
        "Body line 1.\nBody line 2.\n",
    )
    fm, body = parse_memory_frontmatter(p.read_text())
    assert fm["name"] == "x"
    assert fm["citation_count"] == 3
    assert "Body line 1" in body


def test_write_memory_atomic_on_failure(tmp_path, monkeypatch):
    """If os.rename fails, the original file is unchanged."""
    p = tmp_path / "x.md"
    p.write_text("ORIGINAL")

    def fail_rename(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr("claude_plan_execute.memory.os.rename", fail_rename)
    with pytest.raises(OSError):
        write_memory_with_frontmatter(
            p, {"name": "x"}, "new body",
        )
    assert p.read_text() == "ORIGINAL"


def test_write_memory_creates_parent_dir(tmp_path):
    p = tmp_path / "deep" / "nested" / "x.md"
    write_memory_with_frontmatter(
        p, {"name": "x"}, "body",
    )
    assert p.is_file()


# ---------------------------------------------------------------------------
# C. increment_citations_for_plan
# ---------------------------------------------------------------------------


def _seed_memory(d: Path, name: str, count: int = 0) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}.md"
    p.write_text(
        "---\n" +
        yaml.safe_dump({
            "name": name, "type": "feedback",
            "lifecycle": "evergreen", "citation_count": count,
        }, sort_keys=False) +
        "---\n\nBody.\n",
    )
    return p


def test_increment_single_reference(tmp_path):
    p = _seed_memory(tmp_path, "foo")
    result = increment_citations_for_plan(tmp_path, "[MEM: foo]")
    assert result == {"foo": 1}
    fm, _ = parse_memory_frontmatter(p.read_text())
    assert fm["citation_count"] == 1


def test_increment_multiple_references_count_each(tmp_path):
    p = _seed_memory(tmp_path, "foo")
    text = "[MEM: foo] [MEM: foo] [MEM: foo]"
    result = increment_citations_for_plan(tmp_path, text)
    assert result == {"foo": 3}
    fm, _ = parse_memory_frontmatter(p.read_text())
    assert fm["citation_count"] == 3


def test_increment_handles_unknown_name(tmp_path):
    """Names without a matching file resolve to 0 increments but
    appear in the result dict."""
    result = increment_citations_for_plan(tmp_path, "[MEM: ghost]")
    assert result == {"ghost": 0}


def test_increment_no_references_returns_empty(tmp_path):
    _seed_memory(tmp_path, "foo")
    assert increment_citations_for_plan(tmp_path, "no refs here") == {}


def test_increment_missing_dir_returns_zero_for_each_name(tmp_path):
    missing = tmp_path / "nope"
    assert increment_citations_for_plan(missing, "[MEM: foo]") == {"foo": 0}


def test_increment_handles_dashes_and_underscores_in_names(tmp_path):
    p = _seed_memory(tmp_path, "feedback_pydantic-v2")
    increment_citations_for_plan(
        tmp_path, "[MEM: feedback_pydantic-v2]",
    )
    fm, _ = parse_memory_frontmatter(p.read_text())
    assert fm["citation_count"] == 1


def test_increment_does_not_match_lowercase_mem_token(tmp_path):
    """Pin: `[mem: foo]` (lowercase) is NOT matched."""
    p = _seed_memory(tmp_path, "foo")
    increment_citations_for_plan(tmp_path, "[mem: foo]")
    fm, _ = parse_memory_frontmatter(p.read_text())
    assert fm["citation_count"] == 0


def test_increment_initializes_citation_count_when_missing(tmp_path):
    """A memory file whose frontmatter lacks `citation_count` gets
    initialized to 1 on first cite."""
    tmp_path.mkdir(exist_ok=True)
    p = tmp_path / "foo.md"
    # Note: NO citation_count in frontmatter.
    p.write_text(
        "---\nname: foo\ntype: feedback\nlifecycle: evergreen\n---\n\nBody.\n",
    )
    increment_citations_for_plan(tmp_path, "[MEM: foo]")
    fm, _ = parse_memory_frontmatter(p.read_text())
    assert fm["citation_count"] == 1


def test_increment_skips_malformed_frontmatter(tmp_path, capsys):
    """A file with malformed YAML frontmatter is skipped with a
    warning; callers don't double-fault."""
    tmp_path.mkdir(exist_ok=True)
    bad = tmp_path / "bad.md"
    bad.write_text("---\n: not valid yaml :::\n---\n\nBody.\n")
    # Must not raise.
    increment_citations_for_plan(tmp_path, "[MEM: bad]")


# ---------------------------------------------------------------------------
# D. prune_expired_memories
# ---------------------------------------------------------------------------


def _is_done_set(done_set: set):
    """Build an `is_task_done` callable backed by a set."""
    sset = {str(x) for x in done_set}

    def _check(t):
        return str(t) in sset
    return _check


def test_prune_evergreen_never_deleted(tmp_path):
    p = _seed_memory(tmp_path, "x")
    deleted = prune_expired_memories(
        tmp_path, is_task_done=_is_done_set(set()),
        slate_id="foo", slate_complete=True,
    )
    assert deleted == []
    assert p.exists()


def test_prune_task_scoped_when_done(tmp_path):
    p = tmp_path / "lesson.md"
    tmp_path.mkdir(exist_ok=True)
    p.write_text(
        "---\nname: lesson\ntype: feedback\nlifecycle: task:5\n"
        "citation_count: 0\n---\n\nBody.\n",
    )
    deleted = prune_expired_memories(
        tmp_path, is_task_done=_is_done_set({5}),
        slate_id="foo", slate_complete=False,
    )
    assert deleted == ["lesson"]
    assert not p.exists()


def test_prune_task_scoped_kept_when_not_done(tmp_path):
    p = tmp_path / "lesson.md"
    tmp_path.mkdir(exist_ok=True)
    p.write_text(
        "---\nname: lesson\ntype: feedback\nlifecycle: task:5\n"
        "citation_count: 0\n---\n\nBody.\n",
    )
    deleted = prune_expired_memories(
        tmp_path, is_task_done=_is_done_set(set()),
        slate_id="foo", slate_complete=False,
    )
    assert deleted == []
    assert p.exists()


def test_prune_task_scoped_string_id(tmp_path):
    """task:<N> with non-numeric N coerces to string and looks up."""
    p = tmp_path / "lesson.md"
    tmp_path.mkdir(exist_ok=True)
    p.write_text(
        "---\nname: lesson\nlifecycle: task:abc\n---\n\nBody.\n",
    )
    deleted = prune_expired_memories(
        tmp_path, is_task_done=_is_done_set({"abc"}),
        slate_id="foo", slate_complete=False,
    )
    assert deleted == ["lesson"]


def test_prune_slate_scoped_when_complete(tmp_path):
    p = tmp_path / "lesson.md"
    tmp_path.mkdir(exist_ok=True)
    p.write_text(
        "---\nname: lesson\ntype: project\nlifecycle: slate:foo\n"
        "citation_count: 0\n---\n\nBody.\n",
    )
    deleted = prune_expired_memories(
        tmp_path, is_task_done=_is_done_set(set()),
        slate_id="foo", slate_complete=True,
    )
    assert deleted == ["lesson"]


def test_prune_slate_scoped_kept_when_other_slate(tmp_path):
    """A `slate:bar` memory is never pruned during a `slate:foo` run."""
    p = tmp_path / "lesson.md"
    tmp_path.mkdir(exist_ok=True)
    p.write_text(
        "---\nname: lesson\nlifecycle: slate:bar\n---\n\nBody.\n",
    )
    deleted = prune_expired_memories(
        tmp_path, is_task_done=_is_done_set(set()),
        slate_id="foo", slate_complete=True,
    )
    assert deleted == []
    assert p.exists()


def test_prune_slate_scoped_kept_when_incomplete(tmp_path):
    p = tmp_path / "lesson.md"
    tmp_path.mkdir(exist_ok=True)
    p.write_text(
        "---\nname: lesson\nlifecycle: slate:foo\n---\n\nBody.\n",
    )
    deleted = prune_expired_memories(
        tmp_path, is_task_done=_is_done_set(set()),
        slate_id="foo", slate_complete=False,
    )
    assert deleted == []


def test_prune_unknown_lifecycle_kept(tmp_path):
    p = tmp_path / "x.md"
    tmp_path.mkdir(exist_ok=True)
    p.write_text(
        "---\nname: x\nlifecycle: bizarre:value\n---\n\nBody.\n",
    )
    deleted = prune_expired_memories(
        tmp_path, is_task_done=_is_done_set({1, 2, 3}),
        slate_id="foo", slate_complete=True,
    )
    assert deleted == []


def test_prune_missing_lifecycle_kept(tmp_path):
    p = tmp_path / "x.md"
    tmp_path.mkdir(exist_ok=True)
    p.write_text("---\nname: x\n---\n\nBody.\n")
    deleted = prune_expired_memories(
        tmp_path, is_task_done=_is_done_set({1, 2, 3}),
        slate_id="foo", slate_complete=True,
    )
    assert deleted == []


def test_prune_removes_memory_md_lines(tmp_path):
    _seed_memory(tmp_path, "keeper")
    drop = tmp_path / "drop.md"
    drop.write_text(
        "---\nname: drop\nlifecycle: task:5\n---\n\nBody.\n",
    )
    (tmp_path / "MEMORY.md").write_text(
        "- [Keeper](keeper.md) — should remain\n"
        "- [Drop me](drop.md) — should go\n"
    )
    prune_expired_memories(
        tmp_path, is_task_done=_is_done_set({5}),
        slate_id="foo", slate_complete=False,
    )
    text = (tmp_path / "MEMORY.md").read_text()
    assert "drop.md" not in text
    assert "keeper.md" in text


def test_prune_idempotent(tmp_path):
    """Running prune twice without new expirations is a no-op."""
    _seed_memory(tmp_path, "x")
    deleted_1 = prune_expired_memories(
        tmp_path, is_task_done=_is_done_set(set()),
        slate_id="foo", slate_complete=False,
    )
    deleted_2 = prune_expired_memories(
        tmp_path, is_task_done=_is_done_set(set()),
        slate_id="foo", slate_complete=False,
    )
    assert deleted_1 == [] == deleted_2


def test_prune_missing_dir_returns_empty(tmp_path):
    """A missing memory dir is a silent no-op (typical first run)."""
    missing = tmp_path / "nope"
    deleted = prune_expired_memories(
        missing, is_task_done=_is_done_set(set()),
        slate_id="foo", slate_complete=True,
    )
    assert deleted == []


def test_prune_skips_MEMORY_md_when_iterating(tmp_path):
    """`MEMORY.md` itself is never treated as a per-memory file."""
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "MEMORY.md").write_text(
        "- [X](x.md) — y\n",
    )
    _seed_memory(tmp_path, "x")
    deleted = prune_expired_memories(
        tmp_path, is_task_done=_is_done_set(set()),
        slate_id="foo", slate_complete=False,
    )
    assert deleted == []
    assert (tmp_path / "MEMORY.md").exists()


# ---------------------------------------------------------------------------
# F. count_memory_entries (T16)
# ---------------------------------------------------------------------------


def test_count_memory_entries_zero_for_missing_dir(tmp_path):
    assert count_memory_entries(tmp_path / "absent") == 0


def test_count_memory_entries_excludes_MEMORY_md(tmp_path):
    (tmp_path / "MEMORY.md").write_text("# index\n")
    assert count_memory_entries(tmp_path) == 0


def test_count_memory_entries_counts_md_files_only(tmp_path):
    (tmp_path / "MEMORY.md").write_text("# index\n")
    (tmp_path / "a.md").write_text("a\n")
    (tmp_path / "b.md").write_text("b\n")
    (tmp_path / "ignored.txt").write_text("nope\n")
    assert count_memory_entries(tmp_path) == 2


# ---------------------------------------------------------------------------
# G. _drop_memory_file (T16)
# ---------------------------------------------------------------------------


def _seed_for_drop(tmp_path: Path, name: str, body: str = "body") -> None:
    """Create a per-memory file + ensure MEMORY.md indexes it."""
    write_memory_with_frontmatter(
        tmp_path / f"{name}.md",
        {"name": name, "lifecycle": "evergreen", "citation_count": 0},
        body,
    )
    idx = tmp_path / "MEMORY.md"
    line = f"- [{name}]({name}.md) — desc\n"
    if idx.is_file():
        idx.write_text(idx.read_text() + line)
    else:
        idx.write_text(line)


def test_drop_memory_file_unlinks_and_rewrites_index(tmp_path):
    _seed_for_drop(tmp_path, "a")
    _seed_for_drop(tmp_path, "b")
    assert _drop_memory_file(tmp_path, "a") is True
    assert not (tmp_path / "a.md").exists()
    assert (tmp_path / "b.md").exists()
    idx_text = (tmp_path / "MEMORY.md").read_text()
    assert "(a.md)" not in idx_text
    assert "(b.md)" in idx_text


def test_drop_memory_file_silent_noop_on_missing_target(tmp_path):
    (tmp_path / "MEMORY.md").write_text("(other.md)\n")
    assert _drop_memory_file(tmp_path, "ghost") is False


def test_drop_memory_file_silent_noop_on_missing_dir(tmp_path):
    assert _drop_memory_file(tmp_path / "nonexistent", "anything") is False


# ---------------------------------------------------------------------------
# H. apply_memory_curate_findings (T16)
# ---------------------------------------------------------------------------


def test_apply_findings_keep_drop_merge_counts_match(tmp_path):
    _seed_for_drop(tmp_path, "k")
    _seed_for_drop(tmp_path, "d")
    _seed_for_drop(tmp_path, "m1")
    _seed_for_drop(tmp_path, "m2")
    suggestions = tmp_path / "memory-merge-suggestions.json"
    findings = [
        {"memory": "k", "verdict": "keep", "reason": "good"},
        {"memory": "d", "verdict": "drop", "reason": "stale"},
        {"memory": "m1", "verdict": "merge_into:m2", "reason": "dupe"},
    ]
    counts = apply_memory_curate_findings(
        memory_dir=tmp_path,
        findings=findings,
        suggestions_path=suggestions,
    )
    assert counts == {"kept": 1, "dropped": 1, "merge_suggested": 1, "skipped": 0}
    assert (tmp_path / "k.md").exists()
    assert not (tmp_path / "d.md").exists()
    assert (tmp_path / "m1.md").exists()
    assert (tmp_path / "m2.md").exists()
    data = json.loads(suggestions.read_text())
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["from"] == "m1"
    assert data[0]["into"] == "m2"
    assert data[0]["status"] == "pending"


def test_apply_findings_idempotent_on_re_run(tmp_path):
    _seed_for_drop(tmp_path, "m1")
    _seed_for_drop(tmp_path, "m2")
    suggestions = tmp_path / "memory-merge-suggestions.json"
    findings = [{"memory": "m1", "verdict": "merge_into:m2", "reason": "x"}]
    counts1 = apply_memory_curate_findings(
        memory_dir=tmp_path, findings=findings, suggestions_path=suggestions,
    )
    counts2 = apply_memory_curate_findings(
        memory_dir=tmp_path, findings=findings, suggestions_path=suggestions,
    )
    assert counts1["merge_suggested"] == 1
    # Second run: deterministic id hits existing_ids → continue (not counted).
    data = json.loads(suggestions.read_text())
    assert len(data) == 1


def test_apply_findings_skips_unknown_verdict_string(tmp_path):
    _seed_for_drop(tmp_path, "x")
    suggestions = tmp_path / "memory-merge-suggestions.json"
    counts = apply_memory_curate_findings(
        memory_dir=tmp_path,
        findings=[{"memory": "x", "verdict": "Keep", "reason": "case"}],
        suggestions_path=suggestions,
    )
    assert counts == {"kept": 0, "dropped": 0, "merge_suggested": 0, "skipped": 1}
    assert (tmp_path / "x.md").exists()


def test_apply_findings_handles_empty_list(tmp_path):
    suggestions = tmp_path / "memory-merge-suggestions.json"
    counts = apply_memory_curate_findings(
        memory_dir=tmp_path, findings=[], suggestions_path=suggestions,
    )
    assert counts == {"kept": 0, "dropped": 0, "merge_suggested": 0, "skipped": 0}
    assert not suggestions.exists()


def test_apply_findings_skips_non_dict_list_elements(tmp_path):
    """Pin per-element guard at HELPER layer (runner reduces to []
    before the runner ever invokes the helper, so this branch is
    not reachable through execute_task)."""
    suggestions = tmp_path / "memory-merge-suggestions.json"
    counts = apply_memory_curate_findings(
        memory_dir=tmp_path,
        findings=["drop foo", "keep bar"],
        suggestions_path=suggestions,
    )
    assert counts == {"kept": 0, "dropped": 0, "merge_suggested": 0, "skipped": 2}


def test_apply_findings_skips_finding_missing_memory_or_verdict_keys(tmp_path):
    suggestions = tmp_path / "memory-merge-suggestions.json"
    counts = apply_memory_curate_findings(
        memory_dir=tmp_path,
        findings=[{"memory": "foo"}, {"verdict": "keep"}],
        suggestions_path=suggestions,
    )
    assert counts == {"kept": 0, "dropped": 0, "merge_suggested": 0, "skipped": 2}


def test_apply_findings_skips_merge_into_with_missing_target(tmp_path):
    _seed_for_drop(tmp_path, "src")
    suggestions = tmp_path / "memory-merge-suggestions.json"
    counts = apply_memory_curate_findings(
        memory_dir=tmp_path,
        findings=[{"memory": "src", "verdict": "merge_into:nope", "reason": "x"}],
        suggestions_path=suggestions,
    )
    assert counts["skipped"] == 1
    assert counts["merge_suggested"] == 0
    assert not suggestions.exists()


def test_apply_findings_drop_silently_noop_on_missing_file(tmp_path):
    suggestions = tmp_path / "memory-merge-suggestions.json"
    counts = apply_memory_curate_findings(
        memory_dir=tmp_path,
        findings=[{"memory": "ghost", "verdict": "drop", "reason": "x"}],
        suggestions_path=suggestions,
    )
    # missing file → _drop_memory_file returns False → counts as skipped.
    assert counts["dropped"] == 0
    assert counts["skipped"] == 1
