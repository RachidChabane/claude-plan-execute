"""Tests for claude_plan_execute.gates.builtin.make_design_fidelity_gate
and the design-fidelity gate's end-to-end behavior at the
run_gates layer (task 20).

Runner-level integration is covered in test_runner.py Section U.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from claude_plan_execute.gates import (
    Gate,
    GateRegistry,
    default_registry,
    run_gates,
    _when_matches,
)
from claude_plan_execute.gates.agent import _extract_findings
from claude_plan_execute.gates.builtin import make_design_fidelity_gate


# ---------------------------------------------------------------------------
# A. Factory shape — the registered Gate's basic invariants
# ---------------------------------------------------------------------------

def test_make_design_fidelity_gate_returns_a_Gate():
    g = make_design_fidelity_gate(dev_server_url="http://localhost:5173")
    assert isinstance(g, Gate)


def test_make_design_fidelity_gate_kind_is_agent():
    """Pin §D5: kind=agent (browser-driving via Playwright MCP)."""
    g = make_design_fidelity_gate(dev_server_url="http://localhost:5173")
    assert g.kind == "agent"


def test_make_design_fidelity_gate_model_is_sonnet():
    """Pin §D5: model=sonnet (matches task description)."""
    g = make_design_fidelity_gate(dev_server_url="http://localhost:5173")
    assert g.model == "sonnet"


def test_make_design_fidelity_gate_effort_is_medium():
    """Pin §D5: effort=medium (matches task description)."""
    g = make_design_fidelity_gate(dev_server_url="http://localhost:5173")
    assert g.effort == "medium"


def test_make_design_fidelity_gate_on_failure_is_block():
    """Pin §D5: on_failure=block (so a finding writes a FailureRecord
    and is surfaced to the next implement attempt)."""
    g = make_design_fidelity_gate(dev_server_url="http://localhost:5173")
    assert g.on_failure == "block"


def test_make_design_fidelity_gate_parallel_safe_is_False():
    """Pin §D5: parallel_safe=False — assumes exclusive use of the
    running dev server; lands in the serial pool of run_gates."""
    g = make_design_fidelity_gate(dev_server_url="http://localhost:5173")
    assert g.parallel_safe is False


def test_make_design_fidelity_gate_when_is_task_class_ui():
    """Pin §D5: when=('task_class:ui',) — only runs on UI tasks."""
    g = make_design_fidelity_gate(dev_server_url="http://localhost:5173")
    assert g.when == ("task_class:ui",)


def test_make_design_fidelity_gate_timeout_is_10_minutes():
    """Pin §D5: timeout_minutes=10 (Gate default; explicit on the
    factory for clarity)."""
    g = make_design_fidelity_gate(dev_server_url="http://localhost:5173")
    assert g.timeout_minutes == 10


def test_make_design_fidelity_gate_returns_a_frozen_Gate():
    """Pin: the returned Gate inherits frozen-dataclass semantics."""
    g = make_design_fidelity_gate(dev_server_url="http://localhost:5173")
    with pytest.raises(Exception):
        g.timeout_minutes = 99


# ---------------------------------------------------------------------------
# B. Template substitution — {dev_server_url} and {persona} are replaced
# ---------------------------------------------------------------------------

def test_factory_substitutes_dev_server_url_in_prompt_template():
    """Pin §D4: the {dev_server_url} placeholder is replaced verbatim
    by the supplied URL."""
    g = make_design_fidelity_gate(dev_server_url="http://example.test:8080")
    assert g.prompt_template is not None
    assert "{dev_server_url}" not in g.prompt_template  # no leftover placeholder
    assert "http://example.test:8080" in g.prompt_template


def test_factory_substitutes_persona_in_prompt_template():
    """Pin §D2: the {persona} placeholder is replaced by the supplied
    persona block (verbatim, including the ## header)."""
    persona = "## Project conventions\n\nUse 8-pt vertical rhythm.\n\n"
    g = make_design_fidelity_gate(
        dev_server_url="http://localhost:5173", persona=persona,
    )
    assert "{persona}" not in g.prompt_template
    assert "Use 8-pt vertical rhythm" in g.prompt_template
    assert "Project conventions" in g.prompt_template


def test_factory_default_persona_is_empty_string():
    """Pin §D2: persona defaults to '' — a project without persona
    content still gets a renderable prompt."""
    g = make_design_fidelity_gate(dev_server_url="http://localhost:5173")
    assert "{persona}" not in g.prompt_template


def test_factory_prompt_template_is_inline_body_not_a_path():
    """Pin §D4: the rendered body is set as `prompt_template`
    (inline string), NOT a path. AgentGate._load_template falls
    through to verbatim-return for non-path strings (and to verbatim
    on OSError for over-long strings, as the rendered body always
    is)."""
    g = make_design_fidelity_gate(dev_server_url="http://localhost:5173")
    # The rendered body contains newlines, code fences, and is
    # multi-KB; on Python 3.8+ Path.is_file() raises OSError
    # (ENAMETOOLONG) for such strings. Either it raises or it
    # returns False — both are "not a file" for our purposes.
    try:
        is_file = Path(g.prompt_template).is_file()
    except OSError:
        is_file = False
    assert not is_file
    # Substantive check: the rendered body is multi-line and contains
    # the substituted URL — it is not a path string.
    assert "\n" in g.prompt_template
    assert "http://localhost:5173" in g.prompt_template


def test_factory_template_body_starts_with_design_fidelity_header():
    """Defensive: the rendered body has a recognizable header so
    accidental file corruption / wrong-file-shipped surface in tests."""
    g = make_design_fidelity_gate(dev_server_url="http://localhost:5173")
    body_lower = g.prompt_template.lower()
    assert "design-fidelity" in body_lower or "design fidelity" in body_lower


def test_factory_template_has_VERDICT_PASS_pin():
    """Pin §D12: the template contains the literal substring
    'VERDICT: PASS' (case-insensitive forms accepted) so the
    agent's pass-detection contract has the sentinel the
    template instructs the agent to emit."""
    g = make_design_fidelity_gate(dev_server_url="http://localhost:5173")
    assert (
        "VERDICT: PASS" in g.prompt_template
        or "Verdict: PASS" in g.prompt_template
    )


def test_factory_template_mentions_playwright_mcp_tool_names():
    """Pin §D3: the template names the required MCP tools so the
    agent knows what to call. Sample-check on three core tools."""
    g = make_design_fidelity_gate(dev_server_url="http://localhost:5173")
    assert "browser_navigate" in g.prompt_template
    assert "browser_resize" in g.prompt_template
    assert "browser_evaluate" in g.prompt_template


def test_factory_template_lists_three_viewports():
    """Pin §D3: the template specifies the three viewports
    (320, 768, 1280) per the task description."""
    g = make_design_fidelity_gate(dev_server_url="http://localhost:5173")
    body = g.prompt_template
    assert "320" in body
    assert "768" in body
    assert "1280" in body


def test_factory_does_NOT_substitute_diff_path_at_factory_time():
    """Pin T7 §D2 / §D6: the design-fidelity factory substitutes
    `{persona}` and `{dev_server_url}` at factory time but
    deliberately leaves `{diff_path}` as a literal placeholder.
    The substitution happens at gate-run-time in AgentGate.run().
    Regression guard against a future refactor that "helpfully"
    pre-renders all placeholders in the factory — that would
    break the diff_path substitution because the diff is per-
    task and not known at factory time."""
    g = make_design_fidelity_gate(
        dev_server_url="http://localhost:5173",
        persona="## Conventions\n\nFoo.\n",
    )
    # Factory-rendered body still contains the literal placeholder.
    assert "{diff_path}" in g.prompt_template
    # Sanity: persona and dev_server_url WERE substituted (existing
    # contract — pinned by sibling tests in this section).
    assert "{persona}" not in g.prompt_template
    assert "{dev_server_url}" not in g.prompt_template


# ---------------------------------------------------------------------------
# C. Factory does NOT register on default_registry
# ---------------------------------------------------------------------------

def test_make_design_fidelity_gate_does_NOT_register_on_default_registry():
    """Pin §D1 / §D16: the factory does NOT touch default_registry.
    Calling it has no observable side effect on the singleton's
    name list."""
    before = set(default_registry().names())
    make_design_fidelity_gate(dev_server_url="http://localhost:5173")
    make_design_fidelity_gate(dev_server_url="http://localhost:3000")
    after = set(default_registry().names())
    assert before == after


def test_design_fidelity_is_NOT_in_default_registry_after_builtin_import():
    """Pin §D1: design-fidelity is NOT registered import-time.
    The factory pattern means it's constructed per-task in the
    runner. The default registry has test-critique and security-
    auditor (import-time) but never design-fidelity."""
    import claude_plan_execute.gates.builtin  # noqa: F401
    assert default_registry().get("design-fidelity") is None


# ---------------------------------------------------------------------------
# D. when-filter behavior (task_class:ui) — exercised via _when_matches
# ---------------------------------------------------------------------------

def test_when_filter_matches_for_ui_task():
    """Pin §D8: a UI task triggers the gate via task_class:ui."""
    g = make_design_fidelity_gate(dev_server_url="http://localhost:5173")
    ok, _ = _when_matches(g.when, {"id": 1, "task_class": "ui"})
    assert ok is True


def test_when_filter_skips_for_non_ui_task():
    """Pin §D8: a non-UI task is skipped."""
    g = make_design_fidelity_gate(dev_server_url="http://localhost:5173")
    ok, reason = _when_matches(g.when, {"id": 1, "task_class": "schema"})
    assert ok is False
    assert "schema" in reason


def test_when_filter_skips_for_task_with_no_class():
    """Pin §D8: a task without a task_class is skipped (no inheritance
    happens at this layer; the runner is responsible for setting the
    effective class on the task dict before run_gates is called)."""
    g = make_design_fidelity_gate(dev_server_url="http://localhost:5173")
    ok, _ = _when_matches(g.when, {"id": 1})
    assert ok is False


# ---------------------------------------------------------------------------
# E. End-to-end at the run_gates layer (mocked AgentGate.run via run_claude stub)
# ---------------------------------------------------------------------------

def _stub_agent_run_claude(monkeypatch, *, text="VERDICT: PASS\n"):
    """Stub run_claude inside gates.agent so AgentGate.run produces a
    deterministic GateResult without spawning a subprocess. Returns a
    list of (prompt, kwargs) tuples for assertions."""
    captured: list[tuple[str, dict]] = []

    def stub(prompt, **kwargs):
        captured.append((prompt, kwargs))
        return 0, text

    monkeypatch.setattr(
        "claude_plan_execute.gates.agent.run_claude", stub,
    )
    return captured


def test_run_gates_writes_design_fidelity_output_for_ui_task(
    monkeypatch, tmp_path,
):
    """Pin task 11 §D8: a passing AgentGate run writes its output
    to plans/task-{id}/gates/design-fidelity.txt. Use a UI task so
    the when-filter matches."""
    captured = _stub_agent_run_claude(
        monkeypatch, text="VERDICT: PASS\n\nReason: all viewports clean\n",
    )
    g = make_design_fidelity_gate(dev_server_url="http://localhost:5173")
    results = run_gates(
        [("design-fidelity", g)],
        task={"id": 1, "title": "Cards", "description": "D",
              "task_class": "ui"},
        task_dir=tmp_path, cli={},
    )
    assert len(results) == 1
    assert results[0].passed is True
    out_file = tmp_path / "gates" / "design-fidelity.txt"
    assert out_file.is_file()
    body = out_file.read_text()
    assert "VERDICT: PASS" in body
    assert "all viewports clean" in body
    assert len(captured) == 1


def test_run_gates_skips_design_fidelity_for_non_ui_task(
    monkeypatch, tmp_path,
):
    """Pin §D8: when-filter skips the gate for a non-UI task; the
    AgentGate.run is NOT invoked (run_claude not called)."""
    captured = _stub_agent_run_claude(monkeypatch)
    g = make_design_fidelity_gate(dev_server_url="http://localhost:5173")
    results = run_gates(
        [("design-fidelity", g)],
        task={"id": 1, "title": "Schema migration",
              "description": "D", "task_class": "schema"},
        task_dir=tmp_path, cli={},
    )
    assert len(results) == 1
    assert results[0].passed is True
    assert "[skipped:" in results[0].output
    assert captured == []  # AgentGate.run never invoked


def test_run_gates_design_fidelity_FAIL_writes_findings_to_file(
    monkeypatch, tmp_path,
):
    """Pin task description ('Returns PASS/FAIL with itemized findings'):
    a FAIL verdict produces output_path with the findings verbatim."""
    fail_text = (
        "## Verdict: FAIL\n\n"
        "## Findings\n\n"
        "**[A] - / on 320×800**\n"
        "Risk: horizontal scroll on mobile\n"
        "Fix: tighten layout\n"
    )
    _stub_agent_run_claude(monkeypatch, text=fail_text)
    g = make_design_fidelity_gate(dev_server_url="http://localhost:5173")
    results = run_gates(
        [("design-fidelity", g)],
        task={"id": 1, "title": "Cards", "description": "D",
              "task_class": "ui"},
        task_dir=tmp_path, cli={},
    )
    assert results[0].passed is False
    body = (tmp_path / "gates" / "design-fidelity.txt").read_text()
    assert "horizontal scroll on mobile" in body


def test_run_gates_design_fidelity_serial_pool_runs_first(
    monkeypatch, tmp_path,
):
    """Pin §D5: parallel_safe=False puts design-fidelity in the
    serial pool. Verify by running it alongside a parallel-safe
    shell gate; the serial gate's stub call happens before the
    parallel one. (Coarse check via call ordering in captured.)"""
    serial_calls: list[str] = []

    def stub_serial(prompt, **kwargs):
        serial_calls.append("agent")
        return 0, "VERDICT: PASS\n"

    monkeypatch.setattr(
        "claude_plan_execute.gates.agent.run_claude", stub_serial,
    )

    def stub_shell(cmd, **kwargs):
        serial_calls.append("shell")
        class P:
            returncode = 0
            stdout = "ok\n"
            stderr = ""
        return P()

    monkeypatch.setattr(
        "claude_plan_execute.gates.shell.subprocess.run", stub_shell,
    )

    g_design = make_design_fidelity_gate(dev_server_url="http://localhost:5173")
    g_shell = Gate(kind="shell", command=("true",), parallel_safe=True)

    run_gates(
        [("design-fidelity", g_design), ("ok", g_shell)],
        task={"id": 1, "title": "T", "description": "D",
              "task_class": "ui"},
        task_dir=tmp_path, cli={},
    )
    # The serial pool runs first; design-fidelity is serial.
    assert serial_calls[0] == "agent"


def test_run_gates_design_fidelity_forwards_skip_permissions(
    monkeypatch, tmp_path,
):
    """Pin §D10 (inherited from task 15 §D9): cli['skip_permissions']
    is forwarded through to run_claude so the agent's MCP tool calls
    are permitted in autonomous mode."""
    captured = _stub_agent_run_claude(monkeypatch)
    g = make_design_fidelity_gate(dev_server_url="http://localhost:5173")
    run_gates(
        [("design-fidelity", g)],
        task={"id": 1, "title": "T", "description": "D",
              "task_class": "ui"},
        task_dir=tmp_path,
        cli={"skip_permissions": True, "verbose": True},
    )
    assert captured[0][1]["skip_permissions"] is True
    assert captured[0][1]["verbose"] is True


def test_run_gates_design_fidelity_uses_sonnet_medium(monkeypatch, tmp_path):
    """Pin §D5: the registered Gate's model/effort propagate to
    AgentGate's run_claude call."""
    captured = _stub_agent_run_claude(monkeypatch)
    g = make_design_fidelity_gate(dev_server_url="http://localhost:5173")
    run_gates(
        [("design-fidelity", g)],
        task={"id": 1, "title": "T", "description": "D",
              "task_class": "ui"},
        task_dir=tmp_path, cli={},
    )
    assert captured[0][1]["model"] == "sonnet"
    assert captured[0][1]["effort"] == "medium"


# ---------------------------------------------------------------------------
# F. Findings appendix subsection in the rendered template (T11, v0.3)
# ---------------------------------------------------------------------------
#
# Pin: the design-fidelity template gains a "### Structured findings
# appendix (recommended)" subsection. Rendered template (factory output)
# must contain the heading, schema keys, A–F cross-reference, omit
# disclaimer, and the literal HTML-comment markers. The example block
# parses cleanly via the same _extract_findings helper the build tool
# uses.
# ---------------------------------------------------------------------------

def test_design_fidelity_template_has_findings_appendix_section():
    """Pin §D / §6.5: literal `### Structured findings appendix
    (recommended)` heading present in the factory-rendered body."""
    g = make_design_fidelity_gate(dev_server_url="http://localhost:5173")
    assert "### Structured findings appendix (recommended)" in g.prompt_template


def test_design_fidelity_template_documents_per_finding_schema_keys():
    """Pin §D10: per-finding schema for design-fidelity documents
    `finding_letter`, `viewport`, `selectors`, `risk` as backticked
    field names."""
    g = make_design_fidelity_gate(dev_server_url="http://localhost:5173")
    body = g.prompt_template
    for key in ("finding_letter", "viewport", "selectors", "risk"):
        assert f"`{key}`" in body, f"missing backticked key: {key}"


def test_design_fidelity_template_finding_letter_documented_as_A_through_F():
    """Pin §D10: `finding_letter` schema cross-references the existing
    A–F category letters from the template body so aggregations bucket
    consistently. The appendix subsection must mention the letters AND
    tie at least one to its category text (e.g., "A: horizontal
    scroll")."""
    g = make_design_fidelity_gate(dev_server_url="http://localhost:5173")
    body = g.prompt_template
    # Locate the appendix subsection slice.
    section_start = body.index("### Structured findings appendix (recommended)")
    section = body[section_start:]
    # Each letter A–F is named in the cross-reference.
    for letter in ("A", "B", "C", "D", "E", "F"):
        assert f"`{letter}`" in section or letter in section, (
            f"finding letter {letter!r} missing from appendix subsection"
        )
    # At least one canonical letter→category cross-reference is present.
    assert "horizontal scroll" in section


def test_design_fidelity_template_has_html_comment_markers():
    """Pin §D2: the rendered body contains the literal HTML-comment
    markers."""
    g = make_design_fidelity_gate(dev_server_url="http://localhost:5173")
    body = g.prompt_template
    assert "<!-- BEGIN_FINDINGS_JSON -->" in body
    assert "<!-- END_FINDINGS_JSON -->" in body


def test_design_fidelity_template_has_omit_disclaimer():
    """Pin §D10: the optional-appendix disclaimer is present so the
    agent knows the appendix is additive (verdict text still
    blocks)."""
    g = make_design_fidelity_gate(dev_server_url="http://localhost:5173")
    assert (
        "emit an empty `[]` or omit the appendix entirely"
        in g.prompt_template
    )


def test_design_fidelity_template_example_is_valid_findings_appendix():
    """Pin (belt-and-suspenders): the JSON example in the template
    parses cleanly via the same _extract_findings helper the build
    tool uses. Catches typos at CI time before agents see them."""
    g = make_design_fidelity_gate(dev_server_url="http://localhost:5173")
    findings, sidecar = _extract_findings(g.prompt_template, "design-fidelity")
    assert findings is not None, "template example failed extraction"
    assert isinstance(findings, list)
    assert all(isinstance(el, dict) for el in findings)
    assert sidecar is not None
    assert "_raw" not in sidecar


def test_design_fidelity_template_does_NOT_substitute_diff_path_or_persona_in_appendix_section():
    """Pin: the appendix subsection is static across factory
    invocations. Even with a non-empty persona and a custom URL, the
    appendix block contains no `{persona}`, `{dev_server_url}`, or
    `{diff_path}` placeholders. Belt-and-suspenders against a future
    refactor that "helpfully" interpolates inside the example block."""
    g = make_design_fidelity_gate(
        dev_server_url="http://example.test:8080",
        persona="## Conventions\n\nUse 8-pt vertical rhythm.\n",
    )
    body = g.prompt_template
    section_start = body.index(
        "### Structured findings appendix (recommended)"
    )
    section = body[section_start:]
    # Persona / dev_server_url were substituted at factory time
    # OUTSIDE the appendix; the appendix itself names neither.
    assert "{persona}" not in section
    assert "{dev_server_url}" not in section
    assert "Use 8-pt vertical rhythm" not in section
    assert "http://example.test:8080" not in section
    # diff_path is left as a literal placeholder elsewhere; the
    # appendix subsection must not reference it.
    assert "{diff_path}" not in section
