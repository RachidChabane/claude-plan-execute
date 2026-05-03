"""Built-in gate factories and registrations.

Exports two kinds of built-ins:

  - Factory functions for built-ins WITH per-task config:
    `make_tests_gate(test_command)`, `make_lint_gate(lint_command)`,
    `make_untranslated_strings_gate(untranslated_strings_command)`,
    `make_design_fidelity_gate(*, dev_server_url, persona="")`.
    The runner calls these factories at gate-resolution time
    inside `execute_task` (task 12 / 13 / 17 / 20 §D3); the
    resulting Gates are NOT pre-registered on `default_registry()`.

  - Module-import-time registrations for built-ins WITHOUT
    per-task config: `test-critique` (task 15) and
    `security-auditor` (task 16). The Gate is constructed once
    and registered on the default singleton when this module is
    imported (transitively via `runner.py` or directly in tests).

The factory pattern and the import-time pattern coexist:
test-critique (task 15) and security-auditor (task 16) are
import-time registrations on `default_registry()`. Task 20's
`design-fidelity` ships as a FACTORY (per-project knobs:
`dev_server_url`, persona) — same shape as `make_tests_gate` etc.
The import-time count stays at two; the
`_register_all_builtin_agents(registry)` refactor threshold
("three or more import-time agent gates") has not been reached.
Likewise the shell-factory if/elif chain in the runner's Phase-4
dispatch reaches four entries (tests, lint, untranslated-strings,
design-fidelity); this is the natural threshold for a name-keyed
factory map, but with a single new entry shipping in this task
we keep the explicit form to minimise churn — a follow-up
refactor task can land the map.
"""
from __future__ import annotations

from pathlib import Path

from claude_plan_execute.gates import Gate, default_registry


# ---------------------------------------------------------------------------
# Factories for built-ins with per-task config
# ---------------------------------------------------------------------------

def make_tests_gate(test_command: str) -> Gate:
    """Construct the `tests` Gate from a resolved test_command string.

    Wraps the command in `("bash", "-c", test_command)` to preserve
    shell semantics (pipelines, `&&`, env-var substitution) that the
    pre-task-12 implement-prompt path implicitly supported. The
    runner calls this factory inside `execute_task` (task 12 §D3) when
    it sees the name "tests" in the configured gate-name list; the
    Gate is NOT pre-registered on `default_registry()` because the
    command is per-task.
    """
    return Gate(
        kind="shell",
        command=("bash", "-c", test_command),
        timeout_minutes=10,
        parallel_safe=True,
        on_failure="block",
        when=("always",),
    )


def make_lint_gate(lint_command: str) -> Gate:
    """Construct the `lint` Gate from a resolved lint_command string.

    Wraps the command in `("bash", "-c", lint_command)` to preserve
    shell semantics (pipelines, `&&`, env-var substitution) so a
    project's `lint_command: "npm run lint && npm run typecheck"`
    works without the runner pre-splitting. The runner calls this
    factory inside `execute_task` (task 13 §D3) when it sees the
    name "lint" in the configured gate-name list AND the resolved
    lint_command is non-None; the Gate is NOT pre-registered on
    `default_registry()` because the command is per-task.

    An explicit empty string produces a no-op gate (`bash -c ''`
    exits 0). The runner reserves None for the "unconfigured"
    state and skips with a warning instead of calling this
    factory; see §D2.
    """
    return Gate(
        kind="shell",
        command=("bash", "-c", lint_command),
        timeout_minutes=10,
        parallel_safe=True,
        on_failure="block",
        when=("always",),
    )


def make_untranslated_strings_gate(
    untranslated_strings_command: str,
) -> Gate:
    """Construct the `untranslated-strings` Gate from a resolved
    untranslated_strings_command string.

    The canonical place to enforce i18n discipline. The user's
    project-supplied script (or one-liner command) decides what
    "untranslated" means: missing locale entries, TODO(i18n)
    sentinels, key-extraction diff vs. the locale catalog, etc.
    The gate is intentionally schema-free — `tasks.yaml` does not
    surface i18n_files / locales / key formats; the project
    encapsulates all of that in the script.

    Wraps the command in `("bash", "-c",
    untranslated_strings_command)` to preserve shell semantics
    (pipelines, `&&`, env-var substitution, redirection) so a
    project's `untranslated_strings_command: "scripts/i18n.sh
    && python tools/check_locales.py"` works without the runner
    pre-splitting. The runner calls this factory inside
    `execute_task` (task 17 §D3) when it sees the name
    "untranslated-strings" in the configured gate-name list AND
    the resolved command is non-None; the Gate is NOT pre-
    registered on `default_registry()` because the command is
    per-task.

    An explicit empty string produces a no-op gate (`bash -c ''`
    exits 0). The runner reserves None for the "unconfigured"
    state and skips with a warning instead of calling this
    factory; see §D2.
    """
    return Gate(
        kind="shell",
        command=("bash", "-c", untranslated_strings_command),
        timeout_minutes=10,
        parallel_safe=True,
        on_failure="block",
        when=("always",),
    )


def make_design_fidelity_gate(
    *, dev_server_url: str, persona: str = "",
) -> Gate:
    """Construct the `design-fidelity` Gate from a resolved
    dev_server_url and a project persona block.

    Sonnet/medium agent gate that drives the project's running dev
    server through three viewports (320, 768, 1280) using the
    preconfigured Playwright MCP tools, and reports semantic
    design findings (no horizontal scroll, semantic structure,
    touch targets ≥ 44 CSS-px on mobile, focus-visible rings,
    contrast ratios, no overflow). See plans/task-20/plan.md §D5
    for the knob choices.

    The factory reads the shipped template at `gates/templates/
    design-fidelity.md`, substitutes the `{dev_server_url}` and
    `{persona}` placeholders with the supplied values, and
    returns a Gate whose `prompt_template` is the rendered inline
    body (NOT a path). `AgentGate._load_template` falls through
    to verbatim-return for non-path strings, so the inline body
    becomes the prompt body at gate-run time.

    `parallel_safe=False` — the gate assumes exclusive use of the
    running dev server (Playwright manipulates the shared browser
    context). `run_gates` partitions parallel_safe=False into a
    serial pool that runs first.

    `when=("task_class:ui",)` — the gate only runs on tasks whose
    effective class is `ui` (either set on the task or inherited
    from the phase). `_when_matches` (gates/__init__.py) implements
    the matcher; non-UI tasks skip with a passive PASS result.

    The runner calls this factory inside `execute_task` (task 20
    §D7) when it sees the name "design-fidelity" in the configured
    gate-name list; the Gate is NOT pre-registered on
    `default_registry()` because both inputs are per-project.
    """
    template_path = _TEMPLATES_DIR / "design-fidelity.md"
    body = template_path.read_text(encoding="utf-8")
    rendered = body.replace("{dev_server_url}", dev_server_url)
    rendered = rendered.replace("{persona}", persona)
    return Gate(
        kind="agent",
        prompt_template=rendered,
        model="sonnet",
        effort="medium",
        timeout_minutes=10,
        parallel_safe=False,
        on_failure="block",
        when=("task_class:ui",),
    )


# ---------------------------------------------------------------------------
# Built-in agent gates registered on default_registry() at import time
# ---------------------------------------------------------------------------

_TEMPLATES_DIR = Path(__file__).parent / "templates"


def _build_test_critique_gate() -> Gate:
    """Construct the `test-critique` agent Gate.

    Sonnet/medium agent gate that reads the diff and new tests,
    asking 'would these tests catch a regression in the actual
    contract?' Returns PASS/FAIL with suggested cases. Flag-only.
    See plans/task-15/plan.md §D5 for the knob choices.

    The Gate's `prompt_template` is the absolute path to the
    shipped template at `gates/templates/test-critique.md`,
    computed via `Path(__file__).parent / "templates"` so the
    gate is location-independent. (§D2.)

    Tests using the `_clean_default_registry` fixture (which
    replaces the singleton with a fresh empty registry) call
    this helper to register `test-critique` on their fresh
    registry. (§D11.)
    """
    return Gate(
        kind="agent",
        prompt_template=str(_TEMPLATES_DIR / "test-critique.md"),
        model="sonnet",
        effort="medium",
        timeout_minutes=10,
        parallel_safe=True,
        on_failure="block",
        when=("always",),
    )


def _build_security_auditor_gate() -> Gate:
    """Construct the `security-auditor` agent Gate.

    Opus/max agent gate that reads the diff and scans for OWASP-
    relevant security regressions: ACL/IDOR bypasses, missing 403
    paths, secrets, injection, unsafe deserialization, weak
    crypto, permissive CORS, sensitive-data leaks. Returns
    PASS/FAIL with itemized findings (file:line, category,
    evidence, suggested fix). Block on any finding.
    See plans/task-16/plan.md §D5 for the knob choices.

    The Gate's `prompt_template` is the absolute path to the
    shipped template at `gates/templates/security-auditor.md`,
    computed via the module-level `_TEMPLATES_DIR` constant
    (task 15 §D2) so the gate is location-independent.

    Tests using the `_clean_default_registry` fixture (which
    replaces the singleton with a fresh empty registry) call
    this helper to register `security-auditor` on their fresh
    registry. (§D12.)
    """
    return Gate(
        kind="agent",
        prompt_template=str(_TEMPLATES_DIR / "security-auditor.md"),
        model="opus",
        effort="max",
        timeout_minutes=10,
        parallel_safe=True,
        on_failure="block",
        when=("always",),
    )


# Side-effecting registration. Runs once when this module is first
# imported (via `runner.py:13` in production, via test files
# directly in tests). See §D1 for the design choice.
default_registry().register(
    "test-critique", _build_test_critique_gate(),
)
default_registry().register(
    "security-auditor", _build_security_auditor_gate(),
)


__all__ = [
    "make_design_fidelity_gate",
    "make_lint_gate",
    "make_tests_gate",
    "make_untranslated_strings_gate",
]
