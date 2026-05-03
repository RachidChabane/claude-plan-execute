# claude-plan-execute — Enhancement plan

## Context

`claude-plan-execute` today is a ~715-line `claude-plan-execute` script and a ~423-line
`workflow-server.py`, with an HTML dashboard. The Plan → Review → Implement loop is
generic and fine for short, low-stakes slates, but it has measurable thinness for the
Bayān workload (~50 tasks; multilingual; ACL-heavy; cross-screen invariants; mixed model
budgets). Specifically: (1) post-implement validation is one shell command; (2) prompts
have no project-specific persona; (3) resume is "skip if done" and forgets *why* a task
failed; (4) `max_review_rounds` exhaustion silently auto-approves; (5) test critique is
absent; (6) configuration resolution lacks a "phase" or "task class" axis; (7) no place
for accumulated architectural decisions; (8) the dashboard re-implements the loader and
shells out for the runner because the script-style modules can't be imported cleanly.

This plan converts those nine pain points into a coherent v2. It pushes back on three of
the proposed enhancements (i18n schema, design-fidelity gate, test critique as a phase),
folds three into one mechanism (gates), and reorders the work so the package split lands
first.

---

## 1. Executive summary

**Recommended scope cut.** Build six things in order: (1) package split, (2) configuration
overhaul (phases + task classes + audit artifact), (3) persona + decision log auto-load,
(4) failure capture and resume-with-context, (5) a single Gate Registry that subsumes the
test command, the security/invariant/lint checks, the test critique, and Playwright,
(6) hard caps + `on_max_review_rounds: fail`. Defer two things: (a) the design-fidelity
gate (real machinery; non-trivial cost; weak ROI before MVP usage), (b) auto-append of
decision-log entries (risk of agent codifying its own mistakes). Drop one thing as
proposed: a first-class `i18n` schema in `tasks.yaml` (push that into the project's
persona file; structured i18n config does not earn its weight in the runner).

**Verdict counts.** Build as proposed: 3. Build with modification: 5. Defer: 1 (design-fidelity).
Drop / replace: 1 (i18n schema → persona content; test-critique-as-phase → gate). The
package split is a yes.

**Total effort.** ~5–7 weeks of focused work, critical path ~4 weeks. The slate parallelizes
poorly because most items touch `runner.py` and `state.py`.

**Three biggest pushbacks.** (1) Don't model invariants/gates/test-critique/i18n-checks as
four separate subsystems — they are all gates with different commands and prompts; a single
registry covers them. (2) Don't ship the design-fidelity gate in v1 — it requires a Playwright
prerequisite, a screenshot pipeline, a design mockup as runtime input, and a vision-capable
specialist agent; v1 ROI is poor and the failure mode (false negatives) is bad. (3) The
current "max review rounds reached → auto-approve" path is unsafe for autonomous runs and
must flip to fail-by-default before any of the other gate work matters.

---

## 2. Per-capability sections

### 2.1 Specialized quality-gate agents → **Gate Registry** (build with modification)

**Verdict.** Build as proposed, with these refinements: collapse this capability with 2.5
(invariants), 2.7 (test critique), and the existing `test_command` into one **Gate
Registry**. Defer the design-fidelity gate.

**Rationale.** Today, post-implement validation is `test_command` interpolated into the
implement prompt — the agent decides whether the tests passed and whether to commit.
That conflates two responsibilities. Promote validation to a separate post-implement
phase that runs N gates against the produced commit, fails the task on any blocking
gate, and feeds verbatim failure output into the next attempt's prompt. Once that exists,
"E2E with Playwright," "lint+typecheck," "security audit specialist," "architectural
invariant grep," "test critique specialist" are all just registered gates.

**Design sketch.**

A `Gate` is a record:

```python
@dataclass
class Gate:
    name: str                    # "playwright", "lint", "invariant-citations-have-edition", ...
    kind: Literal["shell", "agent"]
    command: list[str] | None    # for kind=shell
    prompt_template: str | None  # for kind=agent (path to template file)
    model: str | None            # agent only; defaults to project gate model
    effort: str | None           # agent only
    timeout_minutes: int = 10
    parallel_safe: bool = True   # if False, gate gets the runner exclusively
    on_failure: Literal["block", "warn"] = "block"
    when: list[str] = field(default_factory=lambda: ["always"])
                                 # "always" | "task_class:ui" | "touches:src/auth/**" | ...
```

Gates declared in `tasks.yaml` (project default), in `phases[].gates_extra`, and
overridable per task. The runner runs all `parallel_safe=True` gates concurrently after
implement; fails the task on any `block` failure; collects all gate outputs into
`plans/task-{id}/gates/{name}.txt`.

`gates/` module layout:

```
claude_plan_execute/gates/
  __init__.py        # Gate dataclass, GateRegistry, run_gates()
  shell.py           # ShellGate.run() → exec command, capture stdout/stderr/exit
  agent.py           # AgentGate.run() → spawn claude with prompt + diff context
  builtin.py         # registers: tests, lint, invariant-grep, test-critique
```

Built-in gates shipped in v1:
- **`tests`** — runs `defaults.test_command`. The current behavior, just relocated.
- **`lint`** — runs a project-supplied lint/type command (e.g., `npm run lint && npm run typecheck`).
- **`invariant-grep`** — reads a project `invariants.yaml`, runs each `grep_assert` rule,
  fails on any violation. (This is how Bayān §10 invariants 2, 4, 6, 9 are enforced cheaply.)
- **`test-critique`** — agent gate (Sonnet, medium effort): given the diff and the new
  tests, asks "do these tests catch a regression in the actual contract?" and returns
  PASS/FAIL + critique. Fail-flag only; never auto-edits tests (see 2.7).
- **`security-auditor`** — agent gate (Opus, max effort): scans the diff for ACL bypasses,
  injection, secrets, missing 403 paths. Uses the existing `security-auditor` subagent
  prompt as the template.

**Decisions made inline.**
- *Defer Playwright gate.* The owner correctly identifies it as needed for Bayān, but the
  gate body — start a dev server, route, screenshot, test selectors, parse logs, kill —
  is half a day's work *per gate invocation*. Ship `playwright` as a built-in gate in
  v1.5 once the registry is stable. This is a phase ordering, not a refusal.
- *Defer design-fidelity gate entirely.* Requires (a) the mockup HTML or screenshot as a
  runtime input, (b) Playwright to render the implemented page, (c) screenshot
  comparison or vision-LLM judgment. The vision-LLM judge is unreliable on small
  layout drifts (the failure mode is "agent renders 90% close but wrong"; vision says
  "looks right"; nobody catches it). Ship the persona-prompt fragment with explicit
  design-spec excerpts and the lint gate; if drift becomes a problem in usage data,
  revisit.
- *Gate ordering.* `tests` first; if it passes, `lint` and `invariant-grep` and
  `test-critique` and `security-auditor` run in parallel; any block-failure fails the
  task. Tests are the cheapest signal and serialize the rest.
- *Gates have their own model budget.* Project-level `defaults.gate_model`,
  `defaults.gate_effort`. Don't let gates inherit per-task `model: opus, effort: max`
  because most gates are cheap critiques.

**Risk.** Gate output bloat — each gate writes a file per task. Cap each at 100 KB; tail
on overflow. `parallel_safe=False` exists for gates that need exclusive runner access (a
Playwright gate that boots a dev server on a fixed port) — don't ship it un-needed.

**Effort.** M (1.5–2 weeks): registry, dataclass, shell/agent runners, three built-in
gates, dashboard surface for gate outputs, tests.

---

### 2.2 Three-tier parameter resolution → **Phase + task-class resolver** (build with modification)

**Verdict.** Build, with the phase axis expressed as a `tasks.yaml` block (not a separate
file or naming convention) and an additional `task_class` axis between phase and task.
Resolved config persists per task as an audit artifact.

**Rationale.** The current resolver is already 4-tier (CLI > task > file_defaults >
hardcoded). The owner asks for project → phase → task with explicit override semantics.
Phase is a real grouping for Bayān (schema, retrieval, synthesis, cluster, UX) and the
task-class axis (2.8) is the better layer for "schema migrations want Opus/max; UI
components want Sonnet/medium." Both fit the same resolver.

**Design sketch.**

`tasks.yaml` schema becomes (additive, backward-compatible):

```yaml
version: 2
defaults:
  model: opus
  effort: max
  test_command: pytest
  persona_file: docs/persona.md          # NEW (capability 2.3)
  decisions_dir: docs/decisions          # NEW (capability 2.9)
  invariants_file: docs/invariants.yaml  # NEW (capability 2.5; optional)
  gates: [tests, lint, invariant-grep, test-critique]   # NEW
  task_class_models:                     # NEW (capability 2.8)
    schema:    { model: opus,   effort: max }
    retrieval: { model: opus,   effort: max }
    synthesis: { model: opus,   effort: max }
    ui:        { model: sonnet, effort: medium }
    polish:    { model: haiku,  effort: medium }
  on_max_review_rounds: fail             # NEW (capability 2.8)
  caps:
    max_minutes_per_phase: 60            # NEW
    max_review_rounds: 3
    max_implement_turns: 200

phases:                                  # NEW (optional; tasks without a phase use defaults)
  - id: foundation
    title: "Schema + ACL"
    defaults:
      task_class: schema
      gates_extra: [security-auditor]    # appended to project gates
  - id: retrieval
    defaults: { task_class: retrieval }
  - id: ui
    defaults: { task_class: ui }
    gates_extra: [playwright]            # if/when shipped
  - id: polish
    defaults: { task_class: polish }

tasks:
  - id: 1
    phase: foundation                    # NEW: belongs to phase
    title: ...
    description: ...
    # task-level overrides win over phase, which wins over task_class, which wins over project
    task_class: schema                   # explicit (else inherited from phase)
    model: opus                          # explicit override
```

Resolver order, lowest precedence first:

```
hardcoded → project defaults → phase defaults → task_class_models[class] → task fields → CLI flags
```

The resolver writes `plans/task-{id}/resolved-config.json` so reviewers can audit what
the agent actually saw. Includes which layer set each value.

**Decisions made inline.**
- *Phases live in `tasks.yaml`, not a separate file.* One file, one source of truth. The
  alternative (per-phase YAML files merged at load time) is overkill for typical sizes.
- *`task_class` is its own axis, not a phase alias.* A phase can mix classes (Phase F
  cluster has `cluster` retrieval + `ui` cards + `polish` filter chips). Tying class to
  phase forces tasks into the wrong bucket.
- *No legacy migration.* `version: 1` files keep working unchanged; new fields default to
  empty/None and the resolver short-circuits on missing layers. The Bayān slate writes a
  `version: 2` file from scratch.
- *Resolved-config artifact format.* JSON, with provenance per key:
  ```json
  { "model": { "value": "opus", "from": "task" },
    "effort": { "value": "max", "from": "task_class:schema" },
    "test_command": { "value": "pytest", "from": "project_defaults" } }
  ```

**Schema validation.** The current loader does ad-hoc dict reads; a typo (`task_clas:`)
silently falls through. Add Pydantic v2 models for `TasksFile`, `Phase`, `Task`,
`Caps`, `TaskClassConfig`, and (capability 2.1) `Gate`. Validation runs on load; errors
include the offending file path, line, and suggested-correction message. This also
self-documents the schema for owners and IDEs.

**Risk.** Resolution-order surprise. Mitigate with the audit artifact — if a value is
"wrong," the JSON shows which layer overrode which. Document the order in a comment
block at the top of `tasks.yaml` examples.

**Effort.** S (3 days). Mostly mechanical — extend resolver, add Pydantic models, write
audit JSON, add fields.

---

### 2.3 Coding standards as agent persona → **Persona prompt fragment** (build as proposed)

**Verdict.** Build as proposed. Persona file injected into Plan + Review + Implement
prompts. Automated checks are gates (capability 2.1).

**Rationale.** Bare principle names ("DRY, SOLID, KISS") in a prompt are decoration —
the agent will claim compliance regardless. Bayān-specific dos and don'ts ("Arabic
chunks always `dir=rtl` and `lang=ar`"; "TanStack Query for data, never `useEffect`";
"server-side `readable_kb_ids` filter on every retrieval call"; "no plain `Citation`
without `edition_id`") materially change what the agent produces. The right form is
prose, not schema, because the agent reads markdown well and the rules don't generalize
across projects.

**Design sketch.**

Project root: `docs/persona.md` (or `defaults.persona_file`). Markdown, 200–800 lines,
sections such as:

```markdown
# Project conventions for the implementing agent

## Stack and primitives
- TanStack Router file-based; routes under `frontend/src/routes/`
- TanStack Query for all server state — never `useEffect` for data fetching
- Backend: FastAPI + SQLAlchemy 2.x async + Alembic; Pydantic v2

## Multilingual rendering rules
- French chrome, classical Arabic content
- Every Arabic-bearing element gets `dir="rtl"` and `lang="ar"`
- Font pairing: Amiri for display Arabic, Noto Naskh for body Arabic, Inter/Fraunces for Latin
- Arabic body renders ~10% smaller than Latin (`font-size: 0.9em` on the Arabic block)
- Mixing scripts inline in a single LTR `<p>` is forbidden — wrap each script in its own span
- See `docs/invariants.yaml` for what's machine-checked

## Citations + ACL contract
- A retrieval result without `edition_id` is a bug; the typed model rejects it
- Server-side ACL filter (`readable_kb_ids`) runs on every chunk read
- ...

## Code conventions
- No half-finished implementations or TODO stubs
- Bug fixes don't ship surrounding cleanup; refactors are their own task
- Comments only for non-obvious WHY; never narrate WHAT
- ...

## Anti-patterns
- Never inject `dir="rtl"` into a `<p>` that mixes Latin and Arabic
- Never use `useEffect` for data; never bypass `readable_kb_ids`
- ...
```

`prompts.py` composes prompts as:

```
[persona.md if present]
[decision log if present — capability 2.9]
[failure record if resuming after a failure — capability 2.6]
[per-phase persona snippet, if any]

You are planning the implementation of a task in this codebase.
...
```

The persona file is read on every prompt build; no caching needed.

**Decisions made inline.**
- *No schema for persona.* Markdown only. Schema would add per-project transformation
  effort and the agent reads markdown natively.
- *Per-phase persona overlay is allowed but optional.* `phases[].persona_extra: docs/persona-ui.md`
  lets a phase append to the global persona. Skip for v1 if it adds friction.
- *Persona file is the only place i18n lives.* See pushback on 2.4.
- *Length cap.* If `persona.md` is over 800 lines, refuse to load with a warning. Long
  personas dilute every prompt.

**Risk.** Persona drift — the file accumulates rules nobody re-reads. Mitigate with the
decision-log split (capability 2.9): one-off decisions go to `decisions/`; durable
conventions go to `persona.md`. Persona is "how we write code"; decisions are "what we
chose for problem X."

**Effort.** S (1–2 days). Read file, concatenate into prompts, propagate path through
config. No new types.

---

### 2.4 i18n configuration with locale primitives → **Replace with persona content + one optional gate** (drop schema, replace)

**Verdict.** Drop the structured `i18n` schema. Replace with: (a) i18n rules belong in
`persona.md` (capability 2.3); (b) one optional `untranslated-strings` gate that runs a
project-supplied check command.

**Rationale.** The owner's diagnosis is correct — generic i18n flags do nothing — but
the proposed fix (typed schema with `target_locales`, `content_languages`, `bidi`,
`font_stack_per_lang`, `untranslated_strings_policy`, key-extraction convention) drags
project-specific decisions into the runner's vocabulary. The runner doesn't render
anything. It dispatches prompts and runs gates. The actually-effective channel for "use
RTL on Arabic, pair Amiri with Noto Naskh, use the `t()` function not hard-coded
strings" is the persona prompt. The actually-effective check is a project-supplied grep
in a gate.

The exception is `untranslated_strings_policy`. That's a directly actionable knob:
"reject any commit that introduces a hard-coded user-visible French string instead of a
translation key." Promote it to a one-line entry under `defaults.gates`:

```yaml
defaults:
  gates: [tests, lint, invariant-grep, test-critique, untranslated-strings]
```

The `untranslated-strings` gate body is project-supplied:

```yaml
# docs/invariants.yaml
- name: untranslated-strings
  kind: shell
  command: ["./scripts/check-untranslated.sh", "--changed-since", "HEAD~1"]
  on_failure: block
```

**Decisions made inline.**
- *No `target_locales`, `content_languages`, `bidi`, `font_stack_per_lang` in YAML.*
  Each of those is a paragraph in `persona.md`.
- *`untranslated_strings_policy: warn|fail` is just `on_failure: warn|block` on the gate.*
  Reuse the existing knob.

**Risk.** A project owner who *wants* structured i18n config has to write more text. That
text is more useful than a JSON map.

**Effort.** S (1 day) — really just shipping the gate hooks; the policy is a per-project
script.

---

### 2.5 Architectural invariants as runtime gates → **Fold into Gate Registry** (build with modification)

**Verdict.** Build, but as a **subtype of gate**, not a separate subsystem. `invariants.yaml`
is optional input that compiles to gate registrations.

**Rationale.** The IA §10 lists 10 invariants. Every one of them is shaped like "this
must be true after every task" — exactly a post-implement gate. Adding "invariants" as
its own concept duplicates machinery: same registry, same parallel runner, same
fail-on-block, same artifact. The right shape is one optional file at project root that
compiles to entries in the gate registry.

**Design sketch.**

`docs/invariants.yaml` (optional; loaded by the runner if `defaults.invariants_file` is
set):

```yaml
# Bayān §10 invariants — runnable assertions
- name: invariant-citations-have-edition
  kind: shell
  command: ["bash", "-c", "! grep -RIn --include='*.py' 'Citation(' src/ | grep -v 'edition_id'"]
  on_failure: block
  comment: "§10.2 — every Citation chip carries edition_id"

- name: invariant-no-external-commentary
  kind: agent
  prompt_template: docs/invariants/no-external-commentary.md
  model: sonnet
  effort: medium
  on_failure: block
  comment: "§10.3 — synthesis paraphrasing a sharḥ not in retrieval results"

- name: invariant-acl-on-chunk-read
  kind: shell
  command: ["bash", "-c", "./scripts/check-acl-on-chunk-read.sh"]
  on_failure: block
  comment: "§10.6 — server-side readable_kb_ids filter on every chunk read"

# ...the rest of §10
```

The runner loads `invariants.yaml`, registers each entry as a Gate, and runs them in the
same parallel pass as other gates. Output goes to `plans/task-{id}/gates/<name>.txt`.

**Decisions made inline.**
- *Invariants run on every task, not just relevant ones.* Bayān §10 is "always true,
  every screen." A non-UI task that violates invariant 6 (server-side ACL) is exactly
  what we need invariants to catch. If a specific invariant is too expensive to run
  every task, scope it with `when: ["touches:src/retrieval/**"]`.
- *Three flavors per the IA §10 wording:* (a) regex/grep on the codebase
  (`kind: shell`); (b) integration-test snippet (also shell, runs pytest with a
  marker); (c) specialist-agent judgment (`kind: agent`). The Gate dataclass already
  covers all three.
- *No DSL for assertions.* The owner could ship a tiny `assert: { regex_must_match: ..., regex_must_not_match: ... }` schema, but that's a fourth subsystem. Use shell. A
  one-liner of grep does what a DSL would.

**Risk.** Agent-judged invariants are non-deterministic (the model decides). Pin model +
effort + temperature where the SDK exposes it; for v1 accept some flap and surface it.

**Effort.** S (2–3 days) on top of capability 2.1, mostly the loader and the
specialist-agent prompt template skeleton.

---

### 2.6 Failure capture and intelligent resumption → **Failure record + context-aware resume** (build as proposed)

**Verdict.** Build as proposed. This is the highest-leverage change in the slate for
autonomous unsupervised runs.

**Rationale.** The current `state.json` per task has `{status, plan_path, review_round}`
and that's it. When a task fails — review rejected after max rounds, gate red,
implementer timed out — the next attempt starts from scratch with no record of *why*.
The autonomous failure mode is "loop on the same misunderstanding," which is exactly
what failure capture prevents.

**Design sketch.**

Per-task state extended:

```python
@dataclass
class FailureRecord:
    failure_class: Literal[
        "review_rejected_after_max_rounds",
        "gate_failed",
        "implementer_timeout",
        "implementer_max_turns",
        "budget_exceeded",
        "manual_stop",
        "plan_not_written",
        "test_failed",         # subtype of gate_failed for backward compat
    ]
    phase: Literal["plan", "review", "implement", "gate"]
    timestamp: str
    reason: str                # one-paragraph human-readable
    verbatim: str              # raw stderr / review file / gate output (last 4 KB)
    artifacts: dict[str, str]  # paths to plan, review, gate outputs, partial diff
    hypothesis: str | None     # optional agent-generated guess at root cause

@dataclass
class TaskState:
    status: str
    plan_path: str | None = None
    review_round: int | None = None
    attempts: list[FailureRecord] = field(default_factory=list)
    current_attempt: int = 1
```

`state.json` schema bump (`schema_version: 2`); old (schema-1, no `schema_version` key)
states **read** with `attempts: []` and `current_attempt: 1` defaulted in. Every **write**
from this task forward emits `schema_version: 2`, so the file upgrades on the first
state save after the runner is updated. No separate migration step; the schema is
self-healing on the next save. Document the upgrade in `docs/migration-v2.md` so the
owner knows to expect the bump on first run.

**Resume behavior.** When a failed task is re-run, the runner prepends a
**failure-context block** to every prompt that phase produces:

```
[FAILURE CONTEXT — previous attempt failed in phase: implement]
Attempt 1: review_rejected_after_max_rounds
- Reason: reviewer flagged missing ACL filter on the new endpoint three rounds running
- Verbatim (last review): "The plan adds `GET /kbs/{id}/works/recent` but does not call
  `require_kb_read(kb_id, current_user)` — this is invariant 6"
- Hypothesis: implementer keeps writing the endpoint but skipping the ACL decorator
- Artifacts: plans/task-12/plan.md, plans/task-12/review-3.md

This attempt should:
1. Address the prior failure verbatim before adding new work
2. Specifically check the hypothesis above
[END FAILURE CONTEXT]
```

The resume prompt uses the failure record to instruct the agent. Hypothesis is optional
and either set by a human in advance, by the prior attempt's reviewer, or auto-generated
by a small Sonnet pass when the runner records the failure (cheap; 1 call).

**Decisions made inline.**
- *Hypothesis is a manually-set field in v1; auto-fill defers to v1.5.* The
  FailureRecord's `verbatim` (last review or last gate output) and `reason` already
  drive the resume prompt directly. An auto-generated hypothesis adds wall-clock and
  spend on every failure and risks anchoring the next attempt on a confident wrong
  guess. Keep the field; let the human (or a follow-up gate) populate it; revisit
  auto-fill once v1 usage shows it's missed.
- *Cap attempts at 3 by default.* `defaults.caps.max_attempts: 3`. A fourth attempt
  marks the task `status: blocked` and proceeds to dependent tasks unblockable;
  dependents stay pending. This is the autonomous-mode safety valve.
- *Manual edit allowed.* The user can edit `plans/task-{id}/state.json` to clear or
  amend the failure record before re-running. This matters for "I see what's wrong;
  re-run with my hint."
- *Don't put failure records into the persona or decision log.* They are per-task and
  ephemeral; mixing them into long-lived artifacts pollutes future tasks.

**Risk.** Failure-context block crowding the prompt. Cap verbatim at 4 KB; cap full
failure context at 8 KB; surface a `[truncated]` marker.

**Effort.** S (2–3 days). Schema bump + record writer + prompt-builder integration +
dashboard surface. (Auto-hypothesis Sonnet pass is deferred — see decisions above.)

---

### 2.7 Independent test review → **Gate, not phase** (build with modification)

**Verdict.** Implement as a **gate** (`test-critique`), not a separate phase. Fail-flag
only; never auto-add tests.

**Rationale.** The owner's diagnosis is right (implementer + tests can mirror the same
flawed assumption). The proposed fix — a phase between Implement and gate-run — adds a
phase to the model when the same outcome falls out of the gate registry for free.
Modeling it as a phase also implies it precedes other gates; in practice you want the
test-critique to run *after* `tests` passes (no point critiquing red tests) and in
parallel with `lint` and `invariant-grep`.

**Design sketch.** Already covered in capability 2.1 — register `test-critique` as a
built-in agent gate. Prompt template: read the diff for the implement commit, read the
new tests, ask "would these tests catch a regression in the actual contract these tests
are claiming to validate?" Return:

```
## Verdict: PASS or FAIL
## Reason
- One paragraph
## Suggested additional cases (if FAIL)
- Bullet list
```

`on_failure: block` if the verdict is FAIL. The next attempt sees the suggested cases in
its failure-context block (capability 2.6).

**Decisions made inline.**
- *No auto-add.* The owner suggested "either auto-adds them or flags." Auto-add is
  unsafe in autonomous mode — the test-critique agent has the same blind spots as the
  implementer (different blind spots, hopefully, but the same model family). Flagging
  forces the implementer to address it on retry, which is the correct loop.
- *Test-critique runs on Sonnet/medium by default.* It reads diffs and reasons; it
  doesn't need Opus.
- *`when: "tests-changed"` filter.* If a task ships zero new tests (rare; flag separately),
  the test-critique gate emits PASS with a comment rather than burning model budget on
  empty input.

**Risk.** False positives — test-critique flags adequate tests as inadequate, blocking
otherwise-good tasks. Mitigate by surfacing the critique in the dashboard so a human can
override (`--gate-override test-critique` flag) before the next attempt.

**Effort.** S (2 days) on top of capability 2.1. Mostly prompt-template tuning.

---

### 2.8 Per-task model routing and hard caps → **task_class router + caps + safer max-rounds default** (build with modification)

**Verdict.** Build, with three pieces: `task_class` mapping (folded into capability 2.2's
resolver), `defaults.caps.*` block, and **flip `on_max_review_rounds` to `fail` by
default**.

**Rationale.** All three sub-asks are right. The current "auto-approve when reviews max
out" is the loudest mis-default for autonomous runs — it silently lowers the bar exactly
when the loop is most stuck. Flipping it is a one-line change and prevents the most
common autonomous failure I've seen on long slates.

**Design sketch.**

`tasks.yaml` schema (already shown in 2.2):

```yaml
defaults:
  task_class_models:
    schema:    { model: opus,   effort: max }
    retrieval: { model: opus,   effort: max }
    synthesis: { model: opus,   effort: max }
    ui:        { model: sonnet, effort: medium }
    polish:    { model: haiku,  effort: medium }
  caps:
    max_minutes_per_phase: 60
    max_review_rounds: 3
    max_implement_turns: 200
    max_attempts: 3
  on_max_review_rounds: fail   # was: auto_approve
```

Resolver order (lowest → highest precedence): hardcoded → project → phase →
`task_class_models[class]` → task → CLI.

`max_minutes_per_phase` is a hard wall-clock; the runner kills the claude subprocess at
that mark and records `failure_class: implementer_timeout` (or planner/reviewer
equivalent). Already half-implemented via `plan_timeout_minutes` etc.; consolidate into
`caps`.

`max_attempts` is enforced in the runner loop: after N failed attempts, the task is
marked `status: blocked` and the slate continues with what it can.

**Decisions made inline.**
- *Default `task_class` is `default`* (no overrides; uses project defaults). A task
  without a class mapping just inherits.
- *No max-tokens cap.* The Claude CLI doesn't expose a hard token cap; turn count and
  wall-clock together are sufficient. The owner's `max_tokens` ask is real but doesn't
  have an enforcement surface today.
- *Flip `on_max_review_rounds` to `fail` in v2.* Document the change loudly. A project
  that wants the old behavior sets `on_max_review_rounds: auto_approve`. The default
  flip prevents the common silent-failure path.

**Risk.** Existing v1 tasks.yaml files that relied on auto-approve now fail. Mitigate by
auto-detecting `version: 1` and keeping the old default for those files (with a warning
on load).

**Effort.** S (1–2 days). Mostly schema + resolver wiring + the timeout consolidation.

---

### 2.9 Decision log with auto-loading → **`decisions/` directory + manual append for v1** (build as proposed)

**Verdict.** Build as proposed. Auto-load on every Plan prompt. Manual append for v1;
revisit auto-append in v2 if it's actually missed.

**Rationale.** Cheap and high-leverage. As tasks land, decisions accumulate (chose
Postgres GIN over Tantivy; one `Citation` table not per-citation-kind; no `Organization`
in MVP). Today these live in commit messages and the IA. Hosting them in
`docs/decisions/` and auto-injecting into Plan prompts means the agent stops
re-deciding. Free architectural changelog.

**Design sketch.**

Structure:

```
docs/decisions/
  index.md             # one-line summary per decision, hand-curated or auto-built
  001-postgres-gin-bm25.md
  002-citation-table-shape.md
  003-no-org-in-mvp.md
```

Each decision file has YAML frontmatter and a body:

```markdown
---
id: 1
date: 2026-05-02
status: accepted   # accepted | superseded | deprecated
scope: [retrieval]
supersedes: []
superseded_by: null
---

# 001 — Postgres GIN with Arabic stemmer for BM25

## Decision
Use Postgres `pg_trgm` + GIN with the Arabic stemmer extension for the BM25 layer
of the hybrid retriever. Tantivy is the fallback if Postgres falls short.

## Rationale
- Already running Postgres
- pgvector co-located avoids cross-system joins
- Arabic stemmer in Postgres is good enough for the genre coverage we need

## Implications
- Embedded queries always JOIN against `document_chunks` for BM25 + vector
- Migration adds a `tsvector` column with a trigger for normalization
```

`prompts.py` concatenates `docs/decisions/index.md` (one-line entries) into every Plan
prompt under a `## Architectural decisions on file` header. Full files are referenced by
filename so the planner can read them on demand.

**Decisions made inline.**
- *Manual append for v1.* The agent does not auto-write decisions. It writes plans,
  reviews, and code; the human writes decisions. v2 may add a "decision-proposed" gate
  that flags candidate decisions for human ratification, but not before we see whether
  it's needed.
- *`index.md` is auto-rebuildable.* A `claude-plan-execute decisions index` command
  scans the directory and rewrites `index.md` from frontmatter. Cheap.
- *Frontmatter `scope` is freeform tags.* Not enforced. Planners can use it to
  cherry-pick relevant decisions ("only inject `scope: retrieval` for this task"), but
  default behavior is "inject all."
- *Status-superseded handling.* Superseded decisions still appear in the index (with a
  strikethrough / `~~`) so the agent doesn't re-derive a deprecated approach as if it
  were novel. Full file content is not injected for superseded entries.

**Risk.** Decision sprawl — too many decisions, every Plan prompt becomes 10 KB of
context. Mitigate by keeping `index.md` to one-line entries. Full files are
opt-in-by-reference. Cap injected index at 200 entries with a warning.

**Effort.** S (2 days). File loader + index builder + prompt-builder integration.

---

## 3. Cross-cutting decisions

### 3.1 Package split (yes — do it first)

**Decision.** Split `claude-plan-execute` (script) and `workflow-server.py` into a Python
package, before any v2 features. Single-file form is fine for v1; v2 wants modules.

**Rationale.** Five concrete frictions today:
1. The server re-implements `find_tasks_file` and `load_tasks` rather than importing —
   the script-style file resists imports.
2. The server shells out to the script for runs because there's no cleanly callable
   `run_task(task_id)` function.
3. `state.json` schema lives implicitly in the script; the server reads it by hand.
4. v2 wants a `gates/` namespace, a `prompts.py` that composes from multiple sources, a
   `failure.py` for failure records — none of which fit alongside 715 lines of mixed
   concerns.
5. Tests are absent (no `tests/` directory, no test command targets the runner itself).
   Modules are testable; scripts aren't.

The split is mostly mechanical, low-risk, and pays for itself immediately when the v2
features start landing.

**Target structure.**

```
claude-plan-execute/                          # repo root
  pyproject.toml                              # NEW — entry points
  README.md
  bin/                                        # NEW — thin shims kept on PATH
    claude-plan-execute                       # → python -m claude_plan_execute
    claude-workflow-dashboard                 # → python -m claude_plan_execute.server
  src/claude_plan_execute/
    __init__.py
    __main__.py                               # entry: cli.main()
    cli.py                                    # argparse, top-level orchestration
    server.py                                 # FastAPI app (was workflow-server.py)
    runner.py                                 # execute_task, plan/review/implement loop
    state.py                                  # State, FailureRecord, TaskState (cap 2.6)
    config.py                                 # 5-tier resolver, audit JSON writer (cap 2.2/2.8)
    loader.py                                 # tasks.yaml + decisions + persona + invariants
    prompts.py                                # build_plan/review/implement/revise/critique
    claude.py                                 # subprocess wrapper, signal handling, kill
    deps.py                                   # validate_dependencies (Kahn topological sort)
    gates/
      __init__.py                             # Gate dataclass, GateRegistry, run_gates
      shell.py                                # ShellGate
      agent.py                                # AgentGate
      builtin.py                              # tests, lint, invariant-grep, test-critique, security-auditor
    dashboard/
      index.html                              # was workflow-dashboard.html
      static/                                 # if assets get extracted later
  tests/
    test_resolver.py
    test_state.py
    test_gates.py
    test_loader.py
    fixtures/
      tasks-v1.yaml
      tasks-v2.yaml
```

**Migration steps (all backward-compatible per step).**

1. **Create package skeleton.** Add `pyproject.toml`, `src/claude_plan_execute/`,
   `bin/` shims pointing at `python -m claude_plan_execute`. The existing
   `claude-plan-execute` and `workflow-server.py` files stay as they are. CI runs both.
2. **Move pure helpers.** `validate_dependencies`, `find_tasks_file`, `load_tasks_file`,
   `State`, color helpers, `resolve` move into modules. Old `claude-plan-execute` script
   imports from the package and shrinks.
3. **Move runner.** `execute_task`, `run_claude`, `_kill_child`, prompt builders move
   into `runner.py`, `claude.py`, `prompts.py`. The old script becomes a 30-line
   `cli.py` wrapper.
4. **Move server.** `workflow-server.py` becomes `claude_plan_execute/server.py` and
   imports `loader`, `state`, `runner`. Drop the re-implemented `find_tasks_file`.
5. **Replace shell-out with import.** The dashboard's `/api/run` endpoint can either
   keep shelling out (for process isolation) or call `runner.run_slate_async()` in a
   subprocess via `multiprocessing` for cleaner cancellation. v1 keeps the shell-out;
   v2 graduates to `multiprocessing` once the package interface stabilizes.
6. **Keep the old root scripts as compat shims.** `claude-plan-execute` and
   `workflow-server.py` at the repo root stay as thin one-line wrappers that delegate
   to the package, so existing symlink installs (`ln -s "$(pwd)/claude-plan-execute"
   ~/.local/bin/...` per the current README) keep working. `bin/claude-plan-execute`
   and `bin/claude-workflow-dashboard` are the new canonical entry points (used by
   `pip install -e .`). README documents both install paths and recommends the pip
   path going forward. No task deletes the root scripts; deletion is a v1.5 cleanup
   when symlink users have migrated.

**CLI surface preservation.** Bit-for-bit. Every flag in `argparse` carries over. The
dashboard's POST/GET endpoints carry over. The `tasks.yaml` v1 schema carries over.

**Risk.** Symlink users (the README install instructions are
`ln -s "$(pwd)/claude-plan-execute" ~/.local/bin/claude-plan-execute`) get a broken
symlink at step 6. Document the `pip install -e .` migration path in the README.

**Effort.** M (1 week). Mechanical moves + new tests + README update. Low risk if done
in the order above.

### 3.2 How gates plug in

The Gate Registry (capability 2.1) is the single execution surface for capabilities 2.1,
2.5, 2.7, and parts of 2.4. It runs after Implement, before `state.set_status(tid, "done")`.

```python
def execute_task(...):
    plan_phase()
    review_phase()
    implement_phase()                             # creates a commit
    gate_results = run_gates(task, registry)      # NEW
    if any(g.failed and g.on_failure == "block" for g in gate_results):
        record_failure(task, gate_results)
        return                                    # does not mark done
    state.set_status(tid, "done")
```

`run_gates` runs `parallel_safe=True` gates concurrently via `concurrent.futures` (subprocess
isolation, not threading — both shell and agent gates spawn external processes). Output
collected to `plans/task-{id}/gates/`.

### 3.3 Persona, decisions, failure context — composition order in prompts

Plan + Review + Implement prompts compose in the same order:

```
[FAILURE CONTEXT]               (capability 2.6, only on resume after failure)
[PROJECT PERSONA]               (capability 2.3, persona.md)
[ARCHITECTURAL DECISIONS]       (capability 2.9, decisions/index.md)

[PHASE-SPECIFIC INSTRUCTIONS]   (existing — task header, plan/review/implement body)
```

Failure context first because it's the most attention-demanding signal; persona before
decisions because conventions are bigger than individual choices; decisions before
phase-specific because they constrain options.

### 3.4 Configuration centralization

`config.py` exposes one function:

```python
def resolve_task_config(task: dict, file_data: dict, cli_overrides: dict) -> ResolvedConfig:
    """Returns ResolvedConfig with .model, .effort, .gates, .caps, .test_command,
       .persona_text, .decisions_text, plus a .provenance dict for audit."""
```

Everything else (runner, prompts, gates) reads `ResolvedConfig`, never the raw YAML or
CLI dicts. This is the chokepoint for capability 2.2 + 2.8 and the place where the audit
artifact gets written.

### 3.5 What we are NOT changing

- The Plan / Review / Revise / Implement loop shape. The phases stay the same; gates run
  after Implement.
- The SIGINT handler and kill semantics. These are subtle and currently correct; touching
  them is out of scope.
- The dashboard's HTML chrome. Adding panels for gates, failure records, and resolved
  config is in scope; redesigning the layout is not.
- The state.json file location (`{plans_dir}/state.json`) and atomic-write semantics.
- The auto-detect-tasks.yaml fallback (`./tasks.yaml`, `./docs/tasks.yaml`).

---

## 4. Build order

Critical-path dependency-ordered list. Each task slug is a candidate `tasks.yaml` entry
(see appendix for the full stub).

1. **Package split, phase 1: skeleton + move pure helpers.** No behavior change.
2. **Package split, phase 2: move runner + claude wrapper.** Old script becomes a
   30-line wrapper.
3. **Package split, phase 3: move server.** Server imports from package.
4. **Tests for the moved modules** (resolver, state, loader, deps).
5. **Configuration overhaul** — phases + task_class + audit artifact + caps.
6. **`on_max_review_rounds: fail` flip** + version-aware default for v1 files.
7. **Persona prompt fragment loader.**
8. **Decision log loader + index builder.**
9. **Failure capture in `state.json`** — schema bump, record writer, schema-2 self-heal
   on first save.
10. **Resume-with-context** — failure-context block injection into prompts.
11. **Gate Registry foundation** — `Gate` dataclass, `ShellGate`, `AgentGate`, parallel
    runner, output capture.
12. **Built-in gate: `tests`** — relocate the existing `test_command` into a gate.
13. **Built-in gate: `lint`** — project-supplied command.
14. **Built-in gate: `invariant-grep`** — `invariants.yaml` loader; compiles to gates.
15. **Built-in gate: `test-critique`** — agent gate; Sonnet/medium; flag-only.
16. **Built-in gate: `security-auditor`** — agent gate; reuses subagent prompt.
17. **Built-in gate: `untranslated-strings`** — shell gate; project-supplied script.
18. **Dashboard surfaces:** resolved-config panel, failure-record panel, gates panel.
19. **README + migration guide.**

Items 1–4 are sequential and mostly mechanical (1 week total). Items 5–10 are strongly
ordered by dependency (state schema → resolver → prompts → failure → resume) — about 1.5
weeks. Items 11–17 parallelize once 5–10 land — about 2 weeks. Item 18 is concurrent
with 11–17. Item 19 is the closer.

**Critical path.** 1 → 2 → 3 → 5 → 9 → 10 → 11 → 12 (in stub numbering: split phases
→ resolver → audit artifact → failure record → resume context → gate registry → tests
gate). Roughly 4 weeks if sequential with reviews and implementation.

**Parallelizable.** Items 7 (persona), 8 (decisions), 13 (lint), 14 (invariant-grep),
15 (test-critique), 16 (security-auditor), 17 (untranslated-strings) are independent
once their parents land. Two streams comfortably; three would thrash.

**Defer to v1.5.**
- Playwright gate.
- Multiprocessing-based dashboard runner (replacing the shell-out).
- Auto-append decisions.
- Auto-hypothesis Sonnet pass on failure record.
- Deletion of the root-level compat shims (after symlink users migrate).

**Defer to v2 (or drop).**
- Design-fidelity gate.
- Concurrent independent-task execution (the IA slate has parallelism opportunities the
  current serial runner ignores; this is its own substantial change to the orchestrator).

---

## 5. Pushback log

In addition to the modifications called out in each capability section, here is the
explicit list of recommendations against the input as authored:

1. **Drop the typed `i18n` schema.** `target_locales`, `content_languages`, `bidi`,
   `font_stack_per_lang`, key-extraction conventions belong in `persona.md`, not
   `tasks.yaml`. The runner does not render anything; structured i18n knobs would be
   data with no consumer. Keep `untranslated_strings_policy` only, and only as a gate.
2. **Defer the design-fidelity gate.** The gate body is not a few lines: it requires
   Playwright running the implemented page, screenshot capture, design mockup as
   runtime input (the mockup is a separate Claude Design output), and a vision-LLM
   judge. Vision-LLM judges fail on small layout drifts. The persona prompt and the
   `lint` gate cover most of the value; revisit when MVP usage data justifies.
3. **Don't model invariants, gates, and test-critique as separate subsystems.** They
   are all post-implement assertions over the produced commit. One Gate Registry, three
   built-in gates, one optional `invariants.yaml`. Treating them as parallel subsystems
   triples the surface area for no gain.
4. **Don't model test-critique as a phase.** Phases imply ordering and prompt
   composition; the test-critique signal is post-implement and parallelizable. It's a
   gate.
5. **Never auto-add tests in test-critique.** The autonomous failure mode of
   "implementer + test-writer mirror the same blind spot" is not solved by "give the
   blind spot a third turn" — it's solved by surfacing the critique as a fail signal so
   the next implementation attempt addresses it explicitly.
6. **Never auto-append decisions.** The agent should not be writing the decision log.
   Decisions are the human's commitment device; auto-appending erodes that.
7. **Flip `on_max_review_rounds` to `fail` by default for v2.** "Auto-approve when
   review rounds max out" is the loudest mis-default for autonomous mode. v1 files keep
   the old behavior with a warning; v2 files default to fail.
8. **Don't add concurrent-task execution in this slate.** Real, but a separate piece of
   work. The current serial runner is fine for the Bayān MVP. Independent-task
   parallelism is a v2 follow-up; it interacts with state, signal handling, and gate
   resource isolation in ways the v2-features-on-top-of-current-runner work does not.
9. **Don't introduce a phase concept that requires explicit tagging.** Phases are a
   `tasks.yaml` grouping (a `phases:` block with default overrides), not a separate
   file or naming convention. Tasks may belong to a phase or not.
10. **Don't add `max_tokens` to caps.** The Claude CLI does not expose token cap
    enforcement; turn count + wall-clock are the actual levers. Adding a config key
    that maps to nothing is misleading.
11. **Don't replace the shell-out runner in the server in v1.** The server's `/api/run`
    shells out to the CLI today. After the package split, both shell-out and direct
    import are options; keep the shell-out for v1 (process isolation, simpler
    cancellation via process group) and graduate to direct call only if the v2 features
    show a need.

---

## 6. `tasks.yaml` stub (appendix)

Realistic effort tags, model defaults, dependency order. This is paste-ready for
`/Users/rachid/dev-env/0-git/claude-plan-execute/tasks.yaml` once the package split
itself is bootstrapped (it eats its own dog food, which is also the strongest test of v2).

```yaml
version: 2
defaults:
  model: opus
  effort: max
  test_command: "pytest -q"
  plans_dir: plans
  persona_file: docs/persona.md
  decisions_dir: docs/decisions
  invariants_file: docs/invariants.yaml
  gates: [tests, lint]
  task_class_models:
    refactor:  { model: opus, effort: max }
    feature:   { model: opus, effort: max }
    polish:    { model: sonnet, effort: medium }
  caps:
    max_minutes_per_phase: 60
    max_review_rounds: 3
    max_implement_turns: 200
    max_attempts: 3
  on_max_review_rounds: fail

phases:
  - id: split
    title: "Package split"
    defaults: { task_class: refactor }
  - id: config
    title: "Configuration overhaul"
    defaults: { task_class: feature }
  - id: prompts
    title: "Persona, decisions, failure context"
    defaults: { task_class: feature }
  - id: failure
    title: "Failure capture and resume"
    defaults: { task_class: feature }
  - id: gates
    title: "Gate Registry and built-in gates"
    defaults: { task_class: feature }
  - id: dashboard
    title: "Dashboard surfaces and polish"
    defaults: { task_class: polish }

tasks:
  # ── Phase: split ─────────────────────────────────────────────────────────
  - id: 1
    phase: split
    title: "Bootstrap package skeleton"
    description: |
      Create `pyproject.toml`, `src/claude_plan_execute/`, `bin/` shims. Move pure
      helpers (color, status_color, validate_dependencies, find_tasks_file,
      load_tasks_file, resolve, State) into modules with no behavior change. The
      existing `claude-plan-execute` script imports from the package and shrinks but
      remains executable. The `workflow-server.py` script unchanged in this task.
      Add `pip install -e .` instructions to README. Tests added: `test_resolver.py`,
      `test_state.py` covering the moved code.
    key_files:
      - claude-plan-execute
      - pyproject.toml
      - src/claude_plan_execute/__init__.py
      - src/claude_plan_execute/state.py
      - src/claude_plan_execute/config.py
      - src/claude_plan_execute/loader.py
      - src/claude_plan_execute/deps.py
      - tests/test_resolver.py
      - tests/test_state.py
    commit_message: "refactor: bootstrap claude_plan_execute package"
    depends_on: []

  - id: 2
    phase: split
    title: "Move runner + claude subprocess wrapper into modules"
    description: |
      Move execute_task, run_claude, _kill_child, _active_child, signal handling, and
      the prompt builders (build_plan/review/revise/implement_prompt) into
      `runner.py`, `claude.py`, `prompts.py`. The original `claude-plan-execute`
      becomes a 30-line wrapper that calls `cli.main()`. Tests: `test_runner.py`
      with subprocess fixtures.
    key_files:
      - claude-plan-execute
      - src/claude_plan_execute/runner.py
      - src/claude_plan_execute/claude.py
      - src/claude_plan_execute/prompts.py
      - src/claude_plan_execute/cli.py
      - tests/test_runner.py
    commit_message: "refactor: move runner and claude wrapper into modules"
    depends_on: [1]

  - id: 3
    phase: split
    title: "Move server into package"
    description: |
      Convert `workflow-server.py` to `src/claude_plan_execute/server.py`. Drop the
      re-implemented find_tasks_file/load_tasks/load_state — import from
      claude_plan_execute.loader and claude_plan_execute.state. Move
      `workflow-dashboard.html` to `src/claude_plan_execute/dashboard/index.html`.
      The `/api/run` endpoint continues to shell out to `claude-plan-execute` for
      v1 (process isolation). Tests: `test_server.py` with FastAPI TestClient.
    key_files:
      - workflow-server.py
      - src/claude_plan_execute/server.py
      - src/claude_plan_execute/dashboard/index.html
      - bin/claude-workflow-dashboard
      - tests/test_server.py
    commit_message: "refactor: move dashboard server into package"
    depends_on: [2]

  # ── Phase: config ────────────────────────────────────────────────────────
  - id: 4
    phase: config
    title: "Add `version: 2` schema with phases, task_class, and Pydantic validation"
    description: |
      Extend the loader to parse `phases:` blocks and `task_class_models` mapping.
      Update the resolver to a 5-tier order: hardcoded → project → phase →
      task_class_models[class] → task → CLI. Backward-compatible: `version: 1` files
      load unchanged. Add `defaults.caps` block (max_minutes_per_phase,
      max_review_rounds, max_implement_turns, max_attempts). Existing
      `*_timeout_minutes` keys still work but get deprecation warning. Add Pydantic
      v2 models for TasksFile, Phase, Task, Caps, TaskClassConfig — validation
      runs on load with helpful errors (file path, key, suggested correction).
    key_files:
      - src/claude_plan_execute/config.py
      - src/claude_plan_execute/loader.py
      - src/claude_plan_execute/schema.py
      - tests/test_resolver.py
      - tests/test_schema.py
      - tests/fixtures/tasks-v2.yaml
    commit_message: "feat: phases, task_class, caps, and Pydantic schema for tasks.yaml v2"
    depends_on: [1]

  - id: 5
    phase: config
    title: "Write resolved-config.json audit artifact per task"
    description: |
      After `resolve_task_config()` runs for a task, write the result to
      `plans/task-{id}/resolved-config.json` with provenance per key. Surface the
      file in the dashboard's task panel.
    key_files:
      - src/claude_plan_execute/config.py
      - src/claude_plan_execute/runner.py
      - src/claude_plan_execute/server.py
      - src/claude_plan_execute/dashboard/index.html
    commit_message: "feat: resolved-config.json audit artifact"
    depends_on: [4]

  - id: 6
    phase: config
    title: "Flip on_max_review_rounds to fail; version-aware default"
    description: |
      `version: 2` files default `on_max_review_rounds: fail`. `version: 1` files
      keep the legacy `auto_approve` behavior with a warning at load. Surface the
      setting in the resolved-config panel.
    key_files:
      - src/claude_plan_execute/config.py
      - src/claude_plan_execute/runner.py
      - tests/test_resolver.py
    commit_message: "feat: on_max_review_rounds: fail default for v2"
    depends_on: [4]

  # ── Phase: prompts ───────────────────────────────────────────────────────
  - id: 7
    phase: prompts
    title: "Persona prompt fragment loader"
    description: |
      Read `defaults.persona_file` (default `docs/persona.md`) on prompt build.
      Concatenate into Plan, Review, Revise, Implement prompts under a
      `## Project conventions` header. Refuse to load if over 800 lines (warning
      and skip). Per-phase `phases[].persona_extra` is appended after the global
      persona.
    key_files:
      - src/claude_plan_execute/prompts.py
      - src/claude_plan_execute/loader.py
      - tests/test_prompts.py
    commit_message: "feat: persona.md auto-injection into Plan/Implement prompts"
    depends_on: [4]

  - id: 8
    phase: prompts
    title: "Decision log loader + index builder"
    description: |
      Read `decisions/*.md` files (frontmatter + body). Inject `decisions/index.md`
      into Plan prompts under `## Architectural decisions on file`. Add
      `claude-plan-execute decisions index` subcommand to rebuild the index from
      frontmatter.
    key_files:
      - src/claude_plan_execute/loader.py
      - src/claude_plan_execute/prompts.py
      - src/claude_plan_execute/cli.py
      - tests/test_decisions.py
    commit_message: "feat: decisions/ auto-load + index subcommand"
    depends_on: [4]

  # ── Phase: failure ───────────────────────────────────────────────────────
  - id: 9
    phase: failure
    title: "Failure record schema + writer (state.json v2)"
    description: |
      Bump state.json `schema_version: 2`. Add `attempts: list[FailureRecord]`,
      `current_attempt: int` to per-task entries. Write FailureRecord on review-rounds
      exhaustion (when on_max_review_rounds=fail), implement timeout, gate failure
      (capability 2.1; gate writer ships in task 12, but the FailureRecord shape
      already accounts for it), max-turns hit, manual stop. Backward compatible:
      schema-1 reads default `attempts: []` and `current_attempt: 1`. Every write
      from this task forward emits `schema_version: 2`, so the file self-heals on
      first save. `hypothesis` is a free-form string field, populated manually for
      v1; auto-fill is deferred to v1.5.
    key_files:
      - src/claude_plan_execute/state.py
      - src/claude_plan_execute/runner.py
      - tests/test_state.py
    commit_message: "feat: FailureRecord schema in state.json v2"
    depends_on: [3, 6]

  - id: 10
    phase: failure
    title: "Resume-with-failure-context prompt block"
    description: |
      `prompts.py` adds a `[FAILURE CONTEXT — previous attempt failed in phase: X]`
      block prepended to Plan/Implement prompts when `attempts` is non-empty. Cap
      verbatim at 4 KB; full block at 8 KB. The block lists prior attempts,
      verbatim reason (last review or gate output), hypothesis (if set), and
      artifact paths.
    key_files:
      - src/claude_plan_execute/prompts.py
      - src/claude_plan_execute/runner.py
      - tests/test_prompts.py
    commit_message: "feat: resume injects failure context into prompts"
    depends_on: [9]

  # ── Phase: gates ─────────────────────────────────────────────────────────
  - id: 11
    phase: gates
    title: "Gate Registry foundation"
    description: |
      Add `gates/` module with `Gate` dataclass (kind, command, prompt_template,
      model, effort, timeout_minutes, parallel_safe, on_failure, when), `ShellGate`,
      `AgentGate`, `GateRegistry`, `run_gates()` parallel runner. Gates run after
      implement; output captured to `plans/task-{id}/gates/{name}.txt`. On any
      `on_failure: block` failure, the runner records the failure and does not mark
      the task done.
    key_files:
      - src/claude_plan_execute/gates/__init__.py
      - src/claude_plan_execute/gates/shell.py
      - src/claude_plan_execute/gates/agent.py
      - src/claude_plan_execute/runner.py
      - tests/test_gates.py
    commit_message: "feat: Gate Registry and parallel gate runner"
    depends_on: [9]

  - id: 12
    phase: gates
    title: "Built-in gate: tests (relocate test_command)"
    description: |
      Move the existing `test_command` execution out of the implement prompt and
      into a `tests` gate. The implement prompt no longer instructs the agent to
      run tests — the gate runs them after implement, on the produced commit. v1
      tasks.yaml files get a default `gates: [tests]`.
    key_files:
      - src/claude_plan_execute/gates/builtin.py
      - src/claude_plan_execute/prompts.py
      - src/claude_plan_execute/runner.py
    commit_message: "feat: tests gate replaces implement-prompt test command"
    depends_on: [11]

  - id: 13
    phase: gates
    title: "Built-in gate: lint"
    description: |
      Project-supplied lint command runs as a parallel gate after implement.
      Configured via `defaults.lint_command` or per gate registration.
    key_files:
      - src/claude_plan_execute/gates/builtin.py
      - tests/test_gates.py
    commit_message: "feat: lint gate"
    depends_on: [11]

  - id: 14
    phase: gates
    title: "Built-in gate: invariant-grep + invariants.yaml loader"
    description: |
      Optional `docs/invariants.yaml` (`defaults.invariants_file`). Each entry is a
      shell or agent gate registered at load time. Bayān-style entries: grep
      assertions (regex must match / must not match) plus optional agent-judged
      invariants. Output collected per gate name.
    key_files:
      - src/claude_plan_execute/gates/builtin.py
      - src/claude_plan_execute/loader.py
      - tests/test_invariants.py
    commit_message: "feat: invariants.yaml + invariant-grep gate"
    depends_on: [11]

  - id: 15
    phase: gates
    title: "Built-in gate: test-critique (agent gate)"
    description: |
      Sonnet/medium agent gate: reads the diff and the new tests, asks "would these
      tests catch a regression in the actual contract?" Returns PASS/FAIL with
      suggested cases. Flag-only — never auto-edits. On FAIL, the failure record's
      next-attempt prompt sees the suggested cases.
    key_files:
      - src/claude_plan_execute/gates/builtin.py
      - src/claude_plan_execute/gates/templates/test-critique.md
    commit_message: "feat: test-critique agent gate"
    depends_on: [11]

  - id: 16
    phase: gates
    title: "Built-in gate: security-auditor (agent gate)"
    description: |
      Reuses the security-auditor subagent prompt. Opus/max budget. Scans the diff
      for ACL bypasses, secrets, missing 403 paths, injection patterns. Block on
      any found.
    key_files:
      - src/claude_plan_execute/gates/builtin.py
      - src/claude_plan_execute/gates/templates/security-auditor.md
    commit_message: "feat: security-auditor agent gate"
    depends_on: [11]

  - id: 17
    phase: gates
    title: "Built-in gate: untranslated-strings (project-supplied script)"
    description: |
      Generic shell-gate variant that runs a project-supplied script. Documented as
      the canonical place to enforce i18n discipline. No first-class i18n schema in
      tasks.yaml.
    key_files:
      - src/claude_plan_execute/gates/builtin.py
    commit_message: "feat: untranslated-strings gate hook"
    depends_on: [11]

  # ── Phase: dashboard ─────────────────────────────────────────────────────
  - id: 18
    phase: dashboard
    title: "Dashboard panels: resolved config, gates, failure record"
    description: |
      Three new tabs/panels in the task detail view: (1) resolved-config.json
      formatted with provenance; (2) gates list with per-gate output collapsible;
      (3) failure record with attempts timeline and hypothesis. WebSocket already
      broadcasts state; no protocol changes.
    key_files:
      - src/claude_plan_execute/dashboard/index.html
      - src/claude_plan_execute/server.py
    commit_message: "feat: dashboard panels for config, gates, failures"
    depends_on: [5, 9, 11]

  - id: 19
    phase: dashboard
    title: "README + migration guide"
    description: |
      Update README for `pip install -e .` install, `version: 2` schema,
      capability summary, link to migration notes. Add `docs/migration-v2.md`
      with side-by-side v1 → v2 examples for tasks.yaml and a note on the
      state.json schema_version: 2 self-heal on first save.
    key_files:
      - README.md
      - docs/migration-v2.md
    commit_message: "docs: README + v2 migration guide"
    depends_on: [3, 4, 6, 9, 11]
```

---

## 7. Verification (how the v2 work proves itself)

The strongest test of v2 is for `claude-plan-execute` to run its own enhancement slate
above on itself. Specifically:

1. After tasks 1–3 (package split) land, run `claude-plan-execute --list` and
   `claude-plan-execute --task 4` against this very tasks.yaml. The runner should
   self-execute its remaining tasks.
2. Tasks 4–6 introduce the v2 schema; the runner should load its own `version: 2`
   tasks.yaml without error and produce `resolved-config.json` for each subsequent
   task.
3. Tasks 9–10 (failure capture + resume context) prove themselves the first time a
   review fails three rounds: the next attempt's prompt should show the prior failure
   record verbatim. Test by deliberately seeding a failing task.
4. Tasks 11+ (gates) prove themselves via `tests` gate output appearing in
   `plans/task-N/gates/tests.txt` for each post-implement step.

In addition:

- `tests/` runs `pytest` cleanly after each module migration.
- `bin/claude-plan-execute --help` and `bin/claude-workflow-dashboard --help` produce
  identical output before and after the split (modulo the addition of the new flags).
- The `ultrareview` flow (if invoked) sees the same project structure it always has.

---

## 8. Verdict summary (for the report)

- Build as proposed: 3 (2.3 persona, 2.6 failure capture, 2.9 decision log).
- Build with modification: 5 (2.1 gate registry with collapsing, 2.2 phase + task_class
  resolver, 2.5 invariants-as-gates, 2.7 test-critique-as-gate, 2.8 caps + flipped
  on_max_review_rounds default).
- Drop / replace: 1 (2.4 typed i18n schema → persona content + one optional gate).
- Sub-item deferrals to v1.5/v2: design-fidelity gate (sub-piece of 2.1),
  auto-hypothesis Sonnet pass on failure record (sub-piece of 2.6), auto-append
  decisions (sub-piece of 2.9).
- Package split: yes, before any v2 features (cross-cutting decision 3.1).

Total effort: 5–7 weeks; critical path ~4 weeks.

Three biggest pushbacks: (1) one gate registry, not four parallel subsystems; (2) defer
design-fidelity; (3) flip `on_max_review_rounds` to `fail` by default in v2.
