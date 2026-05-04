# Migrating from tasks.yaml version 1 → version 2

> **Looking for the v0.2 → v0.3 guide?** See
> [docs/migration-v3.md](migration-v3.md). The v0.3 release adds new
> opt-in primitives (services, branch_per_task, human gates, structured
> findings) with no breaking changes — your existing v0.2 task files
> load and run unchanged.

## Overview

Version 2 is a strict superset of version 1. The only **required** change is bumping `version: 1` to `version: 2`. All new keys (`caps`, `phases`, `task_class_models`, etc.) are optional. Version 1 files continue to load without error — the loader preserves backward compatibility and injects sensible defaults for any keys you have not set.

> **One behavioural change to be aware of**: the default for `on_max_review_rounds` changed from `auto_approve` (v1) to `fail` (v2). If your v1 workflow relied on auto-approve when the plan review cap was reached, set `defaults.on_max_review_rounds: auto_approve` explicitly before upgrading.

---

## Quick upgrade

```diff
-version: 1
+version: 2
```

That is the minimum change. Everything else below is opt-in.

---

## Side-by-side: minimal v1 vs v2

**Version 1**

```yaml
version: 1
defaults:
  model: opus
  effort: max
  plans_dir: plans
  test_command: pytest
  max_review_rounds: 3
  plan_timeout_minutes: 20
  review_timeout_minutes: 15
  implement_timeout_minutes: 60
  implement_max_turns: 200
tasks:
  - id: 1
    title: "My first task"
    description: "Describe what to build."
  - id: 2
    title: "Depends on T1"
    description: "Builds on task 1."
    depends_on: [1]
```

**Version 2 (equivalent)**

```yaml
version: 2
defaults:
  model: opus
  effort: max
  plans_dir: plans
  test_command: pytest
  caps:
    max_review_rounds: 3
    max_minutes_per_phase: 20   # replaces plan/review/implement_timeout_minutes
    max_implement_turns: 200
tasks:
  - id: 1
    title: "My first task"
    description: "Describe what to build."
  - id: 2
    title: "Depends on T1"
    description: "Builds on task 1."
    depends_on: [1]
```

---

## Side-by-side: v1 timeout fields → v2 caps block

v1 used separate flat keys for timeouts and turn limits:

```yaml
# v1
defaults:
  plan_timeout_minutes: 20
  review_timeout_minutes: 15
  implement_timeout_minutes: 60
  implement_max_turns: 200
  max_review_rounds: 3
```

v2 consolidates these under a `caps` block:

```yaml
# v2
defaults:
  caps:
    max_minutes_per_phase: 45      # single cap applies to plan, review, and implement phases
    max_implement_turns: 200
    max_review_rounds: 3
    max_attempts: 3                # new: total retries before a task is marked failed
```

> **Note on legacy fields in v2 files**: If you set `plan_timeout_minutes`, `review_timeout_minutes`, `implement_timeout_minutes`, `max_review_rounds` (at the top-level defaults), or `implement_max_turns` in a `version: 2` file, the loader emits a deprecation warning and maps each value to its `caps.*` equivalent. v1 files using these fields load silently with no warning.

---

## Side-by-side: adding phases

Without phases (v1 style, still valid in v2):

```yaml
version: 2
tasks:
  - id: 1
    title: "Schema"
    description: "Define the data model."
  - id: 2
    title: "UI"
    description: "Build the interface."
    depends_on: [1]
```

With phases (v2):

```yaml
version: 2

phases:
  - id: foundation
    title: "Foundation"
    defaults: { task_class: schema }   # all tasks in this phase default to class "schema"
  - id: ui-polish
    title: "UI polish"
    defaults: { task_class: ui }

tasks:
  - id: 1
    phase: foundation
    title: "Schema"
    description: "Define the data model."
  - id: 2
    phase: ui-polish
    title: "UI"
    description: "Build the interface."
    depends_on: [1]
```

Tasks without a `phase:` key are unphased and inherit only from `defaults`.

---

## Side-by-side: adding task classes

`defaults.task_class_models` maps class names to model/effort pairs. Tasks (or phases) reference a class by name; the resolver substitutes the correct model and effort at runtime.

```yaml
version: 2
defaults:
  task_class_models:
    schema: { model: opus,   effort: max }
    ui:     { model: sonnet, effort: medium }
    polish: { model: haiku,  effort: medium }

phases:
  - id: foundation
    title: "Foundation"
    defaults: { task_class: schema }

tasks:
  - id: 1
    phase: foundation
    title: "Bootstrap"
    description: "First task."
  - id: 2
    phase: foundation
    title: "UI"
    description: "Second."
    task_class: polish    # explicit per-task override of the phase default
    depends_on: [1]
```

---

## New optional defaults

All keys below have no v1 equivalent and are fully opt-in.

| Key | Description |
|-----|-------------|
| `defaults.caps` | Hard limits block (`max_minutes_per_phase`, `max_review_rounds`, `max_implement_turns`, `max_attempts`) |
| `defaults.on_max_review_rounds` | `"fail"` (v2 default) or `"auto_approve"` (v1 legacy default) |
| `defaults.task_class_models` | Per-class model/effort map keyed by class name |
| `defaults.persona_file` | Path to a Markdown file injected into every prompt as persona context |
| `defaults.decisions_dir` | Directory of decision logs surfaced as context during implement |
| `defaults.invariants_file` | Path to `invariants.yaml` used by the `invariant-grep` gate |
| `defaults.gates` | List of gate names to run after each task (e.g. `["tests", "lint"]`) |
| `defaults.lint_command` | Shell command for the `lint` gate |
| `defaults.untranslated_strings_command` | Shell command for the `untranslated-strings` gate |

---

## state.json schema_version: 2

The state file gains a top-level `"schema_version": 2` key. v1 state files (no `schema_version`) self-heal on first access: `State.__init__` injects `"schema_version": 2`, `"attempts": []`, and `"current_attempt": 1` into every task entry, then writes the upgraded file on the next save. No manual migration is needed; the file upgrades in-place automatically.

See `src/claude_plan_execute/state.py` lines 95–99 for the exact self-heal code.

---

## Removed / deprecated features

**Flat timeout fields** (`plan_timeout_minutes`, `review_timeout_minutes`, `implement_timeout_minutes`, `implement_max_turns`, and top-level `max_review_rounds`) still work in both v1 and v2 files but are considered legacy. In v2 files they trigger a deprecation warning; use the `caps` block instead.

**`on_max_review_rounds` default change**: v1 files that do not set this key default to `auto_approve` (the plan is accepted automatically when the review cap is hit). v2 files default to `fail` (the task stops and is marked failed). Projects relying on auto-approve should set `defaults.on_max_review_rounds: auto_approve` explicitly before migrating to v2.
