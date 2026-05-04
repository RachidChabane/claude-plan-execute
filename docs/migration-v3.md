# Migrating from v0.2 → v0.3

## Overview

v0.3 is a strict superset of v0.2. No `tasks.yaml` key was removed or
renamed. Every new primitive is opt-in. A v0.2 task file loads and runs
under v0.3 without modification.

The state file schema_version bumps to 3; v2 state files self-heal on
first access (see "state.json schema_version: 3" below).

---

## Backward compatibility guarantee

| Feature | Schema key | Default | Safe to omit? |
|---------|-----------|---------|---------------|
| T1 Rate-limit retry | `caps.max_rate_limit_retries` | `3` | Yes — retry active by default |
| T2 Token observability | (none) | automatic | Yes — no config needed |
| T3 touches: filter | `when:` on gate entries | (absent) | Yes — gate always runs |
| T4 Per-task gates_extra | `gates_extra` on task | `[]` | Yes — no extra gates |
| T5 --explain flag | (CLI only) | n/a | Yes — CLI flag, no task file |
| T6 WIP-commit resume | (automatic) | automatic | Yes — no config needed |
| T7 Diff precompute | (automatic) | automatic | Yes — `{diff_path}` unused |
| T8 Services | `services:` block | `[]` | Yes — no services |
| T9 Branch per task | `branch_per_task` | `false` | Yes — stays on main |
| T10 Human gate | `kind: human` in invariants.yaml | (absent) | Yes — gate not registered |
| T11 Structured findings | marker in gate template | (absent) | Yes — no JSON appendix |
| T12 Aggregation | `aggregations.yaml` | (absent) | Yes — no rollup panel |

---

## Side-by-side examples (one per new primitive)

### Rate-limit retry (T1)

v0.2 — no configuration needed; the runner crashes on 429.
v0.3 — automatic by default (3 retries). To tune:

```yaml
defaults:
  caps:
    max_rate_limit_retries: 5          # was: not supported
    rate_limit_base_backoff_seconds: 30
```

### Token observability (T2)

v0.3 — automatic; no configuration. Per-run rows written to
`plans/task-{id}/usage.jsonl`. Per-task summary stored in `state.json`
under `tasks.<id>.usage_summary`. No action required.

### touches:\<glob\> gate filter (T3)

Before (no `when:` field — gate always runs):

```yaml
- name: auth-security-check
  kind: shell
  command: "semgrep --config p/python rules/auth.yaml"
```

After (`when:` filter — gate runs only when auth code changes):

```yaml
- name: auth-security-check
  kind: shell
  command: "semgrep --config p/python rules/auth.yaml"
  when: [touches:src/auth/**]
```

### Per-task gates_extra (T4)

Adding extra gates to a single task without changing project defaults:

```yaml
# v0.2 — to add a gate you had to put it in defaults.gates for all tasks
defaults:
  gates: [tests, lint]

# v0.3 — add gates_extra at task level; they append to defaults.gates
tasks:
  - id: 3
    title: "Auth refactor"
    gates_extra: ["security-auditor"]   # only task 3 runs security-auditor
```

### --explain TASK_ID (T5)

CLI-only; no task file change. Dumps resolved model, gates,
when-evaluations, and caps for any task without running it:

```bash
claude-plan-execute --explain 3
# Output:
# Task 3: Auth refactor
# Model:  claude-opus-4-7 (effort: max)
# Gates:  tests, lint, security-auditor
#   tests           — always
#   lint            — always
#   security-auditor — always
# Caps:
#   max_implement_turns: 200
#   max_attempts: 3
#   max_rate_limit_retries: 3
```

### WIP-commit resume (T6)

No task file change. `progress.md` is written automatically by the runner.
WIP commit convention: `task-{id} WIP: <description>`.

If an implement phase is interrupted (context limit, Ctrl+C), re-running
the task detects `progress.md` with incomplete items (`[~]` or `[ ]`)
and resumes from the last incomplete step instead of starting fresh.

### Diff precompute (T7)

No task file change. `{diff_path}` is now available in all custom gate
`prompt_template` values. The runner pre-writes the diff before calling
any agent gate, so the template variable is always resolvable:

```yaml
- name: domain-flow-validator
  kind: agent
  prompt_template: |
    Review the diff at {diff_path} against docs/domain-flows.md.
```

### Services (T8)

v0.2 — no services block; operators start dependencies manually.

v0.3 — declarative `services:` list at the top level:

```yaml
services:
  - name: postgres
    command: "docker compose up -d db"
    ready_check: { kind: tcp, port: 5432 }
    lifetime: run
```

The runner starts services before tasks run and polls the ready-check.

### branch_per_task (T9)

v0.2 — all tasks commit to whatever branch the operator has checked out.

v0.3 — opt-in per-task branch isolation:

```yaml
defaults:
  branch_per_task: true
  base_branch: main
  task_branch_template: "task/{id}-{slug}"
  auto_pr: true
  auto_merge: false
```

Each task gets its own branch (`task/3-auth-refactor`); a PR is opened
automatically after implement completes.

### Human gate (T10)

v0.2 — invariants.yaml supports only `shell` and `agent` kinds.

v0.3 — `kind: human` pauses the runner until a human approves or rejects:

```yaml
# docs/invariants.yaml
- name: design-review
  kind: human
  prompt_template: |
    Please review the plan at {plan_path} and approve or reject.
  timeout_minutes: 1440
```

Approve via dashboard, CLI (`claude-plan-execute approve 5 design-review`),
or API (`POST /api/gates/5/design-review/decision`).

### Structured findings (T11)

No task file change. Agent gate templates emit a JSON appendix between
HTML-comment sentinels:

```html
<!-- BEGIN_FINDINGS_JSON -->
[{"severity": "high", "file": "src/auth.py", "line": 42, "evidence": "..."}]
<!-- END_FINDINGS_JSON -->
```

The runner extracts this into `plans/task-{id}/gates/{name}.json` alongside
the full text output in `{name}.txt`.

### Aggregation (T12)

Drop `aggregations.yaml` (or `docs/aggregations.yaml`) in the project root.
The runner auto-discovers it via `find_aggregations_file()`. No config key
required — adding `aggregations_file` to `defaults:` causes a validation
error (`extra="forbid"`).

Minimal example:

```yaml
- name: security-by-severity
  gate: security-auditor
  field: severity
```

---

## state.json schema_version: 3

The state file schema_version bumps from 2 to 3 for v0.3. New per-task
fields added in v0.3 (`branch`, `pr_url`, `usage_summary`) are absent from
existing state entries until populated by their respective operations (branch
checkout, PR creation, task completion). Callers use `.get()` — the runner
handles their absence gracefully. The `schema_version` key is bumped to 3 on
the next save.

The self-heal code (`state.py`) continues to inject `attempts: []` and
`current_attempt: 1` for v1→v2 entries that lack them. No new injection
happens for the v0.3 fields — that is intentional.

---

## Prerequisites checklist

- Python 3.11+
- Claude Code CLI
- For branch_per_task + auto_pr: `gh` CLI installed and authenticated
- For services: whatever the service needs (Docker, etc.)
