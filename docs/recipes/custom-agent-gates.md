# Recipe: Custom Agent Gates

## The mechanism

`invariants.yaml` is a YAML list of gate entries. The runner loads it at
startup via `defaults.invariants_file` and registers each entry as a named
gate. You can add custom shell and agent gates without touching the runner
source.

## Shell gate example

```yaml
- name: no-console-log
  kind: shell
  command: "! grep -r 'console.log' src/"
  comment: "No console.log in production code"
```

## Agent gate example

```yaml
- name: domain-flow-validator
  kind: agent
  model: sonnet
  effort: medium
  timeout_minutes: 10
  parallel_safe: true
  on_failure: warn           # "block" or "warn"
  prompt_template: |
    You are reviewing the implementation diff at:
    {diff_path}

    Verify that the implementation follows the domain event flow defined in
    docs/domain-flows.md. List any violations. If none, say "LGTM".
```

Valid substitution placeholders in agent gate templates: `{diff_path}`. The
task ID and title are prepended automatically as a fixed header before the
rendered template body — `{task_id}` and `{task_title}` are NOT valid
placeholders in the template.

## Registering the gate

```yaml
# tasks.yaml
defaults:
  invariants_file: docs/invariants.yaml
  gates: [tests, lint, invariant-grep]  # invariant-grep expands all entries
```

Or reference by name for selective use:

```yaml
tasks:
  - id: 3
    gates_extra: [domain-flow-validator]
```

## The invariant-grep meta-name

Listing `invariant-grep` in `defaults.gates` expands to all entries in
`invariants.yaml`. Use this when every custom gate should run on every task.

## When filter — activate only for relevant tasks

```yaml
- name: auth-security-check
  kind: shell
  command: "semgrep --config p/python rules/auth.yaml"
  when: [touches:src/auth/**]
```

The gate only runs for tasks whose `key_files` match the `src/auth/**` pathspec.

## The rename note

The mechanism is called `invariants.yaml` in the codebase, but the concept
covers both architectural invariants AND custom domain gates. Rename the file
to `docs/gates.yaml` or `docs/custom-gates.yaml` if the "invariants" framing
doesn't fit your project — just update `defaults.invariants_file`.
