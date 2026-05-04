# Recipe: Human Checkpoints

## When to use

- Code review sign-off before a branch merges
- Design approval before implementation begins
- Security sign-off on sensitive changes (auth, billing, PII)

## Registering a human gate in invariants.yaml

```yaml
- name: design-review
  kind: human
  prompt_template: |
    ## Design review required

    Please review the plan at {plan_path} and approve or reject with a note.

    Approval sentinel: {approval_path}
    Rejection sentinel: {rejection_path}
  timeout_minutes: 1440         # 24h default
```

Valid placeholders: `{plan_path}`, `{gates_dir}`, `{diff_path}`,
`{approval_path}`, `{rejection_path}`. The task ID and title are injected
automatically as a fixed header above the template body — do not add
`{task_id}` or `{task_title}` to the template.

Then reference it in `defaults.gates` or per-task `gates_extra`:

```yaml
defaults:
  invariants_file: docs/invariants.yaml
  gates: [tests, design-review]
```

## Phase-end checkpoint pattern

Add a sentinel task that depends on all tasks in a phase to pause before
the next phase begins:

```yaml
tasks:
  - id: 5
    title: "Checkpoint: approve phase 1 results"
    description: "Human review before phase 2 begins."
    depends_on: [1, 2, 3, 4]
    gates_extra: [design-review]
```

## What the runner does

1. Writes a request file to `plans/task-{id}/gates/{name}.request.md`
2. Sets task status to `awaiting_human`
3. Polls every 10 seconds for approval or rejection sentinel files

## Approving / Rejecting

**Via dashboard**: navigate to the task panel; Approve/Reject buttons appear
when status is `awaiting_human`.

**Via CLI**:

```bash
claude-plan-execute approve 5 design-review --note "Looks good, proceed"
claude-plan-execute reject 5 design-review --note "Needs rework on auth flow"
```

**Via API** (for CI integrations):

```bash
curl -X POST http://localhost:8420/api/gates/5/design-review/decision \
  -H 'Content-Type: application/json' \
  -d '{"decision": "approve", "note": "LGTM"}'
```

## Timeout behaviour

After `timeout_minutes`, the task fails with `failure_class: human_gate_timeout`.
Reset and re-run to try again.

## Sentinel files (advanced / scripting)

- Approval: `plans/task-{id}/gates/{name}.approval`
- Rejection: `plans/task-{id}/gates/{name}.rejection`

File content becomes the `output` in the gate result record. You can write
these files manually in scripts or CI pipelines.
