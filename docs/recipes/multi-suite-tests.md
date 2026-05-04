# Recipe: Multiple Test Suites

## The problem

A single `test_command: pytest` can't run backend, frontend, and worker test
suites with separate output and failure semantics.

## The solution

Register each suite as a named shell gate in `invariants.yaml`:

```yaml
- name: tests-backend
  kind: shell
  command: "pytest tests/backend/ -q --tb=short"
  on_failure: block
  parallel_safe: true

- name: tests-frontend
  kind: shell
  command: "npm test --prefix frontend -- --watchAll=false"
  on_failure: block
  parallel_safe: true

- name: tests-worker
  kind: shell
  command: "pytest tests/worker/ -q --tb=short"
  on_failure: block
  parallel_safe: true
  when: [touches:src/worker/**]   # only run when worker code changes
```

## Referencing in tasks.yaml

```yaml
defaults:
  invariants_file: docs/invariants.yaml
  gates: [tests-backend, tests-frontend, tests-worker]
```

Or keep the built-in `tests` gate for legacy tasks and use `gates_extra`
for new suites on specific tasks only.

## Failure semantics

Each gate runs independently. A failing `tests-frontend` does not skip
`tests-worker`. All gate results are recorded; the task fails if any
`on_failure: block` gate fails.

## Output isolation

Each gate writes to its own file:

- `plans/task-{id}/gates/tests-backend.txt`
- `plans/task-{id}/gates/tests-frontend.txt`
- `plans/task-{id}/gates/tests-worker.txt`

## When-filter per suite

`tests-worker` in the example above runs only when the task touches
`src/worker/**`. This avoids running expensive worker tests for tasks
that don't modify worker code.
