# Recipe: Structured Findings

## What structured findings are

Agent gates (shell gates excluded) can emit a JSON array between HTML-comment
sentinels at the end of their output. The runner extracts this into a `.json`
sidecar file alongside the full text output. This enables machine-readable
cross-task analysis and aggregation (see [aggregation.md](aggregation.md)).

## The format

```html
<!-- BEGIN_FINDINGS_JSON -->
[
  {"severity": "high", "file": "src/auth.py", "line": 42, "evidence": "..."},
  {"severity": "low",  "file": "src/api.py",  "line": 7,  "evidence": "..."}
]
<!-- END_FINDINGS_JSON -->
```

The JSON must be a valid array. Each object is a "finding." Field names are
template-owned — the runner does not enforce a schema.

## Built-in templates and their schemas

| Template | Fields |
|----------|--------|
| security-auditor | category, severity, file, line, evidence, suggested_fix |
| test-critique | missing_test_for, contract, suggested_test_name |
| design-fidelity | finding_letter, viewport, selectors, risk |

## File locations

- Full response: `plans/task-{id}/gates/{name}.txt`
- Structured findings: `plans/task-{id}/gates/{name}.json`

## Adding structured findings to a custom gate

In your `invariants.yaml` `prompt_template`, instruct the agent to emit the
JSON appendix. Add this at the end of any custom gate template:

```
After your analysis, append a JSON findings block:
<!-- BEGIN_FINDINGS_JSON -->
[ {"category": "...", "severity": "high|medium|low|info", "evidence": "..."} ]
<!-- END_FINDINGS_JSON -->
Omit findings that have no actionable evidence.
```

## Schema is template-owned

The runner does NOT validate finding field names. Missing fields silently
bucket to `"(unknown)"` in aggregations. This keeps the runner gate-agnostic;
schemas evolve independently per template.

## Fail-soft behaviour

If the agent emits malformed JSON, the runner logs a warning, proceeds with
the gate verdict unchanged, and does NOT write a `.json` file. The `.txt`
file is always written.
