# Recipe: Aggregation

## Purpose

Roll up findings across all tasks into a single count-by-key view, surfaced in
the dashboard's Aggregations panel. Use this to answer "how many high-severity
security findings does the whole codebase have?" or "which locales have the most
missing translation keys?"

## The aggregations.yaml file

```yaml
# Mode A — regex on gate .txt output (works on any gate)
- name: untranslated-by-locale
  gate: untranslated-strings
  extract: "(?im)missing key: (\\S+)"

# Mode B — JSON field from structured findings
- name: security-by-severity
  gate: security-auditor
  field: severity

# Mode B — composite key (joins multiple fields)
- name: security-by-category-and-severity
  gate: security-auditor
  fields: [category, severity]
```

## Mode A (regex)

- Applied to `gates/{name}.txt` of every task
- Capture group 1 → the finding key; if no capture group, full match is the key
- Useful for any gate: shell gates, agent gates without structured findings

## Mode B (field / fields)

- Requires `gates/{name}.json` (structured findings from T11)
- Iterates `findings[]`, extracts `finding[field]`; missing field → `"(unknown)"`
- `fields: [a, b]` → key is `"a_value:b_value"` (colon-joined, no spaces)
- Silently skips tasks with no `.json` file

## When to pick which

- **Mode A**: the gate is a shell command, output is unstructured text, or you
  don't control the gate template
- **Mode B**: the gate uses a structured-findings template and you want to pivot
  by specific fields without regex fragility

## API endpoint

```bash
GET /api/aggregate/{name}
```

Response:

```json
[
  {"finding": "high", "total_count": 7, "per_task": {"3": 2, "5": 5}},
  {"finding": "medium", "total_count": 3, "per_task": {"2": 3}}
]
```

Sorted by `total_count` descending.

## Dashboard panel

Bar list per aggregation name; hover on a bar for per-task breakdown with
sparklines.

## No registration required

Drop `aggregations.yaml` (or `docs/aggregations.yaml`) in the project root.
The runner calls `find_aggregations_file()` at startup and auto-discovers
either path. There is no `aggregations_file` config key — adding one to
`defaults:` causes a validation error (`extra="forbid"`).
