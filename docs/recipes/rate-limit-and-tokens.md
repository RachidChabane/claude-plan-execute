# Recipe: Rate Limits and Token Usage

## Rate-limit retry

The runner automatically retries on 429 / rate-limit errors with exponential
backoff.

**Trigger patterns**: `rate_limit_error`, `429`, `Anthropic API rate limit`,
`Retry-After`

**Default**: 3 retries with exponential backoff starting at 60 s
(60 s → 120 s → 240 s)

**Configuration**:

```yaml
defaults:
  caps:
    max_rate_limit_retries: 5          # set 0 to disable retry
    rate_limit_base_backoff_seconds: 30
```

After all retries are exhausted, the task fails with
`failure_class: rate_limited`.

## Reading usage.jsonl

Each row in `plans/task-{id}/usage.jsonl`:

```json
{
  "timestamp": "2026-05-04T12:00:00Z",
  "phase": "implement",
  "model": "opus",
  "input": 14820,
  "output": 3291,
  "cache_read": 11200,
  "cache_create": 1400,
  "duration_seconds": 47.3
}
```

Phases: `plan`, `review`, `revise`, `implement`, `gate:<name>`

## Summing usage across tasks

```bash
# Total output tokens for the current run
cat plans/task-*/usage.jsonl | python3 -c "
import sys, json
total = sum(json.loads(l).get('output', 0) for l in sys.stdin)
print(f'Total output tokens: {total:,}')
"
```

## Per-task summary in state.json

After a task completes, `state.json` stores a rollup under
`tasks.<id>.usage_summary`. The keys use `total_` prefixes (different
from the per-row `usage.jsonl` format):

```json
{
  "tasks": {
    "1": {
      "usage_summary": {
        "total_input_tokens": 28400,
        "total_output_tokens": 6200,
        "total_cache_read": 22000,
        "total_cache_create": 3100,
        "total_runs": 3,
        "duration_seconds": 141.9
      }
    }
  }
}
```

Note the naming difference:

| location | input key | output key | cache read key | cache create key |
|----------|-----------|------------|----------------|-----------------|
| `usage.jsonl` row | `input` | `output` | `cache_read` | `cache_create` |
| `state.json` summary | `total_input_tokens` | `total_output_tokens` | `total_cache_read` | `total_cache_create` |

## Hard caps in v0.4 (preview)

Hard token-budget caps per task are planned but not in v0.3. In the meantime,
use `max_implement_turns` as a proxy.

## Token accounting tip

`cache_read` tokens are billed at ~10% of `input` rate; `cache_create` at
~25%. Effective cost (relative units):

```
input * 1.0 + cache_read * 0.1 + cache_create * 0.25 + output * 3.0
```

Check the Anthropic pricing page for current rates.
