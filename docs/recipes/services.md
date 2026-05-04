# Recipe: Services

## What the `services:` block does

The `services:` block declares external processes — dev servers, databases, message
brokers — that the runner manages for you. Services start before tasks run; the runner
polls the ready-check before handing control to Claude. They stop (or stay running)
according to their `lifetime` setting.

## Schema reference

```yaml
services:
  - name: postgres               # unique key, A-Za-z0-9_-
    command: "docker compose up -d db"
    cwd: null                    # optional; defaults to project root
    env: {}                      # extra env vars passed to the process
    ready_check:
      kind: tcp                  # "http" | "tcp" | "shell"
      port: 5432
    ready_timeout_seconds: 60    # how long to poll before failing
    lifetime: run                # "run" | "phase" | "task" | "gate"
    log_file: logs/postgres.log  # null → stdout/stderr captured internally
    depends_on: []               # other service names that must start first
```

## Ready-check kinds

| kind | field | semantics |
|------|-------|-----------|
| http | url | GET URL, accept 2xx/3xx |
| tcp | port | connect on localhost:{port} |
| shell | command | run command, accept exit 0 |

## Lifetime semantics

| lifetime | when stopped |
|----------|-------------|
| run | process exit |
| phase | after each phase completes |
| task | after each task completes |
| gate | after each gate completes |

## Full worked example

Postgres + Redis + dev-server with correct `depends_on` ordering:

```yaml
services:
  - name: postgres
    command: "docker compose up -d db"
    ready_check: { kind: tcp, port: 5432 }
    ready_timeout_seconds: 60
    lifetime: run

  - name: redis
    command: "docker compose up -d redis"
    ready_check: { kind: tcp, port: 6379 }
    ready_timeout_seconds: 30
    lifetime: run

  - name: dev-server
    command: "npm run dev"
    ready_check:
      kind: http
      url: "http://localhost:3000/health"
    ready_timeout_seconds: 120
    lifetime: run
    depends_on: [postgres, redis]   # waits for DB services first
    log_file: logs/dev-server.log
```

## Partial task activation via `services_required_extra`

When only specific tasks need a service, use `services_required_extra` at the
task level instead of declaring the service globally required:

```yaml
tasks:
  - id: 3
    title: "Integration test"
    services_required_extra: ["dev-server"]  # only this task needs it
```

## Dashboard controls

The Services panel in the dashboard shows:
- Status badge (running / stopped / failed / unstable)
- Log tail (last 50 lines)
- Restart button (`POST /api/services/<name>/restart`)

## Failure semantics

If a service fails its ready-check within `ready_timeout_seconds`, the task is marked
`blocked` with `failure_class: service_unavailable`. Fix the service and reset the task
to retry.

## Health monitoring

The runner polls services every 30 seconds. If a service crashes, it is auto-restarted.
After 3 consecutive crashes the service is marked `service_unstable`, and any tasks that
required it are blocked.
