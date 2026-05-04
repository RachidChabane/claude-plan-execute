# Recipe: Database Migrations via Task Implementer

## The pattern

A task that ships a schema migration also applies it in its Implement phase,
and its test gate verifies the post-migration database. The implementer writes
the migration file, runs it against the live database, and updates application
code — all in one task.

## Why this works with services

With `services:` (T8), the database is guaranteed to be up and reachable
before the task runs. The implementer can call `alembic upgrade head` or
`psql -f migration.sql` as part of the implementation, not as a separate
ops step.

## Full example

```yaml
services:
  - name: postgres
    command: "docker compose up -d db"
    ready_check: { kind: tcp, port: 5432 }
    lifetime: run

tasks:
  - id: 7
    title: "Add users.last_login column"
    description: |
      1. Write migration: ALTER TABLE users ADD COLUMN last_login TIMESTAMP.
      2. Apply with: alembic upgrade head
      3. Update the User model in src/models.py.
      4. Add last_login to the serializer.
    key_files: [src/models.py, migrations/]
    services_required_extra: [postgres]
    commit_message: "feat: add users.last_login"
```

## The gate

The default `tests` gate runs after implement. Because the database is up
and the migration has been applied, the tests run against the post-migration
schema. No special gate configuration needed.

## Rollback

If the task fails mid-migration, the database state may be partially applied.
The task description should include a rollback step (e.g., `alembic downgrade -1`)
to restore consistency before retry.

## With branch_per_task

Each task branch runs its own migration against the shared database. If two
tasks run migrations in parallel, use `depends_on` to serialize them — don't
rely on `key_files` for database migrations (`key_files` guards file-level
contention, not database-level).
