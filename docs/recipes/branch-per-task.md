# Recipe: Branch Per Task

## Overview

`branch_per_task` isolates each task on its own git branch. Benefits:
- Task implementations don't interfere with each other
- Each task gets an automatic GitHub PR
- History is clean: WIP commits are squash-merged when the PR closes

## Prerequisites

- `gh` CLI installed (`brew install gh` / `apt install gh`)
- `gh auth login` completed (or `GITHUB_TOKEN` env var set)
- GitHub-hosted repository (GitLab, Forgejo, Bitbucket not supported in v0.3)

## Enabling branch_per_task

```yaml
defaults:
  branch_per_task: true
  base_branch: main             # default "main"
  task_branch_template: "task/{id}-{slug}"  # default
  auto_pr: true                 # create PR after implement
  auto_merge: false             # squash-merge when CI passes
```

## Branch naming

The slug is the task title lowercased, non-alphanumeric characters replaced with `-`,
collapsed, truncated at 30 chars. Examples:

- "Add user auth" → `task/1-add-user-auth`
- "Implement PostgreSQL schema migrations" → `task/2-implement-postgresql-schema-mi`

## Operator hygiene

- Do not push to `base_branch` while tasks are in-flight.
- Do not merge a task's PR manually before the runner sets `status: done`;
  let `auto_merge` handle it (the PR merge triggers CI; the runner does not
  wait for CI unless `auto_merge: false`).
- Stale branches from aborted runs: `git branch -d task/N-*` after reset.

## depends_on with branch_per_task

A task whose parent is not yet merged will branch from the parent's branch
(not `base_branch`), creating a dependent-branch chain. This is intentional:
the child task sees the parent's uncommitted changes.

## The --explain flag

Use `--explain TASK_ID` to see the resolved branch name and full config before
running a task:

```bash
claude-plan-execute --explain 3
# Shows: branch name, model, gates, caps
```

## Not supported in v0.3

GitLab, Forgejo, Bitbucket, SSH-only remotes. GitHub HTTPS remotes only.
