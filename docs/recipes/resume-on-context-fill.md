# Recipe: Resume on Context Fill

## The problem

Claude Code has a context limit. Long implement phases may exhaust context
mid-task, leaving the working tree dirty and the task incomplete.

## The solution

The runner instructs Claude to commit WIP changes regularly and update
`plans/task-{id}/progress.md` with a checklist:

- `[x]` done
- `[~]` in progress
- `[ ]` not started

## What the runner does on resume

1. Detects `progress.md` with `[~]` or `[ ]` entries
2. Routes the implement phase to `build_resume_prompt()` instead of
   the standard `build_implement_prompt()`
3. Surfaces the current git status (dirty-tree summary) to the implementer
4. Claude picks up where it left off

## The WIP commit convention

Claude is instructed to commit as:

```bash
git commit -m "task-{id} WIP: <what was done>"
```

These commits land on the task branch (or main if `branch_per_task` is off).

## What NOT to do

- Do not manually edit `progress.md` mid-run; the runner reads it as ground
  truth for resume routing.
- Do not reset the task if you want to resume; just re-run. Reset clears
  `progress.md` and starts fresh.
- Do not worry about WIP commits polluting history with `branch_per_task`
  enabled; the final PR squash-merges everything.

## All-[x] condition

If `progress.md` has only `[x]` entries, the runner skips the implement phase
and proceeds directly to gates. This is the expected state after a successful
WIP-commit session.

## Manual override

To force a fresh implement (ignore `progress.md`), run
`--reset TASK_ID` before re-running. This deletes `progress.md` and any WIP
commits on the task branch.
