# Recipe: Cross-Task Memory

## What cross-task memory is

A per-project directory of Markdown files with YAML frontmatter, stored at:

```
~/.claude/projects/<encoded-cwd>/memory/
```

Auto-loaded by Claude Code at every session start via `MEMORY.md`. The runner
augments it with lifecycle pruning and citation tracking.

## The four prompt overlay taxonomy

| Layer | File/Dir | Curated by | Purpose |
|-------|----------|-----------|---------|
| 1 — persona | `docs/persona.md` | Human | Stable per-project conventions |
| 2 — decisions | `docs/decisions/` | Human | ADR-style architectural choices |
| 3 — memory | `~/.claude/.../memory/` | Agent + runner | Tactical pitfalls discovered in flight |
| 4 — failure-context | `plans/task-{id}/failure-N.md` | Runner | Per-attempt prior failure verbatim |

Layer 3 is the ONLY layer agents write to. Layers 1 and 2 are
human-curated. Layer 4 is fully automatic (never edit manually).

## Memory frontmatter schema

```markdown
---
name: <slug>            # file stem; used in [MEM: <name>] citations
description: <one-line> # shown in MEMORY.md index; used by runner to judge relevance
type: user|feedback|project|reference
lifecycle: evergreen | slate:<slate-id> | task:<id>
citation_count: 0
---
```

## Type-aware default lifecycles

| type | default lifecycle | rationale |
|------|------------------|-----------|
| project | `slate:<slate-id>` | Slate-specific facts expire with the slate |
| feedback | `evergreen` | Cross-project learnings should survive |
| reference | `evergreen` | Pointers to external systems don't expire |
| user | `evergreen` | User preferences are long-lived |

## The `[MEM: <name>]` citation pattern

When a Plan phase agent writes `[MEM: feedback_pydantic_v2]` in a plan,
the runner's citation pass (after plan approval) increments `citation_count`
in that memory file. The runner reads the plan text, extracts all
`[MEM: ...]` references, and calls `increment_citations_for_plan()` for
the memory directory.

Citations matter for:
- Deciding which memories are "live" vs stale (high `citation_count` →
  still relevant)
- The `memory-curate` gate (T16) may use `citation_count` as a signal

## Lifecycle pruning

After each task completes (status → `done`), the runner calls
`prune_expired_memories()`. Deletion rules:

| lifecycle | deleted when |
|-----------|-------------|
| `evergreen` | never |
| `task:<N>` | task N is `done` |
| `slate:<id>` | every task in slate `id` is `done` AND lifecycle's slate id matches current run |
| missing / unrecognized | never (safe default) |

> **Footgun**: scoping a memory to `task:<N>` where N is the current task
> deletes it at that task's success boundary — often not what you want.

After deleting a file, the runner rewrites `MEMORY.md` to remove the
dangling index line.

## When the agent should save vs not

**SAVE**:
- Non-obvious pitfall or gotcha worth knowing for future tasks
- Schema/API constraints that bit us and aren't obvious from docs

**DON'T SAVE**:
- Code patterns derivable from the current codebase
- Git history / who-changed-what
- Ephemeral task context (what I'm working on right now)
- Things already documented in CLAUDE.md

## Merge-suggestion review (dashboard, T16)

When a task completes, the runner may surface a "curate memory" gate in the
dashboard listing memory files above a citation-count threshold. The operator
reviews, merges related files, or deletes stale ones. The gate is
informational; rejecting does not block the task.

## Environment variable for slate ID

The current slate id is exposed to agents via the env var
`CLAUDE_PLAN_EXECUTE_SLATE_ID`. Agents read it when writing
`lifecycle: slate:$CLAUDE_PLAN_EXECUTE_SLATE_ID`.
