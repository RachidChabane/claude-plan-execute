# Recipe: Persona vs CLAUDE.md — Prompt Layer Precedence

## The four prompt layers

| Layer | File | Loaded by | Purpose |
|-------|------|-----------|---------|
| failure-context | `plans/task-{id}/failure-N.md` | Runner | Prior attempt summary injected on retry |
| persona | `docs/persona.md` | Runner | Per-phase project conventions overlay |
| decisions | `docs/decisions/*.md` | Runner | ADR-style rationales surfaced at plan/implement time |
| repo conventions | `.claude/CLAUDE.md` | Claude CLI | Repo-wide instructions auto-loaded by Claude CLI |

## CLAUDE.md

Loaded automatically by Claude CLI for every invocation (global or
project-level). Right for: language conventions, formatting rules, repo
layout, forbidden patterns. Lives in `.claude/CLAUDE.md` or
`~/.claude/CLAUDE.md`.

## persona.md

Injected by the runner as a prompt prefix before every Plan / Review /
Implement phase. Right for: project-specific voice, domain glossary,
architecture principles, what-to-avoid. Lives at `docs/persona.md`
(configurable via `defaults.persona_file`).

## decisions/

Individual `.md` files, each describing one architectural decision.
Surfaced as context during Plan and Implement phases. Right for: ADR-style
reasoning that changes over time and should influence future task planning.
Lives at `docs/decisions/` (configurable via `defaults.decisions_dir`).

## failure-context

Automatically written by the runner after each failed attempt. Contains:
failure class, error summary, gate results, what was tried. Injected at the
start of the next attempt's Implement phase. Do not edit manually.

## Decision tree: which layer to use?

- "Always follow X" → **CLAUDE.md**
- "Phase-specific tone or conventions" → **persona.md**
- "Why did we choose approach Y over Z?" → **decisions/**
- "What went wrong last attempt?" → **failure-context** (automatic)
- "Tactical pitfall discovered mid-task" → **memory file** (agent-curated;
  see [cross-task-memory.md](cross-task-memory.md))
