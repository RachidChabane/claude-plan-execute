"""Dependency validation: topological sort + cycle detection."""
from __future__ import annotations

import sys
from collections import deque

from claude_plan_execute.colors import red


def validate_dependencies(tasks: list[dict]) -> list[dict]:
    """Topological sort. Raises on cycles or missing IDs."""
    ids = {t["id"] for t in tasks}
    graph: dict[int, list[int]] = {t["id"]: t.get("depends_on", []) for t in tasks}
    by_id = {t["id"]: t for t in tasks}

    for tid, deps in graph.items():
        for d in deps:
            if d not in ids:
                print(red(f"Error: Task {tid} depends on non-existent task {d}"), file=sys.stderr)
                sys.exit(1)

    # Kahn's algorithm
    in_degree = {tid: len(deps) for tid, deps in graph.items()}
    reverse: dict[int, list[int]] = {tid: [] for tid in ids}
    for tid, deps in graph.items():
        for d in deps:
            reverse[d].append(tid)

    queue = deque(sorted(tid for tid, deg in in_degree.items() if deg == 0))
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        newly_ready = []
        for neighbor in reverse[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                newly_ready.append(neighbor)
        queue.extend(sorted(newly_ready))

    if len(order) != len(ids):
        remaining = ids - set(order)
        print(red(f"Error: Circular dependency detected involving tasks: {remaining}"), file=sys.stderr)
        sys.exit(1)

    return [by_id[tid] for tid in order]
