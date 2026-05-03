#!/usr/bin/env python3
"""
workflow-server.py — Wrapper for the dashboard server.

Kept for backward compatibility with users who symlinked this script per
the README install instructions. The actual server lives in
claude_plan_execute.server. New users should symlink
bin/claude-workflow-dashboard instead.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from claude_plan_execute.server import main

if __name__ == "__main__":
    main()
