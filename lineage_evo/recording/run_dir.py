"""Run directory helpers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import uuid4


class RunDirectoryResolver:
    def __init__(self, parent_dir: str | Path) -> None:
        self.parent_dir = Path(parent_dir)

    def create(self, *, run_id: str, label: str = "run") -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = uuid4().hex[:6]
        safe_label = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in label)
        path = self.parent_dir / f"{timestamp}_{safe_label}_{run_id}_{suffix}"
        path.mkdir(parents=True, exist_ok=False)
        return path
