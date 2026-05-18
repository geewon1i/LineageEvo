"""Simple factor expression representation.

This module intentionally stays backend-neutral. It extracts enough structure for
deterministic validation and prior rewriting evidence without committing to a
particular alpha expression engine.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|-?\d+(?:\.\d+)?|[()+\-*/,]")
NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass(frozen=True)
class FactorExpression:
    raw: str

    @property
    def normalized(self) -> str:
        return " ".join(TOKEN_RE.findall(self.raw))

    @property
    def tokens(self) -> list[str]:
        return TOKEN_RE.findall(self.raw)

    @property
    def length(self) -> int:
        return len(self.tokens)

    def names(self) -> set[str]:
        return set(NAME_RE.findall(self.raw))


@dataclass(frozen=True)
class ExpressionDiff:
    added_tokens: list[str] = field(default_factory=list)
    removed_tokens: list[str] = field(default_factory=list)
    unchanged_tokens: list[str] = field(default_factory=list)
    summary: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "added_tokens": self.added_tokens,
            "removed_tokens": self.removed_tokens,
            "unchanged_tokens": self.unchanged_tokens,
            "summary": self.summary,
        }


def diff_expressions(parent: FactorExpression, child: FactorExpression) -> ExpressionDiff:
    parent_tokens = parent.tokens
    child_tokens = child.tokens
    parent_counts: dict[str, int] = {}
    child_counts: dict[str, int] = {}
    for token in parent_tokens:
        parent_counts[token] = parent_counts.get(token, 0) + 1
    for token in child_tokens:
        child_counts[token] = child_counts.get(token, 0) + 1

    added: list[str] = []
    removed: list[str] = []
    unchanged: list[str] = []
    for token in sorted(set(parent_counts) | set(child_counts)):
        common = min(parent_counts.get(token, 0), child_counts.get(token, 0))
        unchanged.extend([token] * common)
        added.extend([token] * max(0, child_counts.get(token, 0) - parent_counts.get(token, 0)))
        removed.extend([token] * max(0, parent_counts.get(token, 0) - child_counts.get(token, 0)))

    summary = f"added={added or []}; removed={removed or []}"
    return ExpressionDiff(added_tokens=added, removed_tokens=removed, unchanged_tokens=unchanged, summary=summary)

