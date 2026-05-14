"""Payload filter helpers for Qdrant queries."""

from __future__ import annotations

from typing import Any

from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue, Range


def build_filter(filters: dict[str, Any] | Filter | None) -> Filter | None:
    """Convert a plain dict to a Qdrant Filter, or pass through an existing Filter."""
    if filters is None:
        return None
    if isinstance(filters, Filter):
        return filters

    conditions = []
    for field, value in filters.items():
        if isinstance(value, dict):
            conditions.append(
                FieldCondition(
                    key=field,
                    range=Range(
                        gt=value.get("gt"),
                        gte=value.get("gte"),
                        lt=value.get("lt"),
                        lte=value.get("lte"),
                    ),
                )
            )
        elif isinstance(value, list):
            conditions.append(FieldCondition(key=field, match=MatchAny(any=value)))
        else:
            conditions.append(FieldCondition(key=field, match=MatchValue(value=value)))

    return Filter(must=conditions) if conditions else None


class PayloadFilter:
    """Builder for common VTK payload filters."""

    def __init__(self) -> None:
        self._d: dict[str, Any] = {}

    def by_class(self, class_name: str) -> PayloadFilter:
        self._d["class_names"] = class_name
        return self

    def by_role(self, role: str) -> PayloadFilter:
        self._d["role"] = role
        return self

    def by_input_type(self, dtype: str) -> PayloadFilter:
        self._d["input_datatype"] = dtype
        return self

    def by_output_type(self, dtype: str) -> PayloadFilter:
        self._d["output_datatype"] = dtype
        return self

    def min_visibility(self, score: float) -> PayloadFilter:
        self._d["visibility_score"] = {"gte": score}
        return self

    def build(self) -> Filter | None:
        return build_filter(self._d)
