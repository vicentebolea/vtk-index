"""Chunk VTKDocRecord objects into Qdrant-ready Chunk instances."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from .base import Chunk, ChunkType

if TYPE_CHECKING:
    from vtk_knowledge import VTKDocRecord


def chunk_record(record: "VTKDocRecord") -> list[Chunk]:
    """Split a single VTKDocRecord into one or more Chunks."""
    chunks: list[Chunk] = []
    base_payload = dict(
        class_names=[record.class_name],
        module_names=[record.module_name],
        role=record.role.value,
        input_datatype=record.input_datatype,
        output_datatype=record.output_datatype,
        visibility_score=record.visibility_score,
        vtk_version=record.vtk_version,
        source="vtk-knowledge",
        source_path=record.class_name,
    )

    # Class overview
    overview_text = _build_overview(record)
    if overview_text:
        chunks.append(
            Chunk(
                chunk_id=_cid(record.class_name, "overview"),
                chunk_type=ChunkType.CLASS_OVERVIEW,
                content=overview_text,
                **base_payload,
            )
        )

    # One chunk per semantic method
    for method_name in record.semantic_methods:
        method = next((m for m in record.methods if m.name == method_name), None)
        if method is None:
            continue
        content = "\n".join(method.signatures) + ("\n\n" + method.doc if method.doc else "")
        chunks.append(
            Chunk(
                chunk_id=_cid(record.class_name, f"method_{method_name}"),
                chunk_type=ChunkType.METHOD_DOC,
                content=content.strip(),
                **base_payload,
            )
        )

    # Inheritance
    if record.inheritance:
        chunks.append(
            Chunk(
                chunk_id=_cid(record.class_name, "inheritance"),
                chunk_type=ChunkType.INHERITANCE,
                content=f"{record.class_name} inherits from: {', '.join(record.inheritance)}",
                **base_payload,
            )
        )

    return chunks


def _build_overview(record: "VTKDocRecord") -> str:
    parts = [f"{record.class_name} ({record.module_name})"]
    if record.synopsis:
        parts.append(record.synopsis)
    if record.class_doc:
        parts.append(record.class_doc[:500])
    if record.action_phrase:
        parts.append(f"Action: {record.action_phrase}")
    return "\n".join(parts)


def _cid(class_name: str, suffix: str) -> str:
    return hashlib.sha1(f"{class_name}:{suffix}".encode()).hexdigest()[:16]
