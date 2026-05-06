"""Chunk Python example/test files into Qdrant-ready Chunk instances.

This is a thin wrapper that delegates to the existing vtk_rag lifecycle
analysis. It produces PIPELINE_EXAMPLE and QUERY_EXAMPLE chunks.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .base import Chunk, ChunkType


def chunk_code_file(
    source: str,
    path: str,
    vtk_version: str = "",
) -> list[Chunk]:
    """Extract semantic chunks from a VTK Python script.

    Falls back to a single whole-file chunk if the lifecycle analyser fails.
    """
    try:
        from vtk_rag.chunking.code.chunker import CodeChunker
        from vtk_rag.mcp import get_vtk_client
        chunker = CodeChunker(source, path, get_vtk_client())
        raw_chunks = chunker.extract_chunks()
    except Exception:
        raw_chunks = []

    if not raw_chunks:
        return [
            Chunk(
                chunk_id=hashlib.sha1(f"{path}:full".encode()).hexdigest()[:16],
                chunk_type=ChunkType.PIPELINE_EXAMPLE,
                content=source[:4000],
                source="examples",
                source_path=path,
                vtk_version=vtk_version,
            )
        ]

    chunks: list[Chunk] = []
    for i, raw in enumerate(raw_chunks):
        chunk_type = (
            ChunkType.QUERY_EXAMPLE
            if raw.get("chunk_type") == "query"
            else ChunkType.PIPELINE_EXAMPLE
        )
        chunks.append(
            Chunk(
                chunk_id=hashlib.sha1(f"{path}:{i}".encode()).hexdigest()[:16],
                chunk_type=chunk_type,
                content=raw.get("content", ""),
                class_names=raw.get("vtk_class_names", []),
                role=raw.get("role"),
                vtk_version=vtk_version,
                source="examples",
                source_path=path,
            )
        )
    return chunks
