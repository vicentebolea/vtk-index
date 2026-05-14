"""Tests for search: Retriever.hybrid_search and the search CLI command."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from vtk_index.chunking.base import Chunk, ChunkType
from vtk_index.pipeline.cli import app
from vtk_index.query.client import Retriever
from vtk_index.query.filters import PayloadFilter

runner = CliRunner()

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_DENSE_VEC = [0.1] * 384
_SPARSE_INDICES = [1, 5, 42]
_SPARSE_VALUES = [0.9, 0.5, 0.3]


def _fake_sparse():
    s = SimpleNamespace()
    import numpy as np

    s.indices = np.array(_SPARSE_INDICES)
    s.values = np.array(_SPARSE_VALUES)
    return s


def _scored_point(i: int, class_name: str, chunk_type: str, role: str, content: str):
    """Build a minimal ScoredPoint-like object that _to_chunk can consume."""
    return SimpleNamespace(
        id=i,
        score=1.0 - i * 0.1,
        payload={
            "chunk_type": chunk_type,
            "content": content,
            "class_names": [class_name],
            "module_names": ["vtkTestModule"],
            "role": role,
            "input_datatype": None,
            "output_datatype": None,
            "visibility_score": 0.8,
            "source": "vtk-knowledge",
            "source_path": class_name,
            "vtk_version": "9.3.0",
        },
    )


def _fake_query_response(*points):
    return SimpleNamespace(points=list(points))


# ---------------------------------------------------------------------------
# Retriever unit tests (mocked Qdrant + embedders)
# ---------------------------------------------------------------------------


class TestRetrieverHybridSearch:
    @pytest.fixture()
    def retriever(self):
        with (
            patch("vtk_index.query.client.QdrantClient"),
            patch("vtk_index.query.client.DenseEmbedder") as MockDense,
            patch("vtk_index.query.client.SparseEmbedder") as MockSparse,
        ):
            MockDense.return_value.encode.return_value = _DENSE_VEC
            MockSparse.return_value.embed.return_value = _fake_sparse()
            r = Retriever(qdrant_url="http://localhost:6333")
            yield r

    def test_returns_chunk_list(self, retriever):
        pt = _scored_point(0, "vtkSphereSource", "class_overview", "source", "Sphere source.")
        retriever.client.query_points.return_value = _fake_query_response(pt)
        chunks = retriever.hybrid_search("sphere source", "vtk_docs")
        assert len(chunks) == 1
        assert isinstance(chunks[0], Chunk)

    def test_chunk_fields_populated(self, retriever):
        pt = _scored_point(0, "vtkSphereSource", "class_overview", "source", "Sphere source.")
        retriever.client.query_points.return_value = _fake_query_response(pt)
        c = retriever.hybrid_search("sphere source", "vtk_docs")[0]
        assert c.class_names == ["vtkSphereSource"]
        assert c.role == "source"
        assert c.content == "Sphere source."
        assert c.chunk_type == ChunkType.CLASS_OVERVIEW
        assert c.vtk_version == "9.3.0"

    def test_multiple_results_ordered(self, retriever):
        pts = [
            _scored_point(i, f"vtkClass{i}", "class_overview", "filter", f"content {i}")
            for i in range(5)
        ]
        retriever.client.query_points.return_value = _fake_query_response(*pts)
        chunks = retriever.hybrid_search("filter class", "vtk_docs", k=5)
        assert len(chunks) == 5
        assert [c.class_names[0] for c in chunks] == [f"vtkClass{i}" for i in range(5)]

    def test_dense_vector_sent_to_qdrant(self, retriever):
        retriever.client.query_points.return_value = _fake_query_response()
        retriever.hybrid_search("query", "vtk_docs")
        call_kwargs = retriever.client.query_points.call_args
        prefetches = call_kwargs.kwargs["prefetch"]
        assert prefetches[0].query == _DENSE_VEC

    def test_sparse_vector_sent_to_qdrant(self, retriever):
        retriever.client.query_points.return_value = _fake_query_response()
        retriever.hybrid_search("query", "vtk_docs")
        call_kwargs = retriever.client.query_points.call_args
        prefetches = call_kwargs.kwargs["prefetch"]
        assert prefetches[1].query.indices == _SPARSE_INDICES
        assert prefetches[1].query.values == _SPARSE_VALUES

    def test_k_passed_as_limit(self, retriever):
        retriever.client.query_points.return_value = _fake_query_response()
        retriever.hybrid_search("query", "vtk_docs", k=7)
        call_kwargs = retriever.client.query_points.call_args
        assert call_kwargs.kwargs["limit"] == 7

    def test_filter_forwarded_to_prefetch(self, retriever):
        retriever.client.query_points.return_value = _fake_query_response()
        filt = PayloadFilter().by_role("source")
        retriever.hybrid_search("query", "vtk_docs", filters=filt)
        call_kwargs = retriever.client.query_points.call_args
        for prefetch in call_kwargs.kwargs["prefetch"]:
            assert prefetch.filter is not None

    def test_empty_results(self, retriever):
        retriever.client.query_points.return_value = _fake_query_response()
        assert retriever.hybrid_search("query", "vtk_docs") == []

    def test_search_docs_targets_docs_collection(self, retriever):
        retriever.client.query_points.return_value = _fake_query_response()
        retriever.search_docs("query")
        call_kwargs = retriever.client.query_points.call_args
        assert call_kwargs.kwargs["collection_name"] == "vtk_docs"

    def test_search_code_targets_code_collection(self, retriever):
        retriever.client.query_points.return_value = _fake_query_response()
        retriever.search_code("query")
        call_kwargs = retriever.client.query_points.call_args
        assert call_kwargs.kwargs["collection_name"] == "vtk_code"

    def test_unknown_chunk_type_defaults_to_class_overview(self, retriever):
        pt = _scored_point(0, "vtkFoo", "totally_unknown_type", "unknown", "content")
        pt.payload["chunk_type"] = "totally_unknown_type"
        retriever.client.query_points.return_value = _fake_query_response(pt)
        c = retriever.hybrid_search("query", "vtk_docs")[0]
        assert c.chunk_type == ChunkType.CLASS_OVERVIEW


# ---------------------------------------------------------------------------
# CLI search command tests (mocked Retriever)
# ---------------------------------------------------------------------------


def _make_chunks(n: int = 2) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"id{i}",
            chunk_type=ChunkType.CLASS_OVERVIEW,
            content=f"Content for class {i}.",
            class_names=[f"vtkClass{i}"],
            module_names=["vtkTestModule"],
            role="source",
            visibility_score=0.8,
            vtk_version="9.3.0",
        )
        for i in range(n)
    ]


def _patch_retriever(chunks: list[Chunk]):
    """Context manager: patch Retriever so it returns *chunks* without hitting Qdrant."""
    mock_retriever = MagicMock(spec=Retriever)
    mock_retriever.hybrid_search.return_value = chunks
    return patch("vtk_index.query.client.Retriever", return_value=mock_retriever)


class TestSearchCLIQueries:
    def test_results_displayed(self):
        with _patch_retriever(_make_chunks(2)):
            result = runner.invoke(app, ["search", "sphere"])
        assert result.exit_code == 0
        assert "vtkClass0" in result.output
        assert "vtkClass1" in result.output

    def test_numbered_output(self):
        with _patch_retriever(_make_chunks(3)):
            result = runner.invoke(app, ["search", "sphere"])
        assert "[1]" in result.output
        assert "[2]" in result.output
        assert "[3]" in result.output

    def test_content_preview_shown(self):
        with _patch_retriever(_make_chunks(1)):
            result = runner.invoke(app, ["search", "sphere"])
        assert "Content for class 0." in result.output

    def test_role_shown_in_brackets(self):
        with _patch_retriever(_make_chunks(1)):
            result = runner.invoke(app, ["search", "sphere"])
        assert "[source]" in result.output

    def test_empty_results_message(self):
        with _patch_retriever([]):
            result = runner.invoke(app, ["search", "xyzzy"])
        assert result.exit_code == 0
        assert "No results" in result.output

    def test_top_n_forwarded(self):
        mock_retriever = MagicMock(spec=Retriever)
        mock_retriever.hybrid_search.return_value = []
        with patch("vtk_index.query.client.Retriever", return_value=mock_retriever):
            runner.invoke(app, ["search", "sphere", "--top", "3"])
        mock_retriever.hybrid_search.assert_called_once()
        assert mock_retriever.hybrid_search.call_args.kwargs["k"] == 3

    def test_collection_docs_default(self):
        mock_retriever = MagicMock(spec=Retriever)
        mock_retriever.hybrid_search.return_value = []
        with patch("vtk_index.query.client.Retriever", return_value=mock_retriever):
            runner.invoke(app, ["search", "sphere"])
        assert mock_retriever.hybrid_search.call_args.kwargs["collection"] == "vtk_docs"

    def test_collection_code_option(self):
        mock_retriever = MagicMock(spec=Retriever)
        mock_retriever.hybrid_search.return_value = []
        with patch("vtk_index.query.client.Retriever", return_value=mock_retriever):
            runner.invoke(app, ["search", "sphere", "--collection", "code"])
        assert mock_retriever.hybrid_search.call_args.kwargs["collection"] == "vtk_code"

    def test_role_filter_applied(self):
        mock_retriever = MagicMock(spec=Retriever)
        mock_retriever.hybrid_search.return_value = []
        with patch("vtk_index.query.client.Retriever", return_value=mock_retriever):
            runner.invoke(app, ["search", "sphere", "--role", "source"])
        filt = mock_retriever.hybrid_search.call_args.kwargs["filters"]
        built = filt.build()
        assert built is not None
        assert any(c.key == "role" for c in built.must)

    def test_min_visibility_filter_applied(self):
        mock_retriever = MagicMock(spec=Retriever)
        mock_retriever.hybrid_search.return_value = []
        with patch("vtk_index.query.client.Retriever", return_value=mock_retriever):
            runner.invoke(app, ["search", "sphere", "--min-visibility", "0.7"])
        filt = mock_retriever.hybrid_search.call_args.kwargs["filters"]
        built = filt.build()
        assert built is not None
        assert any(c.key == "visibility_score" for c in built.must)

    def test_no_filter_when_defaults(self):
        mock_retriever = MagicMock(spec=Retriever)
        mock_retriever.hybrid_search.return_value = []
        with patch("vtk_index.query.client.Retriever", return_value=mock_retriever):
            runner.invoke(app, ["search", "sphere"])
        filt = mock_retriever.hybrid_search.call_args.kwargs["filters"]
        assert filt.build() is None

    def test_json_output_valid(self):
        with _patch_retriever(_make_chunks(2)):
            result = runner.invoke(app, ["search", "sphere", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)
        assert len(data) == 2

    def test_json_output_fields(self):
        with _patch_retriever(_make_chunks(1)):
            result = runner.invoke(app, ["search", "sphere", "--json"])
        item = json.loads(result.output)[0]
        assert "chunk_type" in item
        assert "content" in item
        assert "class_names" in item

    def test_query_string_forwarded(self):
        mock_retriever = MagicMock(spec=Retriever)
        mock_retriever.hybrid_search.return_value = []
        with patch("vtk_index.query.client.Retriever", return_value=mock_retriever):
            runner.invoke(app, ["search", "read STL file"])
        first_arg = mock_retriever.hybrid_search.call_args.args[0]
        assert first_arg == "read STL file"
