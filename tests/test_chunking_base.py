"""Tests for vtk_index.chunking.base."""

from vtk_index.chunking.base import Chunk, ChunkType


class TestChunkType:
    def test_all_expected_values(self):
        values = {ct.value for ct in ChunkType}
        assert "class_overview" in values
        assert "method_doc" in values
        assert "inheritance" in values
        assert "pipeline_example" in values
        assert "query_example" in values

    def test_is_string_enum(self):
        assert ChunkType.CLASS_OVERVIEW == "class_overview"
        assert ChunkType.METHOD_DOC == "method_doc"

    def test_seven_types_defined(self):
        assert len(ChunkType) == 7


class TestChunk:
    def test_minimal_construction(self):
        c = Chunk(
            chunk_id="abc123",
            chunk_type=ChunkType.CLASS_OVERVIEW,
            content="vtkActor is an actor.",
        )
        assert c.chunk_id == "abc123"
        assert c.chunk_type == ChunkType.CLASS_OVERVIEW
        assert c.content == "vtkActor is an actor."
        assert c.class_names == []
        assert c.module_names == []
        assert c.role is None
        assert c.visibility_score is None
        assert c.source == "vtk-knowledge"
        assert c.source_path == ""
        assert c.vtk_version == ""

    def test_full_construction(self):
        c = Chunk(
            chunk_id="xyz",
            chunk_type=ChunkType.METHOD_DOC,
            content="GetBounds() -> tuple",
            class_names=["vtkActor"],
            module_names=["vtkRenderingCore"],
            role="renderer",
            input_datatype="vtkPolyData",
            output_datatype=None,
            visibility_score=0.9,
            source="vtk-knowledge",
            source_path="vtkActor",
            vtk_version="9.3.0",
        )
        assert c.class_names == ["vtkActor"]
        assert c.module_names == ["vtkRenderingCore"]
        assert c.role == "renderer"
        assert c.visibility_score == 0.9
        assert c.vtk_version == "9.3.0"

    def test_roundtrip_json(self):
        c = Chunk(
            chunk_id="def456",
            chunk_type=ChunkType.INHERITANCE,
            content="vtkActor inherits from: vtkProp3D",
            class_names=["vtkActor"],
            vtk_version="9.6.1",
        )
        serialised = c.model_dump_json()
        restored = Chunk.model_validate_json(serialised)
        assert restored.chunk_id == c.chunk_id
        assert restored.chunk_type == ChunkType.INHERITANCE
        assert restored.content == c.content
        assert restored.vtk_version == "9.6.1"

    def test_chunk_type_from_string(self):
        c = Chunk(
            chunk_id="x",
            chunk_type="pipeline_example",
            content="some pipeline",
        )
        assert c.chunk_type == ChunkType.PIPELINE_EXAMPLE

    def test_model_dump_includes_chunk_type_value(self):
        c = Chunk(
            chunk_id="y",
            chunk_type=ChunkType.PROPERTY_GROUP,
            content="props",
        )
        d = c.model_dump()
        assert d["chunk_type"] == "property_group"
