"""Tests for vtk_index.chunking.doc_chunker."""

from vtk_index.chunking.base import Chunk, ChunkType
from vtk_index.chunking.doc_chunker import _build_overview, _cid, chunk_record


def _make_record(**kwargs):
    from vtk_knowledge.schema.records import VTKDocRecord

    defaults = dict(class_name="vtkActor", module_name="vtkRenderingCore")
    defaults.update(kwargs)
    return VTKDocRecord(**defaults)


class TestChunkId:
    def test_deterministic(self):
        assert _cid("vtkActor", "overview") == _cid("vtkActor", "overview")

    def test_different_for_different_inputs(self):
        assert _cid("vtkActor", "overview") != _cid("vtkActor", "inheritance")
        assert _cid("vtkActor", "overview") != _cid("vtkRenderer", "overview")

    def test_length_16(self):
        assert len(_cid("vtkActor", "overview")) == 16


class TestBuildOverview:
    def test_minimal_record(self):
        r = _make_record()
        text = _build_overview(r)
        assert "vtkActor" in text
        assert "vtkRenderingCore" in text

    def test_synopsis_included(self):
        r = _make_record(synopsis="Renders geometry in a scene.")
        text = _build_overview(r)
        assert "Renders geometry in a scene." in text

    def test_class_doc_truncated_at_500(self):
        r = _make_record(class_doc="x" * 1000)
        text = _build_overview(r)
        assert text.count("x") == 500

    def test_action_phrase_included(self):
        r = _make_record(action_phrase="scene rendering")
        text = _build_overview(r)
        assert "Action: scene rendering" in text

    def test_empty_optional_fields(self):
        r = _make_record()
        text = _build_overview(r)
        assert "Action:" not in text


class TestChunkRecord:
    def test_overview_chunk_produced(self):
        r = _make_record(synopsis="An actor.")
        chunks = chunk_record(r)
        types = [c.chunk_type for c in chunks]
        assert ChunkType.CLASS_OVERVIEW in types

    def test_overview_chunk_id_stable(self):
        r = _make_record()
        chunks = chunk_record(r)
        overview = next(c for c in chunks if c.chunk_type == ChunkType.CLASS_OVERVIEW)
        assert overview.chunk_id == _cid("vtkActor", "overview")

    def test_method_chunks_produced(self):
        from vtk_knowledge.schema.records import VTKMethod

        r = _make_record(
            semantic_methods=["GetBounds"],
            methods=[
                VTKMethod(
                    name="GetBounds",
                    signatures=["GetBounds() -> tuple"],
                    doc="Return bounding box.",
                )
            ],
        )
        chunks = chunk_record(r)
        method_chunks = [c for c in chunks if c.chunk_type == ChunkType.METHOD_DOC]
        assert len(method_chunks) == 1
        assert "GetBounds" in method_chunks[0].content

    def test_method_chunk_includes_doc(self):
        from vtk_knowledge.schema.records import VTKMethod

        r = _make_record(
            semantic_methods=["SetMapper"],
            methods=[
                VTKMethod(name="SetMapper", signatures=["SetMapper(m)"], doc="Set the mapper.")
            ],
        )
        chunks = chunk_record(r)
        mc = next(c for c in chunks if c.chunk_type == ChunkType.METHOD_DOC)
        assert "Set the mapper." in mc.content

    def test_method_not_in_methods_list_skipped(self):
        r = _make_record(semantic_methods=["NonExistentMethod"], methods=[])
        chunks = chunk_record(r)
        method_chunks = [c for c in chunks if c.chunk_type == ChunkType.METHOD_DOC]
        assert method_chunks == []

    def test_inheritance_chunk_produced(self):
        r = _make_record(inheritance=["vtkProp3D", "vtkObjectBase"])
        chunks = chunk_record(r)
        inh = [c for c in chunks if c.chunk_type == ChunkType.INHERITANCE]
        assert len(inh) == 1
        assert "vtkProp3D" in inh[0].content
        assert "vtkObjectBase" in inh[0].content

    def test_no_inheritance_chunk_when_empty(self):
        r = _make_record(inheritance=[])
        chunks = chunk_record(r)
        inh = [c for c in chunks if c.chunk_type == ChunkType.INHERITANCE]
        assert inh == []

    def test_chunk_payload_contains_class_name(self):
        r = _make_record()
        chunks = chunk_record(r)
        for c in chunks:
            assert "vtkActor" in c.class_names

    def test_chunk_payload_contains_module_name(self):
        r = _make_record()
        chunks = chunk_record(r)
        for c in chunks:
            assert "vtkRenderingCore" in c.module_names

    def test_chunk_payload_contains_vtk_version(self):
        r = _make_record(vtk_version="9.6.1")
        chunks = chunk_record(r)
        for c in chunks:
            assert c.vtk_version == "9.6.1"

    def test_chunk_payload_contains_role(self):
        r = _make_record(role="renderer")
        chunks = chunk_record(r)
        for c in chunks:
            assert c.role == "renderer"

    def test_visibility_score_propagated(self):
        r = _make_record(visibility_score=0.75)
        chunks = chunk_record(r)
        for c in chunks:
            assert c.visibility_score == 0.75

    def test_empty_record_produces_overview_only(self):
        r = _make_record()
        chunks = chunk_record(r)
        assert len(chunks) == 1
        assert chunks[0].chunk_type == ChunkType.CLASS_OVERVIEW

    def test_all_chunks_are_chunk_instances(self):
        from vtk_knowledge.schema.records import VTKMethod

        r = _make_record(
            synopsis="Actor",
            inheritance=["vtkProp3D"],
            semantic_methods=["GetBounds"],
            methods=[VTKMethod(name="GetBounds", doc="bounds")],
        )
        chunks = chunk_record(r)
        for c in chunks:
            assert isinstance(c, Chunk)
