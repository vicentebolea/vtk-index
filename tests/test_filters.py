"""Tests for vtk_index.query.filters."""

from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue, Range

from vtk_index.query.filters import PayloadFilter, build_filter


class TestBuildFilter:
    def test_none_returns_none(self):
        assert build_filter(None) is None

    def test_filter_passthrough(self):
        f = Filter()
        assert build_filter(f) is f

    def test_string_value_creates_match_value(self):
        result = build_filter({"role": "renderer"})
        assert isinstance(result, Filter)
        assert len(result.must) == 1
        cond = result.must[0]
        assert isinstance(cond, FieldCondition)
        assert cond.key == "role"
        assert isinstance(cond.match, MatchValue)
        assert cond.match.value == "renderer"

    def test_list_value_creates_match_any(self):
        result = build_filter({"class_names": ["vtkActor", "vtkRenderer"]})
        assert isinstance(result, Filter)
        cond = result.must[0]
        assert isinstance(cond.match, MatchAny)
        assert cond.match.any == ["vtkActor", "vtkRenderer"]

    def test_dict_value_creates_range(self):
        result = build_filter({"visibility_score": {"gte": 0.5}})
        assert isinstance(result, Filter)
        cond = result.must[0]
        assert isinstance(cond.range, Range)
        assert cond.range.gte == 0.5

    def test_empty_dict_returns_none(self):
        assert build_filter({}) is None

    def test_multiple_conditions(self):
        result = build_filter({"role": "filter", "vtk_version": "9.3.0"})
        assert isinstance(result, Filter)
        assert len(result.must) == 2

    def test_range_all_bounds(self):
        result = build_filter({"visibility_score": {"gt": 0.1, "gte": 0.2, "lt": 0.9, "lte": 0.8}})
        cond = result.must[0]
        r = cond.range
        assert r.gt == 0.1
        assert r.gte == 0.2
        assert r.lt == 0.9
        assert r.lte == 0.8


class TestPayloadFilter:
    def test_empty_builds_none(self):
        assert PayloadFilter().build() is None

    def test_by_class(self):
        f = PayloadFilter().by_class("vtkActor").build()
        assert isinstance(f, Filter)
        assert f.must[0].key == "class_names"
        assert f.must[0].match.value == "vtkActor"

    def test_by_role(self):
        f = PayloadFilter().by_role("source").build()
        assert f.must[0].key == "role"
        assert f.must[0].match.value == "source"

    def test_by_input_type(self):
        f = PayloadFilter().by_input_type("vtkPolyData").build()
        assert f.must[0].key == "input_datatype"

    def test_by_output_type(self):
        f = PayloadFilter().by_output_type("vtkUnstructuredGrid").build()
        assert f.must[0].key == "output_datatype"

    def test_min_visibility(self):
        f = PayloadFilter().min_visibility(0.7).build()
        cond = f.must[0]
        assert cond.key == "visibility_score"
        assert isinstance(cond.range, Range)
        assert cond.range.gte == 0.7

    def test_chaining(self):
        f = PayloadFilter().by_role("filter").min_visibility(0.5).build()
        assert len(f.must) == 2

    def test_returns_payload_filter_instance(self):
        pf = PayloadFilter()
        assert pf.by_class("vtkFoo") is pf
