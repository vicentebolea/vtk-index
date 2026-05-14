"""Tests for vtk_index.pipeline.cli."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from vtk_index.pipeline.cli import app

runner = CliRunner()


def _strip_ansi(s: str) -> str:
    import re

    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def _make_jsonl(tmp_path: Path, records: list[dict]) -> Path:
    p = tmp_path / "knowledge.jsonl"
    with open(p, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return p


def _minimal_record(class_name: str = "vtkActor", module_name: str = "vtkRenderingCore") -> dict:
    return {
        "class_name": class_name,
        "module_name": module_name,
        "vtk_version": "9.3.0",
        "schema_version": "1.0",
        "content_hash": "abc",
        "role": "unknown",
        "methods": [],
        "inheritance": [],
        "semantic_methods": [],
    }


class TestChunkCommand:
    def test_produces_doc_chunks_jsonl(self, tmp_path):
        jsonl = _make_jsonl(tmp_path, [_minimal_record()])
        out = tmp_path / "out"
        result = runner.invoke(app, ["chunk", str(jsonl), "--output-dir", str(out)])
        assert result.exit_code == 0, result.output
        assert (out / "doc-chunks.jsonl").exists()

    def test_output_contains_valid_json(self, tmp_path):
        jsonl = _make_jsonl(tmp_path, [_minimal_record()])
        out = tmp_path / "out"
        runner.invoke(app, ["chunk", str(jsonl), "--output-dir", str(out)])
        with open(out / "doc-chunks.jsonl") as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    assert "chunk_type" in obj
                    assert "content" in obj

    def test_success_message_printed(self, tmp_path):
        jsonl = _make_jsonl(tmp_path, [_minimal_record()])
        out = tmp_path / "out"
        result = runner.invoke(app, ["chunk", str(jsonl), "--output-dir", str(out)])
        assert "doc chunks" in result.output

    def test_missing_artifact_exits_nonzero(self, tmp_path):
        result = runner.invoke(app, ["chunk", str(tmp_path / "nonexistent.jsonl")])
        assert result.exit_code != 0

    def test_multiple_records_chunked(self, tmp_path):
        records = [_minimal_record("vtkActor"), _minimal_record("vtkRenderer")]
        jsonl = _make_jsonl(tmp_path, records)
        out = tmp_path / "out"
        result = runner.invoke(app, ["chunk", str(jsonl), "--output-dir", str(out)])
        assert result.exit_code == 0
        lines = [ln for ln in (out / "doc-chunks.jsonl").read_text().splitlines() if ln.strip()]
        assert len(lines) >= 2

    def test_vtk_version_option(self, tmp_path):
        jsonl = _make_jsonl(tmp_path, [_minimal_record()])
        out = tmp_path / "out"
        result = runner.invoke(
            app, ["chunk", str(jsonl), "--output-dir", str(out), "--vtk-version", "9.6.1"]
        )
        assert result.exit_code == 0

    def test_import_error_exits_nonzero(self, tmp_path):
        jsonl = _make_jsonl(tmp_path, [_minimal_record()])
        with patch(
            "vtk_index.pipeline.cli.chunk.__wrapped__"
            if hasattr(app, "__wrapped__")
            else "vtk_knowledge.index.api_index.VTKAPIIndex.from_jsonl",
            side_effect=RuntimeError("simulated failure"),
        ):
            result = runner.invoke(app, ["chunk", str(jsonl)])
        assert result.exit_code != 0


class TestSnapshotCommand:
    def test_qdrant_connection_error_exits_nonzero(self):
        result = runner.invoke(
            app,
            ["snapshot", "--qdrant-url", "http://localhost:19999", "--vtk-version", "9.3.0"],
        )
        assert result.exit_code != 0

    def test_error_message_printed(self):
        result = runner.invoke(
            app,
            ["snapshot", "--qdrant-url", "http://localhost:19999", "--vtk-version", "9.3.0"],
        )
        assert "Error" in result.output or result.exit_code != 0


class TestSearchCommand:
    def test_qdrant_unavailable_exits_nonzero(self):
        result = runner.invoke(
            app,
            ["search", "sphere source", "--qdrant-url", "http://localhost:19999"],
        )
        assert result.exit_code != 0

    def test_error_message_on_failure(self):
        result = runner.invoke(
            app,
            ["search", "sphere source", "--qdrant-url", "http://localhost:19999"],
        )
        assert "Error" in result.output

    def test_help_lists_options(self):
        result = runner.invoke(app, ["search", "--help"])
        assert result.exit_code == 0
        out = _strip_ansi(result.output)
        assert "--collection" in out
        assert "--top" in out
        assert "--role" in out
        assert "--min-visibility" in out
        assert "--json" in out


class TestNoArgsHelp:
    def test_no_args_shows_help(self):
        result = runner.invoke(app, [])
        # no_args_is_help=True exits with code 0 in newer typer, 2 in some versions
        assert result.exit_code in (0, 2)
        assert "chunk" in result.output or "vtk-index" in result.output
