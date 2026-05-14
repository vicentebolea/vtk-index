"""Tests for vtk_index.artifact.fetcher and the download CLI command."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from vtk_index.artifact.fetcher import _CACHE_DIR, _DEFAULT_REPOSITORY, fetch_from_ghcr
from vtk_index.pipeline.cli import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _layer_blob(*chunks: dict) -> bytes:
    """Build a gzipped OCI layer tar containing doc-chunks.jsonl."""
    content = "\n".join(json.dumps(c) for c in chunks).encode()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo(name="doc-chunks.jsonl")
        info.size = len(content)
        tar.addfile(info, io.BytesIO(content))
    return buf.getvalue()


def _minimal_chunk(class_name: str = "vtkActor", chunk_type: str = "class_overview") -> dict:
    return {
        "chunk_id": f"id_{class_name}",
        "chunk_type": chunk_type,
        "content": f"Content for {class_name}.",
        "class_names": [class_name],
        "module_names": ["vtkTestModule"],
        "role": "source",
        "input_datatype": None,
        "output_datatype": None,
        "visibility_score": 0.8,
        "source": "vtk-knowledge",
        "source_path": class_name,
        "vtk_version": "9.3.0",
    }


def _urlopen_side_effect(blob: bytes):
    responses = iter(
        [
            json.dumps({"token": "test-token"}).encode(),
            json.dumps({"layers": [{"digest": "sha256:abc123"}]}).encode(),
            blob,
        ]
    )

    def _open(req_or_url, **_kw):
        data = next(responses)
        cm = MagicMock()
        cm.__enter__ = lambda s: MagicMock(read=lambda: data)
        cm.__exit__ = MagicMock(return_value=False)
        return cm

    return _open


# ---------------------------------------------------------------------------
# fetch_from_ghcr unit tests
# ---------------------------------------------------------------------------


class TestFetchFromGhcr:
    def test_downloads_and_caches(self, tmp_path):
        blob = _layer_blob(_minimal_chunk("vtkSphereSource"))
        with patch(
            "vtk_index.artifact.fetcher.urllib.request.urlopen",
            side_effect=_urlopen_side_effect(blob),
        ):
            path = fetch_from_ghcr("9.6.1", cache_dir=tmp_path)

        assert path.exists()
        assert path.name == "doc-chunks-9.6.1.jsonl"

    def test_cache_hit_skips_network(self, tmp_path):
        cached = tmp_path / "doc-chunks-9.6.1.jsonl"
        cached.write_text(json.dumps(_minimal_chunk()) + "\n")

        with patch("vtk_index.artifact.fetcher.urllib.request.urlopen") as mock_open:
            result = fetch_from_ghcr("9.6.1", cache_dir=tmp_path)
            mock_open.assert_not_called()

        assert result == cached

    def test_returns_correct_path(self, tmp_path):
        blob = _layer_blob(_minimal_chunk())
        with patch(
            "vtk_index.artifact.fetcher.urllib.request.urlopen",
            side_effect=_urlopen_side_effect(blob),
        ):
            path = fetch_from_ghcr("9.3.0", cache_dir=tmp_path)

        assert path == tmp_path / "doc-chunks-9.3.0.jsonl"

    def test_creates_cache_dir(self, tmp_path):
        cache_dir = tmp_path / "new_subdir"
        assert not cache_dir.exists()
        blob = _layer_blob(_minimal_chunk())
        with patch(
            "vtk_index.artifact.fetcher.urllib.request.urlopen",
            side_effect=_urlopen_side_effect(blob),
        ):
            fetch_from_ghcr("9.6.1", cache_dir=cache_dir)
        assert cache_dir.exists()

    def test_file_content_is_valid_jsonl(self, tmp_path):
        chunks = [_minimal_chunk("vtkActor"), _minimal_chunk("vtkRenderer")]
        blob = _layer_blob(*chunks)
        with patch(
            "vtk_index.artifact.fetcher.urllib.request.urlopen",
            side_effect=_urlopen_side_effect(blob),
        ):
            path = fetch_from_ghcr("9.6.1", cache_dir=tmp_path)

        lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
        assert len(lines) == 2
        parsed = [json.loads(ln) for ln in lines]
        assert parsed[0]["class_names"] == ["vtkActor"]
        assert parsed[1]["class_names"] == ["vtkRenderer"]

    def test_chunk_fields_survive_round_trip(self, tmp_path):
        c = _minimal_chunk("vtkSphereSource")
        c["role"] = "source"
        c["visibility_score"] = 0.85
        c["vtk_version"] = "9.6.1"
        blob = _layer_blob(c)
        with patch(
            "vtk_index.artifact.fetcher.urllib.request.urlopen",
            side_effect=_urlopen_side_effect(blob),
        ):
            path = fetch_from_ghcr("9.6.1", cache_dir=tmp_path)

        obj = json.loads(path.read_text())
        assert obj["role"] == "source"
        assert obj["visibility_score"] == 0.85
        assert obj["vtk_version"] == "9.6.1"

    def test_idempotent_second_call(self, tmp_path):
        blob = _layer_blob(_minimal_chunk())
        with patch(
            "vtk_index.artifact.fetcher.urllib.request.urlopen",
            side_effect=_urlopen_side_effect(blob),
        ):
            path1 = fetch_from_ghcr("9.6.1", cache_dir=tmp_path)

        with patch(
            "vtk_index.artifact.fetcher.urllib.request.urlopen",
            side_effect=RuntimeError("network should not be called"),
        ):
            path2 = fetch_from_ghcr("9.6.1", cache_dir=tmp_path)

        assert path1 == path2

    def test_network_error_raises_runtime_error(self, tmp_path):
        with patch(
            "vtk_index.artifact.fetcher.urllib.request.urlopen",
            side_effect=OSError("connection refused"),
        ):
            with pytest.raises(RuntimeError, match="Failed to pull vtk-index artifact"):
                fetch_from_ghcr("9.6.1", cache_dir=tmp_path)

    def test_default_cache_dir(self):
        assert _CACHE_DIR == Path.home() / ".cache" / "vtk-index"

    def test_default_repository(self):
        assert _DEFAULT_REPOSITORY == "vicentebolea/vtk-index"

    def test_repository_lowercased(self, tmp_path):
        blob = _layer_blob(_minimal_chunk())
        captured_urls = []

        def _open(req_or_url, **_kw):
            url = req_or_url if isinstance(req_or_url, str) else req_or_url.full_url
            captured_urls.append(url)
            data = [
                json.dumps({"token": "t"}).encode(),
                json.dumps({"layers": [{"digest": "sha256:x"}]}).encode(),
                blob,
            ][len(captured_urls) - 1]
            cm = MagicMock()
            cm.__enter__ = lambda s: MagicMock(read=lambda: data)
            cm.__exit__ = MagicMock(return_value=False)
            return cm

        with patch("vtk_index.artifact.fetcher.urllib.request.urlopen", side_effect=_open):
            fetch_from_ghcr("9.6.1", repository="VicenteBolea/Vtk-Index", cache_dir=tmp_path)

        assert all("vicentebolea/vtk-index" in u for u in captured_urls[1:])


# ---------------------------------------------------------------------------
# download CLI command tests
# ---------------------------------------------------------------------------


def _double_urlopen_side_effect(blob: bytes):
    """Six responses: token+manifest+blob for chunks, then token+manifest+blob for storage."""
    storage_blob = _storage_blob()
    responses = iter(
        [
            json.dumps({"token": "t1"}).encode(),
            json.dumps({"layers": [{"digest": "sha256:c"}]}).encode(),
            blob,
            json.dumps({"token": "t2"}).encode(),
            json.dumps({"layers": [{"digest": "sha256:s"}]}).encode(),
            storage_blob,
        ]
    )

    def _open(req_or_url, **_kw):
        data = next(responses)
        cm = MagicMock()
        cm.__enter__ = lambda s: MagicMock(read=lambda: data)
        cm.__exit__ = MagicMock(return_value=False)
        return cm

    return _open


def _storage_blob() -> bytes:
    """Minimal OCI layer blob containing a fake storage directory."""
    import io as _io
    import tarfile as _tf

    buf = _io.BytesIO()
    with _tf.open(fileobj=buf, mode="w:gz") as tar:
        for name, content in [
            ("meta.json", b'{"collections":{}}'),
            ("collection/vtk_docs/storage.sqlite", b"SQLite"),
        ]:
            info = _tf.TarInfo(name=name)
            info.size = len(content)
            tar.addfile(info, _io.BytesIO(content))
    return buf.getvalue()


class TestDownloadCommand:
    def test_downloads_both_by_default(self, tmp_path):
        blob = _layer_blob(_minimal_chunk())
        with patch(
            "vtk_index.artifact.fetcher.urllib.request.urlopen",
            side_effect=_double_urlopen_side_effect(blob),
        ):
            result = runner.invoke(app, ["download", "9.6.1", "--output-dir", str(tmp_path)])
        assert result.exit_code == 0
        assert (tmp_path / "doc-chunks-9.6.1.jsonl").exists()
        assert (tmp_path / "storage-9.6.1").exists()

    def test_no_embedded_skips_storage(self, tmp_path):
        blob = _layer_blob(_minimal_chunk())
        with patch(
            "vtk_index.artifact.fetcher.urllib.request.urlopen",
            side_effect=_urlopen_side_effect(blob),
        ):
            result = runner.invoke(
                app, ["download", "9.6.1", "--output-dir", str(tmp_path), "--no-embedded"]
            )
        assert result.exit_code == 0
        assert (tmp_path / "doc-chunks-9.6.1.jsonl").exists()
        assert not (tmp_path / "storage-9.6.1").exists()

    def test_no_chunks_skips_jsonl(self, tmp_path):
        storage_blob = _storage_blob()
        with patch(
            "vtk_index.artifact.fetcher.urllib.request.urlopen",
            side_effect=_urlopen_side_effect(storage_blob),
        ):
            result = runner.invoke(
                app, ["download", "9.6.1", "--output-dir", str(tmp_path), "--no-chunks"]
            )
        assert result.exit_code == 0
        assert not (tmp_path / "doc-chunks-9.6.1.jsonl").exists()
        assert (tmp_path / "storage-9.6.1").exists()

    def test_success_message_printed(self, tmp_path):
        blob = _layer_blob(_minimal_chunk())
        with patch(
            "vtk_index.artifact.fetcher.urllib.request.urlopen",
            side_effect=_double_urlopen_side_effect(blob),
        ):
            result = runner.invoke(app, ["download", "9.6.1", "--output-dir", str(tmp_path)])
        assert "doc-chunks" in result.output
        assert "storage" in result.output

    def test_network_error_exits_nonzero(self, tmp_path):
        with patch(
            "vtk_index.artifact.fetcher.urllib.request.urlopen",
            side_effect=OSError("no route to host"),
        ):
            result = runner.invoke(app, ["download", "9.6.1", "--output-dir", str(tmp_path)])
        assert result.exit_code != 0
        assert "Error" in result.output

    def test_custom_repository_forwarded(self, tmp_path):
        blob = _layer_blob(_minimal_chunk())
        captured = []
        responses = iter(
            [
                json.dumps({"token": "t"}).encode(),
                json.dumps({"layers": [{"digest": "sha256:x"}]}).encode(),
                blob,
            ]
        )

        def _open(req_or_url, **_kw):
            url = req_or_url if isinstance(req_or_url, str) else req_or_url.full_url
            captured.append(url)
            data = next(responses)
            cm = MagicMock()
            cm.__enter__ = lambda s: MagicMock(read=lambda: data)
            cm.__exit__ = MagicMock(return_value=False)
            return cm

        with patch("vtk_index.artifact.fetcher.urllib.request.urlopen", side_effect=_open):
            runner.invoke(
                app,
                ["download", "9.6.1", "--output-dir", str(tmp_path),
                 "--repository", "myorg/vtk-index", "--no-embedded"],
            )

        assert any("myorg/vtk-index" in u for u in captured)

    def test_help_shows_options(self):
        import re

        result = runner.invoke(app, ["download", "--help"])
        assert result.exit_code == 0
        out = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
        assert "--output-dir" in out
        assert "--repository" in out
