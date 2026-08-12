"""Tests for vtk_index.artifact.examples_fetcher."""

from __future__ import annotations

import subprocess
from unittest.mock import patch

import pytest

from vtk_index.artifact.examples_fetcher import _EXAMPLES_REPO_URL, fetch_vtk_examples


def _fake_clone_ok(python_dir):
    def _run(cmd, **kwargs):
        if cmd[:2] == ["git", "clone"]:
            python_dir.mkdir(parents=True, exist_ok=True)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    return _run


class TestFetchVtkExamples:
    def test_clones_and_returns_python_dir(self, tmp_path):
        python_dir = tmp_path / "vtk-examples" / "src" / "Python"
        with patch("subprocess.run", side_effect=_fake_clone_ok(python_dir)):
            result = fetch_vtk_examples(cache_dir=tmp_path)
        assert result == python_dir
        assert result.exists()

    def test_cache_hit_skips_clone(self, tmp_path):
        python_dir = tmp_path / "vtk-examples" / "src" / "Python"
        python_dir.mkdir(parents=True)
        with patch("subprocess.run") as mock_run:
            result = fetch_vtk_examples(cache_dir=tmp_path)
            mock_run.assert_not_called()
        assert result == python_dir

    def test_clone_failure_raises_runtime_error(self, tmp_path):
        def _run(cmd, **kwargs):
            raise subprocess.CalledProcessError(1, cmd, stderr="fatal: repo not found")

        with patch("subprocess.run", side_effect=_run):
            with pytest.raises(RuntimeError, match="Failed to clone"):
                fetch_vtk_examples(cache_dir=tmp_path)

    def test_missing_python_dir_after_clone_raises(self, tmp_path):
        def _run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with patch("subprocess.run", side_effect=_run):
            with pytest.raises(RuntimeError, match="missing"):
                fetch_vtk_examples(cache_dir=tmp_path)

    def test_default_repo_url(self):
        assert _EXAMPLES_REPO_URL == "https://gitlab.kitware.com/vtk/vtk-examples.git"

    def test_custom_repo_url_used_in_clone_args(self, tmp_path):
        python_dir = tmp_path / "vtk-examples" / "src" / "Python"
        captured = []

        def _run(cmd, **kwargs):
            captured.append(cmd)
            if cmd[:2] == ["git", "clone"]:
                python_dir.mkdir(parents=True, exist_ok=True)
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with patch("subprocess.run", side_effect=_run):
            fetch_vtk_examples(cache_dir=tmp_path, repo_url="https://example.com/vtk-examples.git")

        assert any("https://example.com/vtk-examples.git" in cmd for cmd in captured)
