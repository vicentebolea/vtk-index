"""Fetch the VTK Examples repo's Python tree, used as the code-chunk corpus."""

import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

_CACHE_DIR = Path.home() / ".cache" / "vtk-index"
_EXAMPLES_REPO_URL = "https://gitlab.kitware.com/vtk/vtk-examples.git"


def fetch_vtk_examples(
    cache_dir: Path = _CACHE_DIR,
    repo_url: str = _EXAMPLES_REPO_URL,
) -> Path:
    """Shallow-clone the VTK Examples repo and return its ``src/Python`` directory.

    Uses a blobless, sparse, depth-1 clone so only the Python examples tree is
    actually downloaded. Cached by directory presence across calls.

    Args:
        cache_dir: Local cache directory.
        repo_url: Git URL of the vtk-examples repository.

    Returns:
        Absolute ``Path`` to the checked-out ``src/Python`` directory.

    Raises:
        RuntimeError: If the clone fails or the expected directory is missing.
    """
    examples_dir = cache_dir / "vtk-examples"
    python_dir = examples_dir / "src" / "Python"
    if python_dir.exists():
        return python_dir

    cache_dir.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(examples_dir, ignore_errors=True)
    try:
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--filter=blob:none",
                "--sparse",
                repo_url,
                str(examples_dir),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "-C", str(examples_dir), "sparse-checkout", "set", "src/Python"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        shutil.rmtree(examples_dir, ignore_errors=True)
        raise RuntimeError(f"Failed to clone {repo_url}: {exc.stderr}") from exc

    if not python_dir.is_dir():
        shutil.rmtree(examples_dir, ignore_errors=True)
        raise RuntimeError(f"Cloned {repo_url} but {python_dir} is missing")

    logger.info("Fetched VTK Examples to %s", python_dir)
    return python_dir
