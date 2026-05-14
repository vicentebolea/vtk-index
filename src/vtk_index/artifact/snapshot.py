"""Qdrant snapshot packaging and loading for vtk-index artifacts."""

from __future__ import annotations

import logging
import tarfile
from pathlib import Path

from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)

_DOCS_COLLECTION = "vtk_docs"
_CODE_COLLECTION = "vtk_code"


def save_snapshot(
    qdrant_url: str,
    vtk_version: str,
    output_dir: Path,
) -> Path:
    """Create and package a Qdrant snapshot for both collections.

    Args:
        qdrant_url: URL of the running Qdrant instance.
        vtk_version: VTK version string used in the artifact filename.
        output_dir: Directory to write the snapshot tarball.

    Returns:
        Path to the ``vtk-index-{vtk_version}.snapshot.tar.gz`` file.
    """
    client = QdrantClient(url=qdrant_url)
    output_dir.mkdir(parents=True, exist_ok=True)
    tarball = output_dir / f"vtk-index-{vtk_version}.snapshot.tar.gz"

    with tarfile.open(tarball, "w:gz") as tar:
        for collection in (_DOCS_COLLECTION, _CODE_COLLECTION):
            snap = client.create_snapshot(collection_name=collection)
            snap_path = Path(snap.name)
            if snap_path.exists():
                tar.add(snap_path, arcname=snap_path.name)
                logger.info("Added snapshot for %s: %s", collection, snap_path.name)

    logger.info("Snapshot tarball written to %s", tarball)
    return tarball


def load_snapshot(
    tarball: Path,
    qdrant_url: str,
    vtk_version: str = "",
) -> None:
    """Extract a snapshot tarball and restore collections into Qdrant.

    Args:
        tarball: Path to ``vtk-index-{version}.snapshot.tar.gz``.
        qdrant_url: URL of the target Qdrant instance.
        vtk_version: Optional version string for logging.
    """
    client = QdrantClient(url=qdrant_url)
    with tarfile.open(tarball, "r:gz") as tar:
        for member in tar.getmembers():
            collection = _collection_from_name(member.name)
            if collection is None:
                continue
            tmp = Path("/tmp") / member.name
            tar.extract(member, path="/tmp")
            client.recover_snapshot(collection_name=collection, location=str(tmp))
            logger.info("Restored snapshot for %s from %s", collection, member.name)


def _collection_from_name(name: str) -> str | None:
    if "vtk_docs" in name:
        return _DOCS_COLLECTION
    if "vtk_code" in name:
        return _CODE_COLLECTION
    return None
