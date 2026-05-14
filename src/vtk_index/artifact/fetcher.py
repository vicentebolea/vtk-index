"""Download a pre-built vtk-index doc-chunks artifact by VTK version."""

import io
import json
import logging
import tarfile
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

_CACHE_DIR = Path.home() / ".cache" / "vtk-index"
_GHCR_BASE = "https://ghcr.io"
_DEFAULT_REPOSITORY = "vicentebolea/vtk-index"
_EMBEDDED_TAG_SUFFIX = "-embedded"


def _ghcr_token(repository: str) -> str:
    url = f"{_GHCR_BASE}/token?scope=repository:{repository}:pull&service=ghcr.io"
    with urllib.request.urlopen(url) as resp:
        return json.loads(resp.read())["token"]


def fetch_from_ghcr(
    vtk_version: str,
    repository: str = _DEFAULT_REPOSITORY,
    cache_dir: Path = _CACHE_DIR,
) -> Path:
    """Download the doc-chunks JSONL by pulling it from a ghcr.io OCI image.

    The image at ``ghcr.io/{repository}:{vtk_version}`` is a FROM-scratch image
    whose single layer is a tar containing ``/doc-chunks.jsonl``.  This function
    speaks the OCI Distribution Spec HTTP API directly — no docker or podman
    required.

    Args:
        vtk_version: VTK version tag, e.g. ``"9.6.1"``.
        repository: ghcr.io repository path (owner/name), lower-cased.
        cache_dir: Local cache directory.

    Returns:
        Absolute ``Path`` to the local JSONL file.

    Raises:
        RuntimeError: If any network or extraction step fails.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_path = cache_dir / f"doc-chunks-{vtk_version}.jsonl"
    if local_path.exists():
        return local_path

    repo = repository.lower()
    try:
        token = _ghcr_token(repo)
        headers = {"Authorization": f"Bearer {token}"}

        manifest_url = f"{_GHCR_BASE}/v2/{repo}/manifests/{vtk_version}"
        req = urllib.request.Request(
            manifest_url,
            headers={
                **headers,
                "Accept": (
                    "application/vnd.oci.image.manifest.v1+json,"
                    "application/vnd.docker.distribution.manifest.v2+json"
                ),
            },
        )
        with urllib.request.urlopen(req) as resp:
            manifest = json.loads(resp.read())

        digest = manifest["layers"][0]["digest"]
        logger.info("Pulling layer %s from ghcr.io/%s:%s", digest[:19], repo, vtk_version)

        blob_url = f"{_GHCR_BASE}/v2/{repo}/blobs/{digest}"
        req = urllib.request.Request(blob_url, headers=headers)
        with urllib.request.urlopen(req) as resp:
            blob = resp.read()

        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:*") as tar:
            member = next(m for m in tar.getmembers() if m.name.endswith(".jsonl"))
            extracted = tar.extractfile(member)
            if extracted is None:
                raise RuntimeError("JSONL member is not a regular file in the layer tar")
            local_path.write_bytes(extracted.read())

    except Exception as exc:
        raise RuntimeError(
            f"Failed to pull vtk-index artifact from ghcr.io/{repo}:{vtk_version}: {exc}"
        ) from exc

    logger.info("Saved artifact to %s", local_path)
    return local_path


def fetch_embedded_storage(
    vtk_version: str,
    repository: str = _DEFAULT_REPOSITORY,
    cache_dir: Path = _CACHE_DIR,
) -> Path:
    """Download the pre-built embedded Qdrant storage for *vtk_version*.

    The image at ``ghcr.io/{repository}:{vtk_version}-embedded`` is a
    FROM-scratch image whose single layer contains the ``QdrantClient(path=...)``
    storage directory.  The storage is extracted into *cache_dir* and can be
    opened directly with ``QdrantClient(path=str(returned_path))``.

    No Qdrant server, no embedding models, and no restore step required.

    Args:
        vtk_version: VTK version tag, e.g. ``"9.6.1"``.
        repository: ghcr.io repository path (owner/name), lower-cased.
        cache_dir: Local cache directory.

    Returns:
        Absolute ``Path`` to the local storage directory.

    Raises:
        RuntimeError: If any network or extraction step fails.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    storage_path = cache_dir / f"storage-{vtk_version}"
    if storage_path.exists():
        return storage_path

    repo = repository.lower()
    tag = f"{vtk_version}{_EMBEDDED_TAG_SUFFIX}"
    try:
        token = _ghcr_token(repo)
        headers = {"Authorization": f"Bearer {token}"}

        manifest_url = f"{_GHCR_BASE}/v2/{repo}/manifests/{tag}"
        req = urllib.request.Request(
            manifest_url,
            headers={
                **headers,
                "Accept": (
                    "application/vnd.oci.image.manifest.v1+json,"
                    "application/vnd.docker.distribution.manifest.v2+json"
                ),
            },
        )
        with urllib.request.urlopen(req) as resp:
            manifest = json.loads(resp.read())

        digest = manifest["layers"][0]["digest"]
        logger.info("Pulling embedded storage layer %s from ghcr.io/%s:%s", digest[:19], repo, tag)

        blob_url = f"{_GHCR_BASE}/v2/{repo}/blobs/{digest}"
        req = urllib.request.Request(blob_url, headers=headers)
        with urllib.request.urlopen(req) as resp:
            blob = resp.read()

        storage_path.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:*") as tar:
            for member in tar.getmembers():
                name = member.name.lstrip("./")
                if not name:
                    continue
                member.name = name
                tar.extract(member, path=storage_path, filter="data")

    except Exception as exc:
        import shutil

        shutil.rmtree(storage_path, ignore_errors=True)
        raise RuntimeError(
            f"Failed to pull vtk-index embedded storage from ghcr.io/{repo}:{tag}: {exc}"
        ) from exc

    logger.info("Saved embedded storage to %s", storage_path)
    return storage_path
