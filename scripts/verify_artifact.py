#!/usr/bin/env python3
"""Fail loudly if an embedded Qdrant storage is missing collections or empty.

Used by build-artifact.yml right after `vtk-index index`, before the
storage is packaged into an image and pushed to ghcr.io.
"""

from __future__ import annotations

import sys

from qdrant_client import QdrantClient

REQUIRED_COLLECTIONS = ("vtk_docs", "vtk_code")


def main(storage_path: str) -> int:
    client = QdrantClient(path=storage_path)
    existing = {c.name for c in client.get_collections().collections}

    missing = [name for name in REQUIRED_COLLECTIONS if name not in existing]
    if missing:
        print(f"Missing collections in {storage_path}: {missing}", file=sys.stderr)
        return 1

    empty = []
    for name in REQUIRED_COLLECTIONS:
        count = client.count(name).count
        print(f"{name}: {count} points")
        if count == 0:
            empty.append(name)

    if empty:
        print(f"Empty collections in {storage_path}: {empty}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <storage_path>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
