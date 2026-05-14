"""CLI for the vtk-index build pipeline.

Commands:
    vtk-index chunk     -- JSONL knowledge artifact -> doc-chunks.jsonl + code-chunks.jsonl
    vtk-index index     -- chunk JSONLs -> embed + upload to Qdrant
    vtk-index snapshot  -- Qdrant -> snapshot tarball
    vtk-index build     -- convenience wrapper: chunk -> index -> snapshot
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

app = typer.Typer(
    name="vtk-index",
    help="vtk-index build pipeline: chunk, embed, index, snapshot.",
    no_args_is_help=True,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


@app.command()
def chunk(
    knowledge_artifact: Path = typer.Argument(..., help="Path to vtk-knowledge JSONL artifact."),
    output_dir: Path = typer.Option(Path("."), "--output-dir", "-o"),
    vtk_version: str = typer.Option("", "--vtk-version"),
) -> None:
    """Chunk a knowledge JSONL into doc-chunks and code-chunks."""
    try:
        from vtk_knowledge import VTKAPIIndex

        from ..chunking.doc_chunker import chunk_record
    except ImportError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    try:
        index = VTKAPIIndex.from_jsonl(knowledge_artifact)

        doc_chunks = []
        for record in index.classes.values():
            for c in chunk_record(record):
                doc_chunks.append(c.model_dump())

        output_dir.mkdir(parents=True, exist_ok=True)
        doc_out = output_dir / "doc-chunks.jsonl"
        with open(doc_out, "w") as f:
            for c in doc_chunks:
                f.write(json.dumps(c) + "\n")

        typer.echo(f"Wrote {len(doc_chunks)} doc chunks to {doc_out}")
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def index(
    doc_chunks: Path = typer.Option(Path("doc-chunks.jsonl"), "--doc-chunks"),
    code_chunks: Path = typer.Option(Path("code-chunks.jsonl"), "--code-chunks"),
    qdrant_url: str = typer.Option("http://localhost:6333", "--qdrant-url"),
) -> None:
    """Embed chunks and upload to Qdrant."""
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import PointStruct, SparseVector

        from ..embedding.dense import DenseEmbedder
        from ..embedding.sparse import SparseEmbedder
    except ImportError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    try:
        client = QdrantClient(url=qdrant_url)
        dense = DenseEmbedder()
        sparse = SparseEmbedder()

        for chunk_file, collection in [
            (doc_chunks, "vtk_docs"),
            (code_chunks, "vtk_code"),
        ]:
            if not chunk_file.exists():
                typer.echo(f"Skipping {chunk_file} (not found)", err=True)
                continue

            chunks = []
            with open(chunk_file) as f:
                for line in f:
                    if line.strip():
                        chunks.append(json.loads(line))

            _ensure_collection(client, collection)

            points = []
            for i, c in enumerate(chunks):
                content = c.get("content", "")
                dense_vec = dense.encode(content)
                sparse_emb = sparse.embed(content)
                points.append(
                    PointStruct(
                        id=i,
                        vector={
                            "content": dense_vec,
                            "bm25": SparseVector(
                                indices=sparse_emb.indices.tolist(),
                                values=sparse_emb.values.tolist(),
                            ),
                        },
                        payload=c,
                    )
                )

            client.upsert(collection_name=collection, points=points)
            typer.echo(f"Indexed {len(points)} chunks into {collection}")
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def snapshot(
    qdrant_url: str = typer.Option("http://localhost:6333", "--qdrant-url"),
    vtk_version: str = typer.Option("unknown", "--vtk-version"),
    output_dir: Path = typer.Option(Path("."), "--output-dir", "-o"),
) -> None:
    """Package current Qdrant collections as a snapshot tarball."""
    try:
        from ..artifact.snapshot import save_snapshot

        tarball = save_snapshot(qdrant_url, vtk_version, output_dir)
        typer.echo(f"Snapshot written to {tarball}")
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def build(
    knowledge_artifact: Path = typer.Argument(..., help="Path to vtk-knowledge JSONL artifact."),
    qdrant_url: str = typer.Option("http://localhost:6333", "--qdrant-url"),
    output_dir: Path = typer.Option(Path("."), "--output-dir", "-o"),
    vtk_version: str = typer.Option("", "--vtk-version"),
) -> None:
    """Run chunk -> index -> snapshot pipeline."""
    try:
        import tempfile

        from vtk_knowledge import VTKAPIIndex

        from ..artifact.snapshot import save_snapshot
        from ..chunking.doc_chunker import chunk_record
    except ImportError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            ki = VTKAPIIndex.from_jsonl(knowledge_artifact)
            effective_version = vtk_version or ki.vtk_version

            doc_out = tmp / "doc-chunks.jsonl"
            with open(doc_out, "w") as f:
                for record in ki.classes.values():
                    for c in chunk_record(record):
                        f.write(json.dumps(c.model_dump()) + "\n")

            _run_index(str(doc_out), "", qdrant_url)

            output_dir.mkdir(parents=True, exist_ok=True)
            tarball = save_snapshot(qdrant_url, effective_version, output_dir)

        typer.echo(f"Build complete. Snapshot at {tarball}")
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


def _ensure_collection(client, collection: str) -> None:
    from qdrant_client.models import (
        Distance,
        SparseVectorParams,
        VectorParams,
    )

    existing = {c.name for c in client.get_collections().collections}
    if collection not in existing:
        client.create_collection(
            collection_name=collection,
            vectors_config={"content": VectorParams(size=384, distance=Distance.COSINE)},
            sparse_vectors_config={"bm25": SparseVectorParams()},
        )


def _run_index(doc_chunks_path: str, code_chunks_path: str, qdrant_url: str) -> None:
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct, SparseVector

    from ..embedding.dense import DenseEmbedder
    from ..embedding.sparse import SparseEmbedder

    client = QdrantClient(url=qdrant_url)
    dense = DenseEmbedder()
    sparse = SparseEmbedder()

    for path, collection in [(doc_chunks_path, "vtk_docs"), (code_chunks_path, "vtk_code")]:
        if not path or not Path(path).exists():
            continue
        chunks = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))
        _ensure_collection(client, collection)
        points = []
        for i, c in enumerate(chunks):
            content = c.get("content", "")
            dv = dense.encode(content)
            se = sparse.embed(content)
            points.append(
                PointStruct(
                    id=i,
                    vector={
                        "content": dv,
                        "bm25": SparseVector(
                            indices=se.indices.tolist(),
                            values=se.values.tolist(),
                        ),
                    },
                    payload=c,
                )
            )
        client.upsert(collection_name=collection, points=points)


if __name__ == "__main__":
    app()
