# vtk-index

Chunking, embedding, and hybrid retrieval over the VTK knowledge artifact.
Takes the JSONL file produced by `vtk-knowledge`, splits it into searchable
chunks, embeds them with dense and sparse models, and stores them in Qdrant.
The `Retriever` class exposes hybrid search (dense + BM25 Reciprocal Rank
Fusion) over the resulting collections.

This is Layer 2 of a four-layer stack:

```
vtk-knowledge  - schema, artifact, in-memory index
vtk-index      - chunking, dense+sparse embeddings, Qdrant  (this repo)
vtk-validate   - AST-based VTK code validation
vtk-mcp        - MCP gateway, 25 tools over stdio/http
```

Each layer only depends on layers below it.

## Install

`vtk-index` is not yet on PyPI.

### In an external project (consuming vtk-index)

Because `vtk-index` depends on `vtk-knowledge` (also not on PyPI), declare
both as git sources in your `pyproject.toml` before adding:

```toml
[tool.uv.sources]
vtk-knowledge = {git = "https://github.com/vicentebolea/vtk-knowledge.git"}
vtk-index     = {git = "https://github.com/vicentebolea/vtk-index.git"}
```

```bash
uv add vtk-index
```

With pip:

```bash
pip install "vtk-knowledge @ git+https://github.com/vicentebolea/vtk-knowledge.git"
pip install "vtk-index @ git+https://github.com/vicentebolea/vtk-index.git"
```

### For development on vtk-index itself

```bash
git clone https://github.com/vicentebolea/vtk-index
cd vtk-index
uv venv
uv pip install "vtk-knowledge @ git+https://github.com/vicentebolea/vtk-knowledge.git"
uv pip install -e ".[dev]"
pytest tests/
uv run vtk-index   # run the CLI (venv does not need to be activated)
```

If working inside the monorepo workspace, `uv add --dev .` installs vtk-index
into the shared workspace venv. Use `uv run vtk-index` to invoke the CLI — the
workspace venv is not automatically on `PATH`.

## CLI

### download -- get pre-built artifacts (no Qdrant or VTK needed)

Downloads both the doc-chunks JSONL and the pre-built embedded Qdrant storage
to `~/.cache/vtk-index/` by default.

```bash
# download both artifacts (recommended — enables instant search)
vtk-index download 9.6.1

# write to a specific directory instead of ~/.cache/vtk-index/
vtk-index download 9.6.1 -o ./artifacts/

# only the embedded storage (skips doc-chunks JSONL)
vtk-index download 9.6.1 --no-chunks

# only the doc-chunks JSONL (skips embedded storage)
vtk-index download 9.6.1 --no-embedded

# pull from a different ghcr.io repository
vtk-index download 9.6.1 -r myorg/vtk-index
```

Pulls from `ghcr.io/{repository}` via the OCI HTTP API — no docker or podman
required. Two artifacts are cached in `~/.cache/vtk-index/`:

| File | Tag | Use with |
|---|---|---|
| `doc-chunks-9.6.1.jsonl` | `9.6.1` | `search --chunks` (embeds on the fly) |
| `storage-9.6.1/` | `9.6.1-embedded` | `search --vtk-version` (instant, no embedding) |

### search -- query the index from the command line

Three backends, pick the one that suits your setup:

```bash
# fastest: use pre-built embedded storage downloaded by 'vtk-index download'
vtk-index search --vtk-version 9.6.1 "sphere source"
vtk-index search --vtk-version 9.6.1 "read STL file" --role source -n 5
vtk-index search --vtk-version 9.6.1 "isosurface contour" --min-visibility 0.7
vtk-index search --vtk-version 9.6.1 "render window" --collection code
vtk-index search --vtk-version 9.6.1 "sphere source" --json

# alternative: load doc-chunks and embed on the fly (~30s startup)
vtk-index search "sphere source" --chunks doc-chunks-9.6.1.jsonl

# alternative: running Qdrant server
vtk-index search "sphere source" --qdrant-url http://myhost:6333
```

`--vtk-version` downloads the embedded storage on first use and caches it.
Subsequent queries are instant — no server, no embedding models at query time.

### chunk -- split a knowledge artifact into Qdrant-ready chunks

```bash
vtk-index chunk vtk-knowledge-9.6.1.jsonl -o chunks/
# writes chunks/doc-chunks.jsonl
```

### index -- embed chunks and upload to Qdrant

```bash
# write to a local embedded storage directory (no server needed)
vtk-index index --doc-chunks chunks/doc-chunks.jsonl --path ./storage/

# upload to a running Qdrant server
vtk-index index --doc-chunks chunks/doc-chunks.jsonl --qdrant-url http://localhost:6333
```

### snapshot -- package server collections as a tarball

```bash
vtk-index snapshot --vtk-version 9.6.1 -o .
# writes vtk-index-9.6.1.snapshot.tar.gz
```

### build -- chunk + index + snapshot in one step

```bash
vtk-index build vtk-knowledge-9.6.1.jsonl \
  --qdrant-url http://localhost:6333 \
  --output-dir ./artifacts/
```

## Python API

```python
from vtk_index import Retriever

# zero-config: downloads pre-built storage on first call, instant after
retriever = Retriever.from_artifact("9.6.1")
chunks = retriever.search_docs("sphere source", k=5)
for c in chunks:
    print(c.class_names, c.chunk_type, c.content[:120])

# explicit in-memory (requires indexing first)
retriever = Retriever()

# explicit server
retriever = Retriever(qdrant_url="http://localhost:6333")

# explicit local storage path
retriever = Retriever(qdrant_path="./storage-9.6.1")

# hybrid search with filter
from vtk_index.query.filters import PayloadFilter

chunks = retriever.hybrid_search(
    "read STL file",
    collection="vtk_docs",
    k=10,
    filters=PayloadFilter().by_role("source").min_visibility(0.6),
)
```

## Chunk types

Each `VTKDocRecord` is split into one or more `Chunk` objects stored in Qdrant:

| `chunk_type` | Content |
|---|---|
| `class_overview` | class name, module, synopsis, class doc excerpt, action phrase |
| `method_doc` | method signature(s) + docstring for each semantic method |
| `inheritance` | MRO chain as a single sentence |
| `pipeline_example` | whole-file or extracted pipeline segment from example scripts |
| `query_example` | extracted query/search segment from example scripts |
| `constructor` | constructor signatures |
| `property_group` | grouped property getter/setter pairs |

## Payload filters

```python
from vtk_index.query.filters import PayloadFilter, build_filter

# builder pattern
f = PayloadFilter().by_role("source").min_visibility(0.7).build()

# plain dict (same result)
f = build_filter({"role": "source", "visibility_score": {"gte": 0.7}})
```

## CI build workflow

`workflow_dispatch` in `.github/workflows/build-artifact.yml`:

1. **Actions -> Build Chunks Artifact -> Run workflow**
2. Set the VTK version (must already have a `vtk-knowledge` artifact on ghcr.io)

The workflow:
1. Downloads the `vtk-knowledge` JSONL artifact from `ghcr.io/vicentebolea/vtk-knowledge`
2. Runs `vtk-index chunk` to produce `doc-chunks.jsonl`
3. Runs `vtk-index index --path storage/` to build the embedded Qdrant storage
4. Pushes two FROM-scratch OCI images to ghcr.io:
   - `ghcr.io/vicentebolea/vtk-index:{vtk_version}` — doc-chunks JSONL
   - `ghcr.io/vicentebolea/vtk-index:{vtk_version}-embedded` — pre-built Qdrant storage

## Code layout

```
src/vtk_index/
  chunking/base.py         # Chunk model and ChunkType enum (Pydantic)
  chunking/doc_chunker.py  # VTKDocRecord -> list[Chunk]
  chunking/code_chunker.py # Python example scripts -> list[Chunk]
  embedding/dense.py       # DenseEmbedder wrapping sentence-transformers
  embedding/sparse.py      # SparseEmbedder wrapping FastEmbed BM25
  query/client.py          # Retriever: from_artifact / search_docs / search_code / hybrid_search
  query/filters.py         # PayloadFilter builder and build_filter helper
  pipeline/cli.py          # Typer CLI: download / search / chunk / index / snapshot / build
  artifact/fetcher.py      # fetch_from_ghcr / fetch_embedded_storage: OCI pull without docker
  artifact/snapshot.py     # save_snapshot / load_snapshot for Qdrant server tarballs
```

## Related repos

| Repo | Layer | What it does |
|---|---|---|
| [vtk-knowledge](https://github.com/vicentebolea/vtk-knowledge) | 1 | Schema, artifact, index |
| **vtk-index** (here) | 2 | Chunking, embeddings, Qdrant |
| [vtk-validate](https://github.com/vicentebolea/vtk-validate) | 3 | AST validation of VTK code |
| [vtk-mcp](https://github.com/Kitware/vtk-mcp) | 4 | MCP gateway for LLM assistants |
