# vtk-index

Chunking, embedding, and hybrid retrieval over the VTK knowledge artifact.
Takes the JSONL file produced by `vtk-knowledge`, splits it into
searchable chunks, embeds them with dense and sparse models, and stores
them in Qdrant. The `Retriever` class exposes hybrid search (dense + BM25
Reciprocal Rank Fusion) over the resulting collections.

This is Layer 2 of a four-layer stack:

```
vtk-knowledge  - schema, artifact, in-memory index
vtk-index      - chunking, dense+sparse embeddings, Qdrant  (this repo)
vtk-validate   - AST-based VTK code validation
vtk-mcp        - MCP gateway, 25 tools over stdio/http
```

Each layer only depends on layers below it.

## Install

`vtk-index` is not yet on PyPI. Install from GitHub using uv:

```bash
# in your project's pyproject.toml, add the git source overrides:
# [tool.uv.sources]
# vtk-knowledge = {git = "https://github.com/vicentebolea/vtk-knowledge.git"}
# vtk-index    = {git = "https://github.com/vicentebolea/vtk-index.git"}

uv add "vtk-index @ git+https://github.com/vicentebolea/vtk-index.git"
```

Because `vtk-index` depends on `vtk-knowledge` (also not on PyPI), you need
to declare both git sources in your `pyproject.toml` so uv can resolve the
full dependency graph:

```toml
[tool.uv.sources]
vtk-knowledge = {git = "https://github.com/vicentebolea/vtk-knowledge.git"}
vtk-index     = {git = "https://github.com/vicentebolea/vtk-index.git"}
```

Then:

```bash
uv add vtk-index
uv sync
```

With pip (direct git install):

```bash
pip install "vtk-knowledge @ git+https://github.com/vicentebolea/vtk-knowledge.git"
pip install "vtk-index @ git+https://github.com/vicentebolea/vtk-index.git"
```

For development on vtk-index itself:

```bash
git clone https://github.com/vicentebolea/vtk-index
cd vtk-index
uv venv
uv pip install "vtk-knowledge @ git+https://github.com/vicentebolea/vtk-knowledge.git"
uv pip install -e ".[dev]"
pytest tests/
```

## CLI

### download -- get a pre-built chunks artifact (no Qdrant or VTK needed)

```bash
# download doc-chunks for VTK 9.6.1 to the current directory
vtk-index download 9.6.1

# write to a specific directory
vtk-index download 9.6.1 -o ./artifacts/

# pull from a different ghcr.io repository
vtk-index download 9.6.1 -r myorg/vtk-index
```

Pulls `doc-chunks-9.6.1.jsonl` from `ghcr.io/{repository}:{vtk_version}` via
the OCI HTTP API — no docker or podman required. The file is cached in
`~/.cache/vtk-index/` so repeated calls are instant.

### search -- query the index from the command line

```bash
# basic search against the docs collection
vtk-index search "sphere source"

# limit results and filter by role
vtk-index search "read STL file" --role source -n 5

# filter by minimum visibility score
vtk-index search "isosurface contour" --min-visibility 0.7

# search code examples instead of docs
vtk-index search "render window pipeline" --collection code

# machine-readable JSON output
vtk-index search "sphere source" --json

# connect to a persistent Qdrant instance instead of in-memory
vtk-index search "mapper poly data" --qdrant-url http://myhost:6333
```

### chunk -- split a knowledge artifact into Qdrant-ready chunks

```bash
vtk-index chunk vtk-knowledge-9.6.1.jsonl -o chunks/
# writes chunks/doc-chunks.jsonl
```

### index -- embed chunks and upload to Qdrant

```bash
vtk-index index --doc-chunks chunks/doc-chunks.jsonl   # in-memory by default
vtk-index index --doc-chunks chunks/doc-chunks.jsonl --qdrant-url http://localhost:6333
```

### snapshot -- package collections as a portable tarball

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

retriever = Retriever()  # in-memory by default; pass qdrant_url for a persistent server

# hybrid search over the docs collection
chunks = retriever.search_docs("sphere source output poly data", k=5)
for c in chunks:
    print(c.class_names, c.chunk_type, c.content[:120])

# search over indexed code examples
chunks = retriever.search_code("render window interactor pipeline", k=5)

# explicit hybrid search on any collection with a filter
from vtk_index.query.filters import PayloadFilter

filt = PayloadFilter().by_role("source").min_visibility(0.6)
chunks = retriever.hybrid_search(
    "read STL file",
    collection="vtk_docs",
    k=10,
    filters=filt,
)
```

## Chunk types

Each `VTKDocRecord` from the knowledge artifact is split into one or more
`Chunk` objects stored in Qdrant:

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

`PayloadFilter` and `build_filter` convert plain dicts or builder chains
into Qdrant `Filter` objects:

```python
from vtk_index.query.filters import PayloadFilter

# all source-role classes with visibility >= 0.7
f = PayloadFilter().by_role("source").min_visibility(0.7).build()

# plain dict (same result)
from vtk_index.query.filters import build_filter
f = build_filter({"role": "source", "visibility_score": {"gte": 0.7}})
```

## Code layout

```
src/vtk_index/
  chunking/base.py         # Chunk model and ChunkType enum (Pydantic)
  chunking/doc_chunker.py  # VTKDocRecord -> list[Chunk]
  chunking/code_chunker.py # Python example scripts -> list[Chunk]
  embedding/dense.py       # DenseEmbedder wrapping sentence-transformers
  embedding/sparse.py      # SparseEmbedder wrapping FastEmbed BM25
  query/client.py          # Retriever: search_docs / search_code / hybrid_search
  query/filters.py         # PayloadFilter builder and build_filter helper
  pipeline/cli.py          # Typer CLI: download / search / chunk / index / snapshot / build
  artifact/fetcher.py      # fetch_from_ghcr: OCI pull without docker/podman
  artifact/snapshot.py     # save_snapshot / load_snapshot for Qdrant tarballs
```

## Related repos

| Repo | Layer | What it does |
|---|---|---|
| [vtk-knowledge](https://github.com/vicentebolea/vtk-knowledge) | 1 | Schema, artifact, index |
| **vtk-index** (here) | 2 | Chunking, embeddings, Qdrant |
| [vtk-validate](https://github.com/vicentebolea/vtk-validate) | 3 | AST validation of VTK code |
| [vtk-mcp](https://github.com/Kitware/vtk-mcp) | 4 | MCP gateway for LLM assistants |
