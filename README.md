# vtk-index

Chunking, embedding, and hybrid retrieval for VTK knowledge artifacts.

## Installation

```bash
pip install vtk-index
```

## Usage

```python
from vtk_index import Retriever

retriever = Retriever(qdrant_url="http://localhost:6333", vtk_version="9.3.0")
chunks = retriever.search_docs("sphere source pipeline example")
for chunk in chunks:
    print(chunk.content[:200])
```

## Build pipeline

```bash
vtk-index chunk vtk-knowledge-9.3.0.jsonl -o chunks/
vtk-index index --doc-chunks chunks/doc-chunks.jsonl --qdrant-url http://localhost:6333
vtk-index snapshot --vtk-version 9.3.0 -o .
# or all at once:
vtk-index build vtk-knowledge-9.3.0.jsonl --qdrant-url http://localhost:6333
```

## Architecture

Part of the [VTK LLM tooling](https://github.com/vicentebolea/vtk-llm-architecture) stack:

- [vtk-knowledge](https://github.com/vicentebolea/vtk-knowledge) — Layer 1: knowledge schema + artifact
- **vtk-index** (this repo) — Layer 2: chunking + retrieval
- [vtk-validate](https://github.com/vicentebolea/vtk-validate) — Layer 3: AST validation
- [vtk-mcp](https://github.com/vicentebolea/vtk-mcp) — Layer 4: MCP gateway
