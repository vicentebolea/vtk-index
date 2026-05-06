"""vtk-index — chunking, embedding, and hybrid retrieval over VTK knowledge."""

__version__ = "1.0.0"

from .query.client import Retriever

__all__ = ["Retriever"]
