"""Pydantic Chunk model — the unit stored in Qdrant."""

from enum import Enum

from pydantic import BaseModel


class ChunkType(str, Enum):
    CLASS_OVERVIEW = "class_overview"
    CONSTRUCTOR = "constructor"
    PROPERTY_GROUP = "property_group"
    METHOD_DOC = "method_doc"
    PIPELINE_EXAMPLE = "pipeline_example"
    QUERY_EXAMPLE = "query_example"
    INHERITANCE = "inheritance"


class Chunk(BaseModel):
    chunk_id: str
    chunk_type: ChunkType
    content: str
    class_names: list[str] = []
    module_names: list[str] = []
    role: str | None = None
    input_datatype: str | None = None
    output_datatype: str | None = None
    visibility_score: float | None = None
    source: str = "vtk-knowledge"
    source_path: str = ""
    vtk_version: str = ""
