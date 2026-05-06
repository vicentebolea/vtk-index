"""Pydantic Chunk model — the unit stored in Qdrant."""

from enum import Enum
from typing import Optional

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
    role: Optional[str] = None
    input_datatype: Optional[str] = None
    output_datatype: Optional[str] = None
    visibility_score: Optional[float] = None
    source: str = "vtk-knowledge"
    source_path: str = ""
    vtk_version: str = ""
