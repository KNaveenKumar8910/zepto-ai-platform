from pydantic import BaseModel, Field
from typing import List

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Incoming user query")

class QueryResponse(BaseModel):
    answer: str = Field(..., description="Final answer generated or canned response")
    sources: List[str] = Field(default_factory=list, description="List of source document IDs used")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0 and 1")
