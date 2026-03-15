from pydantic import BaseModel
from typing import List, Dict


class RagRequest(BaseModel):
    query: str


class Citation(BaseModel):
    text: str
    metadata: Dict


class RagResponse(BaseModel):
    query: str
    answer: str
    citations: List[Citation]
    explanation: str | None = None
    retrieval_time: float | None = None