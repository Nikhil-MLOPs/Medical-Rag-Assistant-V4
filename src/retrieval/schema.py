from pydantic import BaseModel
from typing import Dict, List


class RetrievalResult(BaseModel):
    text: str
    metadata: Dict
    score: float


class RetrievalResponse(BaseModel):
    query: str
    results: List[RetrievalResult]