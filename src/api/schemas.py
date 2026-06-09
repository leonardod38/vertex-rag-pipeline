# v1.0.0 - 2026-06-09 - Versão inicial
from pydantic import BaseModel, field_validator


class QueryRequest(BaseModel):
    question: str

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("A pergunta não pode ser vazia")
        return v.strip()


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    chunks_used: int


class HealthResponse(BaseModel):
    status: str
