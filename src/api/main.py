# v1.0.0 - 2026-06-09 - Versão inicial
import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from src.api.schemas import QueryRequest, QueryResponse, HealthResponse
from src.retrieval.retriever import Retriever
from src.generation.generator import Generator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BM25_INDEX_PATH = os.environ.get("BM25_INDEX_PATH", "bm25_index.pkl")

retriever: Retriever | None = None
generator: Generator | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever, generator
    logger.info("Inicializando componentes RAG")
    retriever = Retriever(bm25_index_path=BM25_INDEX_PATH)
    generator = Generator()
    logger.info("API pronta")
    yield
    logger.info("API encerrada")


app = FastAPI(title="Vertex RAG Pipeline", version="1.0.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if retriever is None or generator is None:
        raise HTTPException(status_code=503, detail="Componentes ainda não inicializados")

    try:
        logger.info("Query recebida: '%s'", request.question)
        chunks = retriever.retrieve(request.question)
        result = generator.generate(request.question, chunks)
        return QueryResponse(**result)
    except Exception:
        logger.exception("Erro ao processar query: '%s'", request.question)
        raise HTTPException(status_code=500, detail="Erro interno ao processar a pergunta")
