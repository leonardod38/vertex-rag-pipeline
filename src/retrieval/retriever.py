# v1.0.0 - 2026-06-09 - Versão inicial
import logging
import os
from src.ingestion.embedder import Embedder
from src.ingestion.pinecone_store import PineconeStore
from src.ingestion.bm25_indexer import BM25Indexer
from src.retrieval.hybrid_search import reciprocal_rank_fusion
from src.retrieval.reranker import Reranker

logger = logging.getLogger(__name__)

TOP_K_RETRIEVAL = int(os.environ.get("TOP_K_RETRIEVAL", 20))


class Retriever:
    def __init__(self, bm25_index_path: str):
        self.embedder = Embedder()
        self.pinecone = PineconeStore()
        self.reranker = Reranker()

        self.bm25 = BM25Indexer()
        self.bm25.load(bm25_index_path)

        logger.info("Retriever inicializado")

    def retrieve(self, query: str) -> list[dict]:
        if not query or not query.strip():
            raise ValueError("Query vazia")

        logger.info("Iniciando retrieval para query: '%s'", query)

        query_vector = self.embedder.embed_single(query)
        semantic_results = self.pinecone.search(query_vector, top_k=TOP_K_RETRIEVAL)
        lexical_results = self.bm25.search(query, top_k=TOP_K_RETRIEVAL)

        fused = reciprocal_rank_fusion(semantic_results, lexical_results)
        reranked = self.reranker.rerank(query, fused)

        logger.info("Retrieval concluído: %d chunks finais", len(reranked))
        return reranked
