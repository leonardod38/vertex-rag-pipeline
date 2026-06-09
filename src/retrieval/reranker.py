# v1.0.0 - 2026-06-09 - Versão inicial
import logging
import os
from vertexai.preview.generative_models import GenerativeModel
import vertexai

logger = logging.getLogger(__name__)

GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
GCP_LOCATION = os.environ.get("GCP_LOCATION", "us-central1")
RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "semantic-ranker-512@latest")
TOP_K_RERANK = int(os.environ.get("TOP_K_RERANK", 5))


class Reranker:
    def __init__(self):
        if not GCP_PROJECT_ID:
            raise ValueError("GCP_PROJECT_ID não configurado")
        vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
        logger.info("Reranker iniciado: modelo=%s", RERANKER_MODEL)

    def rerank(self, query: str, chunks: list[dict]) -> list[dict]:
        if not chunks:
            raise ValueError("Lista de chunks vazia para reranking")
        if not query or not query.strip():
            raise ValueError("Query vazia para reranking")

        from google.cloud.discoveryengine_v1alpha import RankServiceClient, RankRequest, RankingRecord

        client = RankServiceClient()
        records = [
            RankingRecord(id=str(i), content=chunk["text"])
            for i, chunk in enumerate(chunks)
        ]

        request = RankRequest(
            ranking_config=f"projects/{GCP_PROJECT_ID}/locations/global/rankingConfigs/default_ranking_config",
            model=RERANKER_MODEL,
            top_n=TOP_K_RERANK,
            query=query,
            records=records,
        )

        response = client.rank(request=request)
        id_to_chunk = {str(i): chunk for i, chunk in enumerate(chunks)}

        results = [
            {**id_to_chunk[record.id], "rerank_score": record.score}
            for record in response.records
        ]
        logger.info("Reranker retornou %d chunks (de %d candidatos)", len(results), len(chunks))
        return results
