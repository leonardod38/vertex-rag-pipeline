# v1.0.0 - 2026-06-09 - Versão inicial
import logging
import os
from vertexai.language_models import TextEmbeddingModel
import vertexai

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "textembedding-gecko@003")
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
GCP_LOCATION = os.environ.get("GCP_LOCATION", "us-central1")


class Embedder:
    def __init__(self):
        if not GCP_PROJECT_ID:
            raise ValueError("GCP_PROJECT_ID não configurado")
        vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
        self.model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)
        logger.info("Embedder iniciado: modelo=%s", EMBEDDING_MODEL)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            raise ValueError("Lista de textos vazia")

        embeddings = self.model.get_embeddings(texts)
        vectors = [e.values for e in embeddings]
        logger.info("Embeddings gerados: %d vetores", len(vectors))
        return vectors

    def embed_single(self, text: str) -> list[float]:
        return self.embed([text])[0]
