# v1.0.0 - 2026-06-09 - Versão inicial
import logging
import os
from pinecone import Pinecone

logger = logging.getLogger(__name__)

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")


class PineconeStore:
    def __init__(self):
        if not PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY não configurado")
        if not PINECONE_INDEX_NAME:
            raise ValueError("PINECONE_INDEX_NAME não configurado")

        pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index = pc.Index(PINECONE_INDEX_NAME)
        logger.info("PineconeStore conectado ao índice: %s", PINECONE_INDEX_NAME)

    def upsert(self, chunks: list[dict], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks e vectors devem ter o mesmo tamanho")

        records = [
            {
                "id": f"{chunk['source']}__chunk_{chunk['chunk_index']}",
                "values": vector,
                "metadata": {"text": chunk["text"], "source": chunk["source"]},
            }
            for chunk, vector in zip(chunks, vectors)
        ]

        batch_size = 100
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            self.index.upsert(vectors=batch)
            logger.info("Upsert batch %d/%d (%d vetores)", i // batch_size + 1, -(-len(records) // batch_size), len(batch))

        logger.info("Upsert concluído: %d vetores indexados", len(records))

    def search(self, vector: list[float], top_k: int = 20) -> list[dict]:
        response = self.index.query(vector=vector, top_k=top_k, include_metadata=True)
        results = [
            {
                "text": match.metadata["text"],
                "source": match.metadata["source"],
                "pinecone_score": match.score,
            }
            for match in response.matches
        ]
        logger.info("Pinecone retornou %d resultados", len(results))
        return results
