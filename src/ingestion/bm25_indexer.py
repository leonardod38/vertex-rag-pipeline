# v1.0.0 - 2026-06-09 - Versão inicial
import logging
import pickle
from pathlib import Path
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class BM25Indexer:
    def __init__(self):
        self.index: BM25Okapi | None = None
        self.chunks: list[dict] = []

    def build(self, chunks: list[dict]) -> None:
        if not chunks:
            raise ValueError("Lista de chunks vazia para indexação BM25")

        self.chunks = chunks
        tokenized = [chunk["text"].lower().split() for chunk in chunks]
        self.index = BM25Okapi(tokenized)
        logger.info("Índice BM25 construído: %d documentos", len(chunks))

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        if self.index is None:
            raise RuntimeError("Índice BM25 não foi construído")

        tokens = query.lower().split()
        scores = self.index.get_scores(tokens)

        ranked = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:top_k]

        results = [
            {**self.chunks[idx], "bm25_score": float(score)}
            for idx, score in ranked
            if score > 0
        ]
        logger.info("BM25 retornou %d resultados para query: '%s'", len(results), query)
        return results

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"index": self.index, "chunks": self.chunks}, f)
        logger.info("Índice BM25 salvo em: %s", path)

    def load(self, path: str) -> None:
        if not Path(path).exists():
            raise FileNotFoundError(f"Índice BM25 não encontrado: {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.index = data["index"]
        self.chunks = data["chunks"]
        logger.info("Índice BM25 carregado: %d documentos", len(self.chunks))
