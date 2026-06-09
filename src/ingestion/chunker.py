# v1.0.0 - 2026-06-09 - Versão inicial
import logging
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 50))


class Chunker:
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " ", ""],
        )
        logger.info("Chunker iniciado: chunk_size=%d, overlap=%d", CHUNK_SIZE, CHUNK_OVERLAP)

    def split(self, text: str, source: str) -> list[dict]:
        if not text or not text.strip():
            raise ValueError(f"Texto vazio para fonte: {source}")

        chunks = self.splitter.split_text(text)
        result = [
            {"text": chunk, "source": source, "chunk_index": i}
            for i, chunk in enumerate(chunks)
        ]
        logger.info("Chunks gerados: %d (fonte: %s)", len(result), source)
        return result
