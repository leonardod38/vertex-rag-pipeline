# v1.0.0 - 2026-06-09 - Versão inicial
import logging
import os
from dotenv import load_dotenv
from src.ingestion.loader import GCSLoader
from src.ingestion.chunker import Chunker
from src.ingestion.embedder import Embedder
from src.ingestion.bm25_indexer import BM25Indexer
from src.ingestion.pinecone_store import PineconeStore

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BM25_INDEX_PATH = os.environ.get("BM25_INDEX_PATH", "bm25_index.pkl")
GCS_PREFIX = os.environ.get("GCS_PREFIX", "")


def run():
    logger.info("Pipeline de ingestão iniciado")

    loader = GCSLoader()
    chunker = Chunker()
    embedder = Embedder()
    bm25 = BM25Indexer()
    store = PineconeStore()

    doc_names = loader.list_documents(prefix=GCS_PREFIX)
    if not doc_names:
        logger.warning("Nenhum documento encontrado no GCS com prefix='%s'", GCS_PREFIX)
        return

    all_chunks: list[dict] = []
    for name in doc_names:
        try:
            text = loader.download_text(name)
            chunks = chunker.split(text, source=name)
            all_chunks.extend(chunks)
        except Exception:
            logger.exception("Erro ao processar documento: %s", name)

    if not all_chunks:
        logger.error("Nenhum chunk gerado — abortando ingestão")
        return

    texts = [c["text"] for c in all_chunks]
    vectors = embedder.embed(texts)
    store.upsert(all_chunks, vectors)

    bm25.build(all_chunks)
    bm25.save(BM25_INDEX_PATH)

    logger.info("Pipeline de ingestão concluído: %d chunks processados", len(all_chunks))


if __name__ == "__main__":
    run()
