# v1.0.0 - 2026-06-09 - Versão inicial
import logging
import os
from dotenv import load_dotenv
from src.retrieval.retriever import Retriever
from src.generation.generator import Generator

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BM25_INDEX_PATH = os.environ.get("BM25_INDEX_PATH", "bm25_index.pkl")


def run(query: str) -> dict:
    if not query or not query.strip():
        raise ValueError("Query vazia")

    logger.info("Pipeline de consulta iniciado: '%s'", query)

    retriever = Retriever(bm25_index_path=BM25_INDEX_PATH)
    generator = Generator()

    chunks = retriever.retrieve(query)
    result = generator.generate(query, chunks)

    logger.info("Pipeline de consulta concluído")
    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso: python -m pipelines.query_pipeline \"<pergunta>\"")
        sys.exit(1)

    question = sys.argv[1]
    output = run(question)
    print("\n--- Resposta ---")
    print(output["answer"])
    print("\n--- Fontes ---")
    for s in output["sources"]:
        print(f"  • {s}")
