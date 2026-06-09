# v1.0.0 - 2026-06-09 - Versão inicial
import logging
import os
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

logger = logging.getLogger(__name__)

GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
GCP_LOCATION = os.environ.get("GCP_LOCATION", "us-central1")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-pro")

SYSTEM_PROMPT = (
    "Você é um assistente especializado. Responda à pergunta do usuário "
    "baseando-se EXCLUSIVAMENTE no contexto fornecido. "
    "Se a resposta não estiver no contexto, diga que não há informação suficiente. "
    "Cite as fontes ao final da resposta."
)


class Generator:
    def __init__(self):
        if not GCP_PROJECT_ID:
            raise ValueError("GCP_PROJECT_ID não configurado")
        vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
        self.model = GenerativeModel(
            model_name=GEMINI_MODEL,
            system_instruction=SYSTEM_PROMPT,
        )
        self.config = GenerationConfig(temperature=0.2, max_output_tokens=1024)
        logger.info("Generator iniciado: modelo=%s", GEMINI_MODEL)

    def generate(self, query: str, chunks: list[dict]) -> dict:
        if not query or not query.strip():
            raise ValueError("Query vazia")
        if not chunks:
            raise ValueError("Nenhum chunk fornecido para geração")

        context = "\n\n".join(
            f"[Fonte: {c['source']}]\n{c['text']}" for c in chunks
        )
        sources = list({c["source"] for c in chunks})

        prompt = f"Contexto:\n{context}\n\nPergunta: {query}"
        response = self.model.generate_content(prompt, generation_config=self.config)
        answer = response.text

        logger.info("Resposta gerada (%d chars) com %d fontes", len(answer), len(sources))
        return {
            "answer": answer,
            "sources": sources,
            "chunks_used": len(chunks),
        }
