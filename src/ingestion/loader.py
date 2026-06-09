# v1.0.0 - 2026-06-09 - Versão inicial
import logging
import os
from google.cloud import storage

logger = logging.getLogger(__name__)


class GCSLoader:
    def __init__(self):
        self.bucket_name = os.environ.get("GCS_BUCKET_NAME")
        if not self.bucket_name:
            raise ValueError("GCS_BUCKET_NAME não configurado")
        self.client = storage.Client()

    def list_documents(self, prefix: str = "") -> list[str]:
        bucket = self.client.bucket(self.bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)
        names = [b.name for b in blobs if b.name.endswith((".pdf", ".txt"))]
        logger.info("Documentos encontrados no GCS: %d", len(names))
        return names

    def download_text(self, blob_name: str) -> str:
        bucket = self.client.bucket(self.bucket_name)
        blob = bucket.blob(blob_name)

        if blob_name.endswith(".txt"):
            content = blob.download_as_text()
            logger.info("Carregado TXT: %s (%d chars)", blob_name, len(content))
            return content

        if blob_name.endswith(".pdf"):
            return self._extract_pdf(blob, blob_name)

        raise ValueError(f"Formato não suportado: {blob_name}")

    def _extract_pdf(self, blob, blob_name: str) -> str:
        import io
        import pypdf

        data = blob.download_as_bytes()
        reader = pypdf.PdfReader(io.BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(pages)
        logger.info("Carregado PDF: %s (%d páginas, %d chars)", blob_name, len(pages), len(text))
        return text
