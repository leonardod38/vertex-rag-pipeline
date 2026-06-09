import os
import pytest

os.environ.setdefault("CHUNK_SIZE", "512")
os.environ.setdefault("CHUNK_OVERLAP", "50")

from src.ingestion.chunker import Chunker


@pytest.fixture
def chunker():
    return Chunker()


def test_split_retorna_chunks(chunker):
    # Arrange
    text = "palavra " * 600
    # Act
    result = chunker.split(text, source="doc.txt")
    # Assert
    assert len(result) > 1
    assert all("text" in c and "source" in c and "chunk_index" in c for c in result)


def test_split_preserva_source(chunker):
    # Arrange
    text = "conteúdo de teste " * 100
    # Act
    result = chunker.split(text, source="arquivo.pdf")
    # Assert
    assert all(c["source"] == "arquivo.pdf" for c in result)


def test_split_indexa_sequencialmente(chunker):
    # Arrange
    text = "token " * 600
    # Act
    result = chunker.split(text, source="doc.txt")
    # Assert
    indices = [c["chunk_index"] for c in result]
    assert indices == list(range(len(result)))


def test_split_texto_vazio_levanta_erro(chunker):
    # Arrange / Act / Assert
    with pytest.raises(ValueError):
        chunker.split("", source="doc.txt")


def test_split_texto_apenas_espacos_levanta_erro(chunker):
    # Arrange / Act / Assert
    with pytest.raises(ValueError):
        chunker.split("   \n  ", source="doc.txt")
