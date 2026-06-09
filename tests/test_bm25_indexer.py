import os
import tempfile
import pytest
from src.ingestion.bm25_indexer import BM25Indexer


@pytest.fixture
def indexer_com_dados():
    indexer = BM25Indexer()
    chunks = [
        {"text": "inteligência artificial e machine learning", "source": "doc1.txt", "chunk_index": 0},
        {"text": "redes neurais e deep learning", "source": "doc2.txt", "chunk_index": 0},
        {"text": "processamento de linguagem natural", "source": "doc3.txt", "chunk_index": 0},
    ]
    indexer.build(chunks)
    return indexer


def test_build_cria_indice(indexer_com_dados):
    assert indexer_com_dados.index is not None


def test_search_retorna_resultado_relevante(indexer_com_dados):
    # Arrange / Act
    results = indexer_com_dados.search("machine learning", top_k=2)
    # Assert
    assert len(results) >= 1
    assert any("machine learning" in r["text"] for r in results)


def test_search_sem_match_retorna_vazio(indexer_com_dados):
    # Arrange / Act
    results = indexer_com_dados.search("xyzxyzxyz", top_k=5)
    # Assert
    assert results == []


def test_build_lista_vazia_levanta_erro():
    # Arrange
    indexer = BM25Indexer()
    # Act / Assert
    with pytest.raises(ValueError):
        indexer.build([])


def test_search_sem_build_levanta_erro():
    # Arrange
    indexer = BM25Indexer()
    # Act / Assert
    with pytest.raises(RuntimeError):
        indexer.search("qualquer coisa")


def test_save_e_load(indexer_com_dados):
    # Arrange
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        path = f.name
    try:
        # Act
        indexer_com_dados.save(path)
        novo = BM25Indexer()
        novo.load(path)
        results = novo.search("machine learning", top_k=1)
        # Assert
        assert len(results) >= 1
    finally:
        os.unlink(path)
