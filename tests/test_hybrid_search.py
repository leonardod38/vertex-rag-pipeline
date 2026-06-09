import pytest
from src.retrieval.hybrid_search import reciprocal_rank_fusion


def _make_results(texts: list[str], score_key: str) -> list[dict]:
    return [{"text": t, "source": "doc.txt", score_key: float(i)} for i, t in enumerate(texts)]


def test_rrf_retorna_todos_unicos():
    # Arrange
    semantic = _make_results(["A", "B", "C"], "pinecone_score")
    lexical = _make_results(["B", "C", "D"], "bm25_score")
    # Act
    result = reciprocal_rank_fusion(semantic, lexical)
    texts = [r["text"] for r in result]
    # Assert
    assert set(texts) == {"A", "B", "C", "D"}


def test_rrf_item_em_ambos_tem_score_maior():
    # Arrange
    semantic = _make_results(["X", "Y"], "pinecone_score")
    lexical = _make_results(["X", "Z"], "bm25_score")
    # Act
    result = reciprocal_rank_fusion(semantic, lexical)
    scores = {r["text"]: r["rrf_score"] for r in result}
    # Assert — X aparece nos dois rankings, deve ter score maior que Y e Z
    assert scores["X"] > scores["Y"]
    assert scores["X"] > scores["Z"]


def test_rrf_lista_vazia_retorna_vazio():
    # Arrange / Act
    result = reciprocal_rank_fusion([], [])
    # Assert
    assert result == []


def test_rrf_apenas_semantico():
    # Arrange
    semantic = _make_results(["A", "B"], "pinecone_score")
    # Act
    result = reciprocal_rank_fusion(semantic, [])
    # Assert
    assert len(result) == 2


def test_rrf_apenas_lexical():
    # Arrange
    lexical = _make_results(["A", "B"], "bm25_score")
    # Act
    result = reciprocal_rank_fusion([], lexical)
    # Assert
    assert len(result) == 2
