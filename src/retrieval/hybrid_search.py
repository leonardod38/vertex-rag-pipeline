# v1.0.0 - 2026-06-09 - Versão inicial
import logging

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    semantic_results: list[dict],
    lexical_results: list[dict],
    k: int = 60,
) -> list[dict]:
    """
    Funde dois rankings usando RRF.
    k=60 é o valor padrão da literatura — controla o peso de posições baixas.
    """
    scores: dict[str, float] = {}
    texts: dict[str, dict] = {}

    for rank, item in enumerate(semantic_results):
        key = item["text"]
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        texts[key] = item

    for rank, item in enumerate(lexical_results):
        key = item["text"]
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        texts[key] = item

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    results = [{**texts[text], "rrf_score": score} for text, score in fused]
    logger.info("RRF fundiu %d resultados semânticos + %d lexicais → %d únicos", len(semantic_results), len(lexical_results), len(results))
    return results
