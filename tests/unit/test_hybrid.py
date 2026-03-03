from src.retrieval.hybrid import HybridRetriever
from src.retrieval.schema import RetrievalResult


def test_hybrid_fusion():

    dense = [
        RetrievalResult(text="doc1", metadata={}, score=0.9)
    ]

    sparse = [
        RetrievalResult(text="doc1", metadata={}, score=0.5)
    ]

    hybrid = HybridRetriever(alpha=0.7)

    results = hybrid.fuse(dense, sparse, top_k=1)

    expected_score = 0.7 * 0.9 + 0.3 * 0.5

    assert len(results) == 1
    assert abs(results[0].score - expected_score) < 1e-6