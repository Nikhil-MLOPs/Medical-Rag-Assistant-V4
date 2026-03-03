from src.retrieval.reranker import Reranker
from src.retrieval.schema import RetrievalResult


def test_reranker_sorts(mocker):

    mock_model = mocker.MagicMock()
    mock_model.predict.return_value = [0.2, 0.9]

    mocker.patch(
        "src.retrieval.reranker.CrossEncoder",
        return_value=mock_model
    )

    reranker = Reranker("mock-model")

    results = [
        RetrievalResult(text="doc1", metadata={}, score=0.0),
        RetrievalResult(text="doc2", metadata={}, score=0.0),
    ]

    reranked = reranker.rerank("query", results)

    assert reranked[0].text == "doc2"
    assert reranked[0].score == 0.9