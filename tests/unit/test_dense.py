import numpy as np
from src.retrieval.dense import DenseRetriever


def test_dense_retriever_search(mocker):

    # Mock embedding model
    mock_model = mocker.MagicMock()
    mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])

    mocker.patch(
        "src.retrieval.dense.SentenceTransformer",
        return_value=mock_model
    )

    # Mock Chroma collection
    mock_collection = mocker.MagicMock()
    mock_collection.query.return_value = {
        "documents": [["doc1"]],
        "metadatas": [[{"topic": "A"}]],
        "distances": [[0.2]]
    }

    mock_client = mocker.MagicMock()
    mock_client.get_collection.return_value = mock_collection

    mocker.patch(
        "src.retrieval.dense.chromadb.PersistentClient",
        return_value=mock_client
    )

    retriever = DenseRetriever(
        persist_directory="dummy",
        collection_name="test",
        model_name="mock-model"
    )

    results = retriever.search("query", 1)

    assert len(results) == 1
    assert results[0].metadata["topic"] == "A"
    assert results[0].score == 1 - 0.2