def test_full_retrieval_pipeline(mocker):

    mock_response = {
        "documents": [["doc1"]],
        "metadatas": [[{"topic": "Pneumonia"}]],
        "distances": [[0.1]]
    }

    # Mock Dense internals
    mock_model = mocker.MagicMock()
    mock_model.encode.return_value = [0.1, 0.2]

    mocker.patch("src.retrieval.dense.SentenceTransformer", return_value=mock_model)

    mock_collection = mocker.MagicMock()
    mock_collection.query.return_value = mock_response

    mock_client = mocker.MagicMock()
    mock_client.get_collection.return_value = mock_collection

    mocker.patch(
        "src.retrieval.dense.chromadb.PersistentClient",
        return_value=mock_client
    )

    # Mock Reranker
    mock_cross = mocker.MagicMock()
    mock_cross.predict.return_value = [0.9]

    mocker.patch(
        "src.retrieval.reranker.CrossEncoder",
        return_value=mock_cross
    )

    mock_cfg = mocker.MagicMock()
    mock_cfg.persist_directory = "dummy"
    mock_cfg.collection_name = "test"
    mock_cfg.embedding_model_name = "mock-model"
    mock_cfg.reranker_model_name = "mock-reranker"
    mock_cfg.chunks_path = "dummy"
    mock_cfg.top_k_dense = 1
    mock_cfg.top_k_sparse = 0
    mock_cfg.top_k_final = 1
    mock_cfg.hybrid_alpha = 1.0

    mocker.patch(
        "src.retrieval.retriever.load_retrieval_config",
        return_value=mock_cfg
    )

    mock_sparse = mocker.MagicMock()
    mock_sparse.search.return_value = []

    mock_hybrid = mocker.MagicMock()
    mock_hybrid.fuse.side_effect = lambda dense, sparse, top_k: dense

    mocker.patch(
        "src.retrieval.retriever.SparseRetriever",
        return_value=mock_sparse
    )

    mocker.patch(
        "src.retrieval.retriever.HybridRetriever",
        return_value=mock_hybrid
    )

    from src.retrieval.retriever import RetrieverService

    service = RetrieverService()
    response = service.retrieve("pneumonia")

    assert len(response.results) == 1
    assert response.results[0].metadata["topic"] == "Pneumonia"