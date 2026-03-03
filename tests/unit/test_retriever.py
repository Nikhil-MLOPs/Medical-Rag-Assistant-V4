from src.retrieval.retriever import RetrieverService


def test_retriever_orchestration(mocker):

    mock_cfg = mocker.MagicMock()
    mock_cfg.persist_directory = "dummy"
    mock_cfg.collection_name = "test"
    mock_cfg.embedding_model_name = "embed"
    mock_cfg.reranker_model_name = "rerank"
    mock_cfg.chunks_path = "dummy"
    mock_cfg.top_k_dense = 2
    mock_cfg.top_k_sparse = 2
    mock_cfg.top_k_final = 1
    mock_cfg.hybrid_alpha = 0.5

    mocker.patch(
        "src.retrieval.retriever.load_retrieval_config",
        return_value=mock_cfg
    )

    mock_dense = mocker.MagicMock()
    mock_dense.search.return_value = []

    mock_sparse = mocker.MagicMock()
    mock_sparse.search.return_value = []

    mock_hybrid = mocker.MagicMock()
    mock_hybrid.fuse.return_value = []

    mock_reranker = mocker.MagicMock()
    mock_reranker.rerank.return_value = []

    mocker.patch("src.retrieval.retriever.DenseRetriever", return_value=mock_dense)
    mocker.patch("src.retrieval.retriever.SparseRetriever", return_value=mock_sparse)
    mocker.patch("src.retrieval.retriever.HybridRetriever", return_value=mock_hybrid)
    mocker.patch("src.retrieval.retriever.Reranker", return_value=mock_reranker)

    service = RetrieverService()
    response = service.retrieve("query")

    assert response.query == "query"
    assert response.results == []