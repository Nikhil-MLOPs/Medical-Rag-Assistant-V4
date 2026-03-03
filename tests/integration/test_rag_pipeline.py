def test_full_rag_pipeline(mocker):

    # Mock config
    mock_cfg = mocker.MagicMock()
    mock_cfg.ollama_model = "mock"
    mock_cfg.temperature = 0.1
    mock_cfg.max_tokens = 100
    mock_cfg.use_memory = False
    mock_cfg.guardrails_enabled = False
    mock_cfg.explainability_enabled = True

    mocker.patch(
        "src.rag.rag_service.load_rag_config",
        return_value=mock_cfg
    )

    # Mock retriever
    mock_result = mocker.MagicMock()
    mock_result.text = "Pneumonia is lung infection"
    mock_result.metadata = {"topic": "Pneumonia"}
    mock_result.score = 1.0

    mock_retriever = mocker.MagicMock()
    mock_retriever.retrieve.return_value.results = [mock_result]

    mocker.patch(
        "src.rag.rag_service.RetrieverService",
        return_value=mock_retriever
    )

    # Mock ollama
    mock_ollama = mocker.MagicMock()
    mock_ollama.chat.return_value = {
        "message": {"content": "It is caused by bacteria."}
    }

    mocker.patch(
        "src.rag.rag_service.ollama",
        mock_ollama
    )

    from src.rag.rag_service import RagService

    service = RagService()
    response = service.ask("What causes pneumonia?")

    assert response.answer
    assert len(response.citations) == 1
    assert response.explanation is not None