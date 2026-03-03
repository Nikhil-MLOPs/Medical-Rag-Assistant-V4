def test_rag_service_orchestration(mocker):

    mock_cfg = mocker.MagicMock()
    mock_cfg.ollama_model = "mock"
    mock_cfg.temperature = 0.1
    mock_cfg.max_tokens = 100
    mock_cfg.use_memory = False
    mock_cfg.guardrails_enabled = False
    mock_cfg.explainability_enabled = False

    mocker.patch(
        "src.rag.rag_service.load_rag_config",
        return_value=mock_cfg
    )

    mock_retriever = mocker.MagicMock()
    mock_retriever.retrieve.return_value.results = []

    mocker.patch(
        "src.rag.rag_service.RetrieverService",
        return_value=mock_retriever
    )

    mock_ollama = mocker.MagicMock()
    mock_ollama.chat.return_value = {
        "message": {"content": "Mock answer"}
    }

    mocker.patch(
        "src.rag.rag_service.ollama",
        mock_ollama
    )

    from src.rag.rag_service import RagService

    service = RagService()
    response = service.ask("Test")

    assert response.answer == "Mock answer"