import json
import numpy as np
from pathlib import Path

from src.embeddings.vectorstore import store_embeddings


def test_store_embeddings_success(mocker, tmp_path):
    """
    Unit test for store_embeddings()
    Verifies:
    - embeddings + metadata are loaded
    - upsert is called correctly
    - batching works
    """

    # Setup fake files
    emb_path = tmp_path / "embeddings.npy"
    meta_path = tmp_path / "metadata.json"
    chunks_path = tmp_path / "chunks.jsonl"
    persist_dir = tmp_path / "chroma"

    # Fake embeddings (2 records, 3-dim vectors)
    fake_embeddings = np.array([[0.1, 0.2, 0.3],
                                [0.4, 0.5, 0.6]])
    np.save(emb_path, fake_embeddings)

    # Fake metadata
    fake_metadata = [
        {"topic": "A"},
        {"topic": "B"},
    ]
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(fake_metadata, f)

    # Fake chunk texts
    with open(chunks_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"text": "First chunk"}) + "\n")
        f.write(json.dumps({"text": "Second chunk"}) + "\n")


    # Mock Config
    mock_cfg = mocker.MagicMock()
    mock_cfg.embeddings_path = str(emb_path)
    mock_cfg.metadata_path = str(meta_path)
    mock_cfg.chunks_path = str(chunks_path)
    mock_cfg.persist_directory = str(persist_dir)
    mock_cfg.collection_name = "test_collection"
    mock_cfg.batch_size = 10  # bigger than dataset

    mocker.patch(
        "src.embeddings.vectorstore.load_vectorstore_config",
        return_value=mock_cfg,
    )

    # Mock Chroma Client + Collection
    mock_collection = mocker.MagicMock()
    mock_client = mocker.MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection

    mocker.patch(
        "src.embeddings.vectorstore.get_vector_store",
        return_value=mock_client,
    )


    # Run function
    store_embeddings()


    # Assertions
    mock_client.get_or_create_collection.assert_called_once_with(
        name="test_collection"
    )

    mock_collection.upsert.assert_called_once()

    # Extract call args
    args, kwargs = mock_collection.upsert.call_args

    assert len(kwargs["ids"]) == 2
    assert len(kwargs["embeddings"]) == 2
    assert len(kwargs["metadatas"]) == 2
    assert len(kwargs["documents"]) == 2

    assert kwargs["metadatas"][0]["topic"] == "A"
    assert kwargs["documents"][0] == "First chunk"