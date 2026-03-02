import json
import numpy as np
from pathlib import Path
from types import SimpleNamespace

from src.embeddings.embed import embed_chunks


def test_embed_chunks_logic(mocker, tmp_path):

    # Setup fake chunks file
    chunks_file = tmp_path / "chunks.jsonl"
    out_dir = tmp_path / "embeddings"
    out_dir.mkdir()

    fake_chunks = [
        {"text": "First medical chunk", "metadata": {"topic": "A"}},
        {"text": "Second medical chunk", "metadata": {"topic": "B"}},
    ]

    with open(chunks_file, "w", encoding="utf-8") as f:
        for c in fake_chunks:
            f.write(json.dumps(c) + "\n")


    # Mock Config (Proper Object)
    mock_cfg = SimpleNamespace(
        model_name="mock-model",
        batch_size=2,
        input_chunks_path=str(chunks_file),
        output_dir=str(out_dir),
    )

    mocker.patch(
        "src.embeddings.embed.load_embedding_config",
        return_value=mock_cfg,
    )

    # Mock SentenceTransformer
    mock_model = mocker.MagicMock()
    mock_model.encode.return_value = np.random.rand(2, 384)

    mocker.patch(
        "src.embeddings.embed.SentenceTransformer",
        return_value=mock_model,
    )

    # RUN
    embed_chunks()

    # ASSERTIONS
    emb_path = out_dir / "embeddings.npy"
    meta_path = out_dir / "metadata.json"

    assert emb_path.exists()
    assert meta_path.exists()

    loaded_embs = np.load(emb_path)
    assert loaded_embs.shape == (2, 384)

    with open(meta_path, "r", encoding="utf-8") as f:
        loaded_meta = json.load(f)

    assert len(loaded_meta) == 2
    assert loaded_meta[0]["topic"] == "A"

    mock_model.encode.assert_called_once()