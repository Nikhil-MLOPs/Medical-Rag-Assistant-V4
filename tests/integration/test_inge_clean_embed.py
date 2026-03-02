import json
import numpy as np
from pathlib import Path
from langchain_core.documents import Document

from src.ingestion.ingest import ingest
from src.cleaning.clean import clean_and_chunk
from src.embeddings.embed import embed_chunks
from src.utils.config import CleaningConfig


def test_full_etl_pipeline(mocker, tmp_path):
    """
    Full pipeline:
    Ingestion -> Cleaning -> Embedding
    """

    # -------------------------------------------------
    # 1️⃣ Setup directory structure
    # -------------------------------------------------
    base_dir = tmp_path / "data"
    raw_dir = base_dir / "raw"
    processed_dir = base_dir / "processed"
    chunks_dir = processed_dir / "chunks"
    embeddings_dir = base_dir / "embeddings"

    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    chunks_dir.mkdir(parents=True)
    embeddings_dir.mkdir(parents=True)

    # -------------------------------------------------
    # 2️⃣ Mock Ingestion Config
    # -------------------------------------------------
    mock_ingest_cfg = mocker.MagicMock()
    mock_ingest_cfg.raw_dir = str(raw_dir)
    mock_ingest_cfg.processed_dir = str(processed_dir)
    mock_ingest_cfg.skip_start_pages = 0
    mock_ingest_cfg.skip_after_pages = 100

    mocker.patch(
        "src.ingestion.ingest.load_ingestion_config",
        return_value=mock_ingest_cfg,
    )

    # -------------------------------------------------
    # 3️⃣ Mock PDF Reader
    # -------------------------------------------------
    fake_pdf = raw_dir / "medical.pdf"
    fake_pdf.write_text("fake content")

    mock_page = mocker.MagicMock()
    mock_page.get_text.return_value = (
        "Pneumonia\n"
        "definition\n"
        "Infection of lungs.\n"
        "causes\n"
        "Bacteria."
    )

    mock_doc = mocker.MagicMock()
    mock_doc.__len__.return_value = 1
    mock_doc.load_page.return_value = mock_page

    mocker.patch("pymupdf.open", return_value=mock_doc)

    # -------------------------------------------------
    # 4️⃣ Run Ingestion
    # -------------------------------------------------
    ingest()

    pages_file = processed_dir / "pages.jsonl"
    assert pages_file.exists()

    # -------------------------------------------------
    # 5️⃣ Run Cleaning
    # -------------------------------------------------
    pages = []
    with open(pages_file, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            pages.append(
                Document(
                    page_content=record["text"],
                    metadata=record["metadata"],
                )
            )

    clean_cfg = CleaningConfig(chunk_size=500, chunk_overlap=0)
    chunks = clean_and_chunk(pages, clean_cfg)

    assert len(chunks) >= 1
    assert chunks[0].metadata["topic"] == "Pneumonia"

    # Save chunks (simulate cleaning stage output)
    chunks_file = chunks_dir / "chunks.jsonl"
    with open(chunks_file, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(
                json.dumps(
                    {
                        "text": chunk.page_content,
                        "metadata": chunk.metadata,
                    }
                )
                + "\n"
            )

    # -------------------------------------------------
    # 6️⃣ Mock Embedding Config
    # -------------------------------------------------
    mock_embed_cfg = mocker.MagicMock()
    mock_embed_cfg.model_name = "mock-model"
    mock_embed_cfg.batch_size = 2
    mock_embed_cfg.input_chunks_path = str(chunks_file)
    mock_embed_cfg.output_dir = str(embeddings_dir)

    mocker.patch(
        "src.embeddings.embed.load_embedding_config",
        return_value=mock_embed_cfg,
    )

    # -------------------------------------------------
    # 7️⃣ Mock SentenceTransformer
    # -------------------------------------------------
    mock_model = mocker.MagicMock()
    mock_model.encode.return_value = np.random.rand(len(chunks), 384)

    mocker.patch(
        "src.embeddings.embed.SentenceTransformer",
        return_value=mock_model,
    )

    # -------------------------------------------------
    # 8️⃣ Run Embedding
    # -------------------------------------------------
    embed_chunks()

    # -------------------------------------------------
    # 9️⃣ Final Assertions
    # -------------------------------------------------
    emb_file = embeddings_dir / "embeddings.npy"
    meta_file = embeddings_dir / "metadata.json"

    assert emb_file.exists()
    assert meta_file.exists()

    vectors = np.load(emb_file)

    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert vectors.shape[0] == len(chunks)
    assert meta[0]["topic"] == "Pneumonia"
    assert meta[0]["section"] == "definition"