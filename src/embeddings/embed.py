import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils.logging import setup_logging
from src.utils.config import load_embedding_config

from dotenv import load_dotenv

load_dotenv()

# Set up logging for Embeddings stage
logger = setup_logging("Embeddings")

# Path to the embedding configuration file
CONFIG_PATH = Path("configs/embeddings.yaml")

# Main function to perform embedding of text chunks
def embed_chunks():
    logger.info("Starting embedding stage")

    cfg = load_embedding_config(CONFIG_PATH)

    model = SentenceTransformer(cfg.model_name)

    chunks_file = Path(cfg.input_chunks_path)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    texts = []
    metadatas = []

    if not chunks_file.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_file}")

    with open(chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            texts.append(record["text"])
            metadatas.append(record["metadata"])

    logger.info(f"Loaded {len(texts)} chunks for embedding")

    embeddings = model.encode(
        texts,
        batch_size=cfg.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    emb_path = output_dir / "embeddings.npy"
    meta_path = output_dir / "metadata.json"

    np.save(emb_path, embeddings)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadatas, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved embeddings to {emb_path}")
    logger.info(f"Saved metadata to {meta_path}")
    logger.info("Embedding pipeline completed successfully")


if __name__ == "__main__":
    embed_chunks()