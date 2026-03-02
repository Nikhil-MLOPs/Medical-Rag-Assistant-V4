import json
from pathlib import Path
import numpy as np
import chromadb
from chromadb.config import Settings

from src.utils.logging import setup_logging
from src.utils.config import load_vectorstore_config

# Set up logging for the vector store
logger = setup_logging("VectorStore")

# Path to the vector store configuration file
CONFIG_PATH = Path("configs/vectorstore.yaml")

# Function to initialize and return a Chroma vector store client
def get_vector_store(persist_directory: str):
    return chromadb.PersistentClient(path=persist_directory, settings=Settings(anonymized_telemetry=False))

# Main function to load embeddings and metadata, and store them in the vector store
def store_embeddings():
    logger.info("Starting vector store build stage")

    cfg = load_vectorstore_config(CONFIG_PATH)

    emb_path = Path(cfg.embeddings_path)
    meta_path = Path(cfg.metadata_path)
    chunks_path = Path(cfg.chunks_path)

    if not (emb_path.exists() and meta_path.exists()):
        raise FileNotFoundError("Embedding files not found. Run embedding stage first.")

    # Load data
    logger.info("Loading embeddings and metadata...")

    embeddings = np.load(emb_path).tolist()

    with open(meta_path, "r", encoding="utf-8") as f:
        metadatas = json.load(f)

    texts = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            texts.append(json.loads(line)["text"])


    # Initialize Chroma
    client = get_vector_store(cfg.persist_directory)

    collection = client.get_or_create_collection(name=cfg.collection_name)

    ids = [f"id_{i}" for i in range(len(texts))]

    total_records = len(ids)
    batch_size = cfg.batch_size

    logger.info(f"Storing {total_records} vectors in batches of {batch_size}...")

    # Batch Upsert
    try:
        for i in range(0, total_records, batch_size):
            end_idx = min(i + batch_size, total_records)

            collection.upsert(ids=ids[i:end_idx], embeddings=embeddings[i:end_idx], metadatas=metadatas[i:end_idx], documents=texts[i:end_idx])

            logger.info(f"Stored batch {i} to {end_idx}")

        logger.info("Vector storage completed successfully.")

    except Exception as e:
        logger.error(f"Error during storage: {e}")
        raise


if __name__ == "__main__":
    store_embeddings()