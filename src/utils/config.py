import yaml
from pydantic import BaseModel, Field



# Phase-1: Ingestion Config

class IngestionConfig(BaseModel):
    raw_dir: str = Field(description="Directory where raw data is stored")
    processed_dir: str = Field(description="Directory where processed data will be stored")
    skip_start_pages: int = Field(description="Number of pages to skip from start")
    skip_after_pages: int = Field(description="Number of pages to skip after limit")


# Phase-2: Cleaning Config

class CleaningConfig(BaseModel):
    chunk_size: int = Field(description="Size of each chunk in characters")
    chunk_overlap: int = Field(description="Overlap between consecutive chunks")


# Phase-3: Embedding Config

class EmbeddingConfig(BaseModel):
    model_name: str = Field(description="SentenceTransformer model name")
    batch_size: int = Field(description="Batch size for embedding")
    input_chunks_path: str = Field(description="Path to cleaned chunks JSONL file")
    output_dir: str = Field(description="Directory to store embeddings")


# Shared YAML Loader

def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_ingestion_config(path: str) -> IngestionConfig:
    return IngestionConfig(**load_yaml(path))


def load_cleaning_config(path: str) -> CleaningConfig:
    return CleaningConfig(**load_yaml(path))


def load_embedding_config(path: str) -> EmbeddingConfig:
    return EmbeddingConfig(**load_yaml(path))