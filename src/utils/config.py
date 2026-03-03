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


# Phase-4: Vector Store Config

class VectorStoreConfig(BaseModel):
    embeddings_path: str = Field(description="Path to embeddings .npy file")
    metadata_path: str = Field(description="Path to metadata JSON file")
    chunks_path: str = Field(description="Path to cleaned chunks JSONL file")
    persist_directory: str = Field(description="Directory to persist Chroma DB")
    collection_name: str = Field(description="Chroma collection name")
    batch_size: int = Field(description="Batch size for vector insertion")


# Phase-5: Retrieval Config

class RetrievalConfig(BaseModel):
    persist_directory: str
    collection_name: str
    embedding_model_name: str
    reranker_model_name: str
    chunks_path: str

    top_k_dense: int
    top_k_sparse: int
    top_k_final: int

    hybrid_alpha: float

# Phase-6: Rag Config

class RagConfig(BaseModel):
    ollama_model: str
    temperature: float
    max_tokens: int
    use_memory: bool
    guardrails_enabled: bool
    explainability_enabled: bool

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


def load_vectorstore_config(path: str) -> VectorStoreConfig:
    return VectorStoreConfig(**load_yaml(path))


def load_retrieval_config(path: str) -> RetrievalConfig:
    return RetrievalConfig(**load_yaml(path))


def load_rag_config(path: str) -> RagConfig:
    return RagConfig(**load_yaml(path))