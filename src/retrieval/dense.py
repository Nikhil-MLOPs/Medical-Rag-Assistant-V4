import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from src.retrieval.schema import RetrievalResult
from src.utils.logging import setup_logging

logger = setup_logging("DenseRetriever")


class DenseRetriever:

    def __init__(self, persist_directory: str, collection_name: str, model_name: str):
        logger.info("Initializing DenseRetriever")

        self.model = SentenceTransformer(model_name)

        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )

        self.collection = self.client.get_collection(name=collection_name)

    def search(self, query: str, top_k: int):
        query_embedding = self.model.encode(
            query, normalize_embeddings=True
        )

        if hasattr(query_embedding, "tolist"):
            query_embedding = query_embedding.tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        retrieval_results = []

        for doc, meta, dist in zip(documents, metadatas, distances):
            score = 1 - dist  # Convert distance to similarity
            retrieval_results.append(
                RetrievalResult(text=doc, metadata=meta, score=score)
            )

        return retrieval_results