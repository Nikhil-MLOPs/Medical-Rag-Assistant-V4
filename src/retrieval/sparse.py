import json
from rank_bm25 import BM25Okapi
from pathlib import Path

from src.retrieval.schema import RetrievalResult
from src.utils.logging import setup_logging

logger = setup_logging("SparseRetriever")


class SparseRetriever:

    def __init__(self, chunks_path: str):
        logger.info("Initializing SparseRetriever")

        texts = []
        metadatas = []

        with open(Path(chunks_path), "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                texts.append(record["text"])
                metadatas.append(record["metadata"])

        self.texts = texts
        self.metadatas = metadatas

        tokenized_corpus = [doc.split() for doc in texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int):
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        results = []

        for idx in ranked_indices:
            results.append(
                RetrievalResult(
                    text=self.texts[idx],
                    metadata=self.metadatas[idx],
                    score=float(scores[idx])
                )
            )

        return results