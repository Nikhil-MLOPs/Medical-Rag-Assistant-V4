from sentence_transformers import CrossEncoder
from src.utils.logging import setup_logging

logger = setup_logging("Reranker")


class Reranker:

    def __init__(self, model_name: str):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, results):

        pairs = [(query, r.text) for r in results]
        scores = self.model.predict(pairs)

        for r, score in zip(results, scores):
            r.score = float(score)

        results.sort(key=lambda x: x.score, reverse=True)

        return results