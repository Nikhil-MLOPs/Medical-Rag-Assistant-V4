from collections import defaultdict
from src.retrieval.schema import RetrievalResult


class HybridRetriever:

    def __init__(self, alpha: float):
        self.alpha = alpha

    def fuse(self, dense_results, sparse_results, top_k):

        combined = defaultdict(lambda: {"dense": 0, "sparse": 0, "meta": None})

        for r in dense_results:
            combined[r.text]["dense"] = r.score
            combined[r.text]["meta"] = r.metadata

        for r in sparse_results:
            combined[r.text]["sparse"] = r.score
            combined[r.text]["meta"] = r.metadata

        fused_results = []

        for text, values in combined.items():
            final_score = (
                self.alpha * values["dense"]
                + (1 - self.alpha) * values["sparse"]
            )

            fused_results.append(
                RetrievalResult(
                    text=text,
                    metadata=values["meta"],
                    score=final_score,
                )
            )

        fused_results.sort(key=lambda x: x.score, reverse=True)

        return fused_results[:top_k]