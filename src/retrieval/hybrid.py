from collections import defaultdict
from src.retrieval.schema import RetrievalResult


class HybridRetriever:

    def __init__(self, alpha: float):
        self.alpha = alpha

    def fuse(self, dense_results, sparse_results, top_k):

        from collections import defaultdict

        combined = defaultdict(lambda: {"dense": 0.0, "sparse": 0.0, "meta": None})

        # Collect raw scores
        for r in dense_results:
            combined[r.text]["dense"] = r.score
            combined[r.text]["meta"] = r.metadata

        for r in sparse_results:
            combined[r.text]["sparse"] = r.score
            combined[r.text]["meta"] = r.metadata

        # Normalize sparse scores
        sparse_scores = [v["sparse"] for v in combined.values()]
        max_sparse = max(sparse_scores) if sparse_scores else 1.0

        fused_results = []

        for text, values in combined.items():

            dense_scores = [v["dense"] for v in combined.values()]
            max_dense = max(dense_scores) if dense_scores else 1.0

            dense_score = values["dense"] / max_dense if max_dense > 0 else 0.0
            sparse_score = values["sparse"] / max_sparse if max_sparse > 0 else 0.0

            final_score = (
                self.alpha * dense_score
                + (1 - self.alpha) * sparse_score
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