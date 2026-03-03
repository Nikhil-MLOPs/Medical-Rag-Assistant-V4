from pathlib import Path

from src.utils.config import load_retrieval_config
from src.retrieval.schema import RetrievalResponse
from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse import SparseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import Reranker


CONFIG_PATH = Path("configs/retrieval.yaml")


class RetrieverService:

    def __init__(self):
        self.cfg = load_retrieval_config(CONFIG_PATH)

        self.dense = DenseRetriever(
            persist_directory=self.cfg.persist_directory,
            collection_name=self.cfg.collection_name,
            model_name=self.cfg.embedding_model_name,
        )

        self.sparse = SparseRetriever(
            chunks_path=self.cfg.chunks_path
        )

        self.hybrid = HybridRetriever(
            alpha=self.cfg.hybrid_alpha
        )

        self.reranker = Reranker(
            model_name=self.cfg.reranker_model_name
        )

    def retrieve(self, query: str):

        dense_results = self.dense.search(
            query, self.cfg.top_k_dense
        )

        sparse_results = self.sparse.search(
            query, self.cfg.top_k_sparse
        )

        hybrid_results = self.hybrid.fuse(
            dense_results,
            sparse_results,
            self.cfg.top_k_final
        )

        final_results = self.reranker.rerank(
            query, hybrid_results
        )

        return RetrievalResponse(
            query=query,
            results=final_results
        )