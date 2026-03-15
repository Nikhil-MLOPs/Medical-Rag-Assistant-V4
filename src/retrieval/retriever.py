from pathlib import Path

from src.utils.config import load_retrieval_config
from src.retrieval.schema import RetrievalResponse
from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse import SparseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import Reranker
from src.utils.logging import setup_logging

logger = setup_logging("RetrieverService")


CONFIG_PATH = Path("configs/retrieval.yaml")


class RetrieverService:

    # def __init__(self):
    #     self.cfg = load_retrieval_config(CONFIG_PATH)
    def __init__(self, config_override: dict | None = None):
        self.cfg = load_retrieval_config(CONFIG_PATH)

        if config_override:
            for key, value in config_override.items():
                if hasattr(self.cfg, key):
                    setattr(self.cfg, key, value)

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

        # self.reranker = Reranker(
        #     model_name=self.cfg.reranker_model_name
        # )
        self.reranker = Reranker(
        model_name=self.cfg.reranker_model_name,
        boost_strength=getattr(self.cfg, "reranker_boost", 0.3)
)

    def retrieve(self, query: str, topic: str | None = None, section: str | None = None):

        print("\n================ RETRIEVAL DEBUG ================")
        print("Incoming query:", query)
        print("Topic constraint:", topic)
        print("Section constraint:", section)
        print("=================================================\n")

        topic_lower = topic.lower() if topic else None

        dense_results = self.dense.search(
            query, self.cfg.top_k_dense
        )

        print("\n========== DENSE RESULTS ==========")
        for i, r in enumerate(dense_results[:5]):
            print(f"[{i}] score={r.score}")
            print(r.text[:200])
            print()

        sparse_results = self.sparse.search(
            query, self.cfg.top_k_sparse, topic
        )

        print("\n========== SPARSE RESULTS ==========")
        for i, r in enumerate(sparse_results[:5]):
            print(f"[{i}] score={r.score}")
            print(r.text[:200])
            print()

        hybrid_results = self.hybrid.fuse(
            dense_results,
            sparse_results,
            self.cfg.top_k_dense * 2
        )

        print("\n========== HYBRID RESULTS ==========")
        for i, r in enumerate(hybrid_results[:5]):
            print(f"[{i}] score={r.score}")
            print(r.text[:200])
            print()

        # Rerank
        final_results = self.reranker.rerank(
            query, hybrid_results
        )

        # HARD TOPIC LOCK (after reranking)
        # ------------------------------------------------
        if topic:

            topic_lower = topic.lower()

            topic_locked = [
                r for r in final_results
                if topic_lower in r.metadata.get("topic", "").lower()
            ]

            # only apply if we found matches
            if topic_locked:
                final_results = topic_locked

        # # STRICT topic filtering
        # # ----------------------------
        # if topic:

        #     topic_lower = topic.lower()

        #     topic_matches = [
        #         r for r in hybrid_results
        #         if topic_lower in r.metadata.get("topic", "").lower()
        #     ]

        #     # if we find matches → use only them
        #     if topic_matches:
        #         hybrid_results = topic_matches

        #     # if no matches → keep original results
        #     else:
        #         logger.warning(
        #             f"No topic matches found for topic='{topic}'. Using fallback results."
        #         )

        print("\n========== RERANKED RESULTS ==========")
        for i, r in enumerate(final_results[:5]):
            print(f"[{i}] score={r.score}")
            print("topic:", r.metadata.get("topic"))
            print("section:", r.metadata.get("section"))
            print(r.text[:200])
            print()

        # Section filtering
        if section:

            section_lower = section.lower()

            section_matches = [
                r for r in final_results
                if section_lower in r.metadata.get("section", "").lower()
            ]

            # fallback if nothing matched
            if section_matches:
                final_results = section_matches

        return RetrievalResponse(
            query=query,
            results=final_results[: self.cfg.top_k_final]
        )