import json
import time
from pathlib import Path

from src.experiments.metrics import (
    precision_at_k,
    recall_at_k,
    citation_coverage,
    answer_relevance_score,
    faithfulness_score,
    latency_penalty,
)


class Evaluator:

    def __init__(self, rag_service, config):
        self.rag = rag_service
        self.cfg = config

    def evaluate_sample(self, sample):
        start = time.time()

        response = self.rag.ask(sample["question"])

        total_latency = time.time() - start

        answer = response.answer

        retrieved_sources = response.citations

        retrieved_texts = [c.text for c in retrieved_sources]

        precision = precision_at_k(
            retrieved_sources, sample["expected_sources"]
        )

        recall = recall_at_k(
            retrieved_sources, sample["expected_sources"]
        )

        coverage = citation_coverage(
            retrieved_sources, sample["expected_sources"]
        )

        relevance = answer_relevance_score(
            response.answer, sample["expected_answer"]
        )

        faithfulness = faithfulness_score(
            response.answer, retrieved_texts
        )

        latency_score = latency_penalty(
            total_latency,
            self.cfg.latency_threshold_seconds
        )

        return {
            "precision": precision,
            "recall": recall,
            "citation_coverage": coverage,
            "answer_relevance": relevance,
            "faithfulness": faithfulness,
            "latency": total_latency,
            "latency_score": latency_score,
        }