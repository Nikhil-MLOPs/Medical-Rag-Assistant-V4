import numpy as np
from typing import List


def _get_source_page(item):
    """
    Safely extract source + page from either:
    - Citation object
    - dict format
    """

    # Citation object case
    if hasattr(item, "metadata"):
        return (
            item.metadata.get("source"),
            item.metadata.get("page"),
        )

    # dict case
    if isinstance(item, dict):
        return (
            item.get("source"),
            item.get("page"),
        )

    return None, None


def precision_at_k(retrieved_sources, expected_sources):

    if not retrieved_sources:
        return 0.0

    correct = 0

    for r in retrieved_sources:
        for e in expected_sources:

            if (
                r.metadata.get("pdf") == e["pdf"]
                and r.metadata.get("page") == e["page"]
            ):
                correct += 1
                break

    return correct / len(retrieved_sources)


def recall_at_k(retrieved_sources, expected_sources):

    if not expected_sources:
        return 0.0

    correct = 0

    for e in expected_sources:
        for r in retrieved_sources:

            if (
                r.metadata.get("pdf") == e["pdf"]
                and r.metadata.get("page") == e["page"]
            ):
                correct += 1
                break

    return correct / len(expected_sources)


def citation_coverage(answer_sources, expected_sources):
    return recall_at_k(answer_sources, expected_sources)


def answer_relevance_score(answer: str, expected_answer: str):

    overlap = set(answer.lower().split()) & set(expected_answer.lower().split())

    return len(overlap) / max(len(expected_answer.split()), 1)


def faithfulness_score(answer: str, retrieved_texts: List[str]):

    joined = " ".join(retrieved_texts).lower()

    hallucinated_words = [
        w for w in answer.lower().split()
        if w not in joined
    ]

    hallucination_ratio = len(hallucinated_words) / max(len(answer.split()), 1)

    return 1 - hallucination_ratio


def latency_penalty(latency, threshold):

    if latency <= threshold:
        return 1.0

    return max(0.0, 1 - (latency - threshold) / threshold)