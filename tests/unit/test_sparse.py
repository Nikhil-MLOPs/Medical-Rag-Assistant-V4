import json
from pathlib import Path
from src.retrieval.sparse import SparseRetriever


def test_sparse_retriever(tmp_path):

    chunks_file = tmp_path / "chunks.jsonl"

    with open(chunks_file, "w") as f:
        f.write(json.dumps({"text": "pneumonia causes bacteria", "metadata": {"topic": "P"}}) + "\n")
        f.write(json.dumps({"text": "heart disease symptoms", "metadata": {"topic": "H"}}) + "\n")

    retriever = SparseRetriever(str(chunks_file))

    results = retriever.search("pneumonia bacteria", top_k=1)

    assert len(results) == 1
    assert results[0].metadata["topic"] == "P"