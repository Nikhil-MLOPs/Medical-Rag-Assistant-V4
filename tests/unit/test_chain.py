from src.rag.chain import RagChain
from src.retrieval.schema import RetrievalResult


def test_chain_builds_prompt():

    chain = RagChain()

    results = [
        RetrievalResult(text="Medical info", metadata={}, score=1.0)
    ]

    prompt = chain.build_prompt("What?", results)

    assert "Medical info" in prompt
    assert "What?" in prompt