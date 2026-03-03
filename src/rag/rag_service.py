from pathlib import Path
import ollama

from src.utils.config import load_rag_config
from src.retrieval.retriever import RetrieverService
from src.rag.memory import ConversationMemory
from src.rag.guardrails import Guardrails
from src.rag.explainability import generate_explanation
from src.rag.chain import RagChain
from src.rag.schema import RagResponse, Citation

CONFIG_PATH = Path("configs/rag.yaml")


class RagService:

    def __init__(self):

        self.cfg = load_rag_config(CONFIG_PATH)

        # No static caching
        self.llm = ollama

        self.retriever = RetrieverService()

        self.memory = (
            ConversationMemory() if self.cfg.use_memory else None
        )

        self.guardrails = (
            Guardrails() if self.cfg.guardrails_enabled else None
        )

        self.chain = RagChain()

    def ask(self, query: str) -> RagResponse:

        if self.guardrails:
            self.guardrails.validate(query)

        retrieval_response = self.retriever.retrieve(query)

        memory_context = ""
        if self.memory:
            memory_context = self.memory.get_context()

        prompt = self.chain.build_prompt(
            query,
            retrieval_response.results,
            memory_context,
        )

        response = self.llm.chat(
            model=self.cfg.ollama_model,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": self.cfg.temperature,
                "num_predict": self.cfg.max_tokens,
            },
        )

        answer = response["message"]["content"]

        if self.memory:
            self.memory.add(query, answer)

        explanation = None
        if self.cfg.explainability_enabled:
            explanation = generate_explanation(
                retrieval_response.results
            )

        citations = [
            Citation(text=r.text, metadata=r.metadata)
            for r in retrieval_response.results
        ]

        return RagResponse(
            query=query,
            answer=answer,
            citations=citations,
            explanation=explanation,
        )