# from pathlib import Path
# import ollama

# from src.utils.config import load_rag_config
# from src.retrieval.retriever import RetrieverService
# from src.rag.memory import ConversationMemory
# from src.rag.guardrails import Guardrails
# from src.rag.explainability import generate_explanation
# from src.rag.chain import RagChain
# from src.rag.schema import RagResponse, Citation

# CONFIG_PATH = Path("configs/rag.yaml")


# class RagService:

#     def __init__(self):

#         self.cfg = load_rag_config(CONFIG_PATH)

#         # No static caching
#         self.llm = ollama

#         self.retriever = RetrieverService()

#         self.memory = (
#             ConversationMemory() if self.cfg.use_memory else None
#         )

#         self.guardrails = (
#             Guardrails() if self.cfg.guardrails_enabled else None
#         )

#         self.chain = RagChain()

#     def ask(self, query: str) -> RagResponse:

#         if self.guardrails:
#             self.guardrails.validate(query)

#         retrieval_response = self.retriever.retrieve(query)

#         memory_context = ""
#         if self.memory:
#             memory_context = self.memory.get_context()

#         prompt = self.chain.build_prompt(
#             query,
#             retrieval_response.results,
#             memory_context,
#         )

#         response = self.llm.chat(
#             model=self.cfg.ollama_model,
#             messages=[{"role": "user", "content": prompt}],
#             options={
#                 "temperature": self.cfg.temperature,
#                 "num_predict": self.cfg.max_tokens,
#             },
#         )

#         answer = response["message"]["content"]

#         if self.memory:
#             self.memory.add(query, answer)

#         explanation = None
#         if self.cfg.explainability_enabled:
#             explanation = generate_explanation(
#                 retrieval_response.results
#             )

#         citations = [
#             Citation(text=r.text, metadata=r.metadata)
#             for r in retrieval_response.results
#         ]

#         return RagResponse(
#             query=query,
#             answer=answer,
#             citations=citations,
#             explanation=explanation,
#         )

# Phase-7

from pathlib import Path
import ollama
import time

from src.utils.config import load_rag_config
from src.retrieval.retriever import RetrieverService
from src.rag.memory import ConversationMemory
from src.rag.guardrails import Guardrails
from src.rag.explainability import generate_explanation
from src.rag.chain import RagChain
from src.rag.schema import RagResponse, Citation
from src.rag.query_rewriter import QueryRewriter
from src.rag.query_intent import detect_section

CONFIG_PATH = Path("configs/rag.yaml")


class RagService:

    def __init__(self, config_override: dict | None = None):

        self.cfg = load_rag_config(CONFIG_PATH)

        if config_override:
            for key, value in config_override.items():
                if hasattr(self.cfg, key):
                    setattr(self.cfg, key, value)

        self.llm = ollama
        self.retriever = RetrieverService(config_override=config_override)

        self.memory = (
            ConversationMemory() if self.cfg.use_memory else None
        )

        self.guardrails = (
            Guardrails() if self.cfg.guardrails_enabled else None
        )

        self.chain = RagChain()
        self.rewriter = QueryRewriter()

    # -------------------------------------------------
    # Blocking call
    # -------------------------------------------------
    def ask(self, query: str) -> RagResponse:

        full_answer = ""

        for token in self.stream(query):
            full_answer += token

        return self._finalize_response(query, full_answer)

    # -------------------------------------------------
    # Streaming call
    # -------------------------------------------------
    def stream(self, query: str):

        if self.guardrails:
            self.guardrails.validate(query)

        memory_context = ""
        topic = None

        # -------------------------------------------------
        # MEMORY + TOPIC DETECTION
        # -------------------------------------------------
        if self.memory:

            memory_context = self.memory.get_context()

            # Detect topic if user introduces one
            detected_topic = self.memory.detect_topic(query)

            if detected_topic:
                self.memory.active_topic = detected_topic.lower()

            topic = self.memory.get_active_topic()

        # -------------------------------------------------
        # INTENT DETECTION
        # -------------------------------------------------
        section = detect_section(query)

        # -------------------------------------------------
        # QUERY REWRITING
        # -------------------------------------------------
        history = self.memory.history if self.memory else []
        rewritten_query = self.rewriter.rewrite(query, history, topic)

        # BUILD RETRIEVAL QUERY (Improved Follow-up Grounding)
        # -------------------------------------------------

        query_for_retrieval = rewritten_query

        if topic:

            q_lower = rewritten_query.lower()

            followup_markers = [
                "how is it",
                "how can it",
                "what are its",
                "what are the symptoms",
                "what are the causes",
                "how is it treated",
                "how can it be prevented",
            ]

            # Only inject topic for follow-up questions
            if any(marker in q_lower for marker in followup_markers):
                query_for_retrieval = f"{topic} {rewritten_query}"
            else:
                query_for_retrieval = rewritten_query

        # DEBUG (optional but useful)
        print("\n================ RETRIEVAL INPUT ================")
        print("Original query:", query)
        print("Rewritten query:", rewritten_query)
        print("Active topic:", topic)
        print("Final retrieval query:", query_for_retrieval)
        print("Section:", section)
        print("=================================================\n")

        # -------------------------------------------------
        # RETRIEVAL
        # -------------------------------------------------
        

        start_retrieval = time.time()

        retrieval_response = self.retriever.retrieve(
            query_for_retrieval,
            topic=topic,
            section=section
        )

        self._retrieval_time = time.time() - start_retrieval

        # store retrieval for final response
        self._last_retrieval = retrieval_response

        # -------------------------------------------------
        # PROMPT BUILDING
        # -------------------------------------------------
        prompt = self.chain.build_prompt(
            query,
            retrieval_response.results,
            memory_context,
        )

        # -------------------------------------------------
        # LLM CALL
        # -------------------------------------------------
        response = self.llm.chat(
            model=self.cfg.ollama_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a medical assistant. "
                        "Answer ONLY using the provided sources. "
                        "If the answer is not found in the sources, say so."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            options={
                "temperature": self.cfg.temperature,
                "num_predict": self.cfg.max_tokens,
            },
            stream=True,
        )

        collected = ""

        # -------------------------------------------------
        # STREAMING HANDLER
        # -------------------------------------------------
        if isinstance(response, dict):

            token = response["message"]["content"]
            collected += token
            yield token

        else:

            for chunk in response:
                token = chunk["message"]["content"]
                collected += token
                yield token

        # -------------------------------------------------
        # MEMORY UPDATE
        # -------------------------------------------------
        if self.memory:
            self.memory.add(query, collected)

    # -------------------------------------------------
    # FINAL STRUCTURED RESPONSE
    # -------------------------------------------------
    def _finalize_response(self, query: str, answer: str):

        retrieval_response = getattr(self, "_last_retrieval", None)

        citations = []
        explanation = None

        if retrieval_response:

            citations = [
                Citation(text=r.text, metadata=r.metadata)
                for r in retrieval_response.results
            ]

            if self.cfg.explainability_enabled:
                explanation = generate_explanation(
                    retrieval_response.results
                )

        retrieval_time = getattr(self, "_retrieval_time", None)

        return RagResponse(
            query=query,
            answer=answer,
            citations=citations,
            explanation=explanation,
            retrieval_time=retrieval_time
        )