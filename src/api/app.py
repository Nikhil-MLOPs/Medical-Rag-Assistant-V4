import json
import time
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.rag.rag_service import RagService
from src.utils.logging import setup_logging


logger = setup_logging("API")
app = FastAPI(title="Medical RAG API")

rag_service = RagService()  # singleton


class ChatRequest(BaseModel):
    query: str


@app.get("/")
def root():
    return {"message": "Medical RAG API running"}


@app.post("/chat")
def chat(request: ChatRequest):

    def stream():
        logger.info(f"Incoming Query: {request.query}")

        start_total = time.time()

        yield json.dumps({"status": "thinking"}) + "\n"

        # --- Retrieval timing ---
        start_retrieval = time.time()
        memory_context = ""
        if rag_service.memory:
            memory_context = rag_service.memory.get_context()

        retrieval = rag_service.retriever.retrieve(
            request.query,
            memory_context
        )
        retrieval_time = time.time() - start_retrieval

        logger.info(f"Retrieved {len(retrieval.results)} chunks")
        logger.info(f"Retrieval Time: {retrieval_time:.2f}s")

        # --- Build prompt ---
        memory_context = ""
        if rag_service.memory:
            memory_context = rag_service.memory.get_context()

        prompt = rag_service.chain.build_prompt(
            request.query,
            retrieval.results,
            memory_context,
        )

        # --- LLM timing ---
        start_llm = time.time()

        stream_response = rag_service.llm.chat(
            model=rag_service.cfg.ollama_model,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": rag_service.cfg.temperature,
                "num_predict": rag_service.cfg.max_tokens,
            },
            stream=True,
        )

        full_answer = ""

        for chunk in stream_response:
            token = chunk["message"]["content"]
            full_answer += token
            yield json.dumps({"token": token}) + "\n"

        llm_time = time.time() - start_llm
        total_time = time.time() - start_total
        logger.info(f"LLM Time: {llm_time:.2f}s")
        logger.info(f"Total Time: {total_time:.2f}s")
        logger.info(f"Answer Length: {len(full_answer)} characters")

        # --- Memory update ---
        if rag_service.memory:
            rag_service.memory.add(request.query, full_answer)

        # --- Prepare sources ---
        sources = []
        for idx, r in enumerate(retrieval.results):
            sources.append({
                "id": idx + 1,
                "document": r.text[:200],
                "page": r.metadata.get("page"),
                "pdf": r.metadata.get("pdf"),
                "topic": r.metadata.get("topic"),
                "section": r.metadata.get("section")
            })

        yield json.dumps({
            "done": True,
            "timing": {
                "retrieval_time": retrieval_time,
                "llm_time": llm_time,
                "total_time": total_time,
            },
            "sources": sources,
        }) + "\n"

    return StreamingResponse(stream(), media_type="application/json")


@app.get("/health")
def health():
    return {"status": "healthy"}