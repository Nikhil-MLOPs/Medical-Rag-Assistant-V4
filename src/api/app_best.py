import json
import time
import yaml
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.rag.rag_service import RagService
from src.utils.logging import setup_logging
from dotenv import load_dotenv
load_dotenv()


logger = setup_logging("API_BEST")
app = FastAPI(title="Medical RAG API (Best Config)")

# Load best config
with open("configs/best_config.yaml") as f:
    best_config = yaml.safe_load(f)

# Cast numeric params (MLflow saves them as strings)
int_keys = ["top_k_dense", "top_k_sparse", "top_k_final"]
float_keys = ["hybrid_alpha", "temperature", "reranker_boost"]

for k in int_keys:
    if k in best_config:
        best_config[k] = int(best_config[k])

for k in float_keys:
    if k in best_config:
        best_config[k] = float(best_config[k])


# Initialize RAG with best config
rag_service = RagService(config_override=best_config)


class ChatRequest(BaseModel):
    query: str


@app.get("/")
def root():
    return {"message": "Medical RAG API running (Best Config)"}


@app.post("/chat")
def chat(request: ChatRequest):

    def stream():

        logger.info(f"Incoming Query: {request.query}")

        start_total = time.time()

        yield json.dumps({"status": "thinking"}) + "\n"

        # start_llm = time.time()

        full_answer = ""

        # USE THE RAG PIPELINE
        for token in rag_service.stream(request.query):
            full_answer += token
            yield json.dumps({"token": token}) + "\n"

        total_time = time.time() - start_total

        retrieval_time = getattr(rag_service, "_retrieval_time", 0)

        llm_time = total_time - retrieval_time

        retrieval = getattr(rag_service, "_last_retrieval", None)

        sources = []

        if retrieval:
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