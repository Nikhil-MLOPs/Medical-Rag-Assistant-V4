
# ⚕️ Medical-RAG-Assistant-V4

A **production-grade Retrieval-Augmented Generation (RAG) system** designed for **clinically grounded medical question answering**.

The system retrieves information **directly from medical textbook PDFs** and generates answers **strictly grounded in sources**, eliminating hallucinations and ensuring **traceable citations**.

---

## 🚀 Features

- ✔ Hybrid Retrieval (**Dense + Sparse + Reranking**)
- ✔ PDF-based medical knowledge base
- ✔ **Query rewriting + intent detection** for follow-up questions
- ✔ **Conversation memory with topic tracking**
- ✔ **Streaming responses** for real-time answers
- ✔ **LangSmith tracing** for observability
- ✔ **MLflow experiment tracking** for configuration optimization
- ✔ **Gradio UI** for interactive chat
- ✔ **Dockerized deployment**

---

## 🧠 System Architecture

```
User
│
▼
Gradio UI
│
▼
FastAPI Backend
│
├── Query Rewriter
├── Intent Detection
├── Hybrid Retriever
│   ├── Dense Retrieval (Sentence Transformers)
│   ├── Sparse Retrieval (BM25)
│   └── Reranker (Cross Encoder)
│
├── Prompt Builder
├── LLM (Ollama)
│
└── Citation Generator
```

All answers are generated **only from retrieved documents**.

---

## 📚 Knowledge Base

The system uses a **medical textbook corpus** containing:

- Gale Encyclopedia of Medicine
- Disease definitions
- Symptoms, Causes, Treatments, Prevention

Each answer includes **exact citations with page references**.

**Example:**

```
📚 Sources

[1] Gale Encyclopedia of Medicine
Page 1931 — Hypertension — Definition
```

---

## 🔎 Example Query

**User**
```
What causes diabetes?
```

**Response**
```
The causes of diabetes mellitus are believed to involve both hereditary and environmental factors.
Research indicates that individuals who develop diabetes may have common genetic markers.
In Type I diabetes, the immune system may be triggered by a virus or microorganism that destroys
insulin-producing cells in the pancreas. In Type II diabetes, factors such as age, obesity, and
family history play a role.

Sources:
[1] Gale Encyclopedia of Medicine — Page 1186
```

---

## 📊 Experiment Tracking

The project uses **MLflow** to identify the best-performing RAG configuration.

Experiments optimize:

- Dense vs sparse retrieval weights
- Reranker boost values
- Top-k retrieval parameters
- Temperature and generation settings

> The best configuration is automatically loaded at runtime.

---

## 🔍 Observability with LangSmith

LangSmith tracing provides full visibility into the RAG pipeline.

**Trace example:**

```
rag_stream_pipeline
│
├── query_rewrite
├── retrieval
├── prompt_building
└── llm_generation
```

This allows debugging:
- Retrieval failures
- Prompt issues
- Hallucination sources

---

## 🖥️ User Interface

The assistant includes a custom **Gradio UI** with:

- Dark medical theme
- Streaming answers
- Source citations
- Performance metrics

**Example:**
```
Performance:
Retrieval: 0.41s | LLM: 6.12s | Total: 6.53s
```

---

## ⚡ Performance

| Stage      | Time     |
|------------|----------|
| Retrieval  | ~0.4s    |
| LLM        | ~5–7s    |
| Total      | ~6s      |

---

## 🐳 Docker Deployment

The entire system is containerized.

**Build**
```bash
docker-compose build
```

**Run**
```bash
docker-compose up
```

**Access**

| Service | URL |
|---------|-----|
| API | http://localhost:8001 |
| UI  | http://localhost:7860 |

---

## 🛠️ Tech Stack

| Component           | Technology           |
|---------------------|----------------------|
| Backend             | FastAPI              |
| UI                  | Gradio               |
| Vector Store        | ChromaDB             |
| Embeddings          | Sentence Transformers|
| Sparse Search       | BM25                 |
| Reranking           | Cross Encoder        |
| LLM                 | Ollama               |
| Observability       | LangSmith            |
| Experiment Tracking | MLflow               |
| Deployment          | Docker               |

---

## 📁 Project Structure

```
Medical-RAG-Assistant-V4
│
├── src
│   ├── api
│   │   └── app_best.py
│   │
│   ├── rag
│   │   ├── rag_service.py
│   │   ├── chain.py
│   │   ├── memory.py
│   │   ├── query_rewriter.py
│   │   └── query_intent.py
│   │
│   ├── retrieval
│   │   ├── dense.py
│   │   ├── sparse.py
│   │   ├── hybrid.py
│   │   └── reranker.py
│   │
│   └── frontend
│       └── gradio_app.py
│
├── configs
│   ├── rag.yaml
│   └── best_config.yaml
│
├── docker-compose.yml
├── Dockerfile
└── README.md
```

---

## 🧪 Testing

Unit and integration tests are included.

**Run tests:**
```bash
pytest
```

---

## 🔐 Guardrails

The system includes safeguards:

- No hallucinated medical advice
- Answers strictly grounded in documents
- Clear fallback when sources lack information

---

## 📌 Roadmap

Future improvements:

- [ ] Evaluation benchmarks for RAG accuracy
- [ ] Medical NER for better retrieval
- [ ] GPU acceleration for reranking
- [ ] Production deployment with Kubernetes

---

## 👨‍💻 Author

**Nikhil Bhardwaj**  
Machine Learning & AI Engineer

---

## ⭐ Support

If you find this project useful, consider giving the repository a **star ⭐**