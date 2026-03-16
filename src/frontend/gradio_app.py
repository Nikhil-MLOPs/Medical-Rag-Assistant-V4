import gradio as gr
import requests
import json
import uuid

API_URL = "http://rag-api:8001/chat"

# THEME SETUP
custom_theme = gr.themes.Soft(
    primary_hue="cyan",
    secondary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Syne"), "ui-sans-serif", "sans-serif"],
).set(
    body_background_fill="*neutral_950",
    block_background_fill="*neutral_900",
    block_border_width="1px",
    block_title_text_color="*primary_400",
    button_primary_background_fill="*primary_600",
    input_background_fill="*neutral_800",
)

# CSS
custom_css = """
input, textarea {
    color: #ffffff !important;
    background-color: #1f2937 !important;
}

input::placeholder, textarea::placeholder {
    color: #94a3b8 !important;
}

.message-wrap .prose {
    color: #ffffff !important;
}

#project-specs {
    border: 1px solid #22d3ee !important;
    background: #0f172a !important;
    margin-top: 20px !important;
}

#project-specs .prose,
#project-specs p,
#project-specs li {
    color: #f1f5f9 !important;
}

#project-specs h3 {
    color: #22d3ee !important;
}

.gradio-container {
    border: 1px solid #22d3ee !important;
}
"""


def chat_with_backend(message, history, session_id):

    answer = ""

    if session_id is None:
        session_id = str(uuid.uuid4())

    with requests.post(
        API_URL,
        json={"query": message, "session_id": session_id},
        stream=True,
    ) as r:

        for line in r.iter_lines():

            if not line:
                continue

            data = json.loads(line.decode())

            if "status" in data:
                yield "Thinking...", session_id

            elif "token" in data:
                answer += data["token"]
                yield answer, session_id

            elif "done" in data:

                retrieval_time = data["timing"]["retrieval_time"]
                llm_time = data["timing"]["llm_time"]
                total_time = data["timing"]["total_time"]

                sources = data.get("sources", [])

                source_block = "\n\n### 📚 Source of Truth:\n"
                for s in sources:
                    source_block += (
                        f"- [{s['id']}] {s['document']} "
                        f"(PDF: {s.get('pdf')} | Page {s.get('page')} | Topic: {s.get('topic')} | Section: {s.get('section')})\n"
                    )

                final_answer = (
                    f"{answer}\n\n"
                    f"{source_block}\n"
                    f"---\n"
                    f" **Performance:** Retrieval: {retrieval_time:.2f}s | "
                    f"LLM: {llm_time:.2f}s | Total: {total_time:.2f}s"
                )

                yield final_answer, session_id


# UI LAYOUT
with gr.Blocks() as demo:

    session_state = gr.State(None)

    gr.HTML("""
        <div style="text-align: center; padding: 20px;">
            <h1 style="color: #22d3ee; font-weight: 800; font-size: 2.5rem; margin-bottom: 0;">
                ⚕️ Medical-RAG-Assistant-V4
            </h1>
            <p style="color: #94a3b8; font-size: 1.1rem;">
                Secure • Evidence-Based • Clinical Intelligence
            </p>
        </div>
    """)

    chat = gr.ChatInterface(
        fn=chat_with_backend,
        additional_inputs=[session_state],
        additional_outputs=[session_state],
    )

    with gr.Accordion(
        " Project Specs & Architecture",
        open=False,
        elem_id="project-specs",
    ):
        gr.Markdown("""
        ### 🩺 About This Project
        Production-grade Medical RAG Assistant.

        ### 🛠️ Tech Stack
        - FastAPI Backend
        - Ollama (Qwen2.5:1.5B)
        - ChromaDB Vector Store
        - MLflow Experiment Tracking
        - LangSmith Tracing
        - Gradio UI

        Author: Nikhil Bhardwaj
        """)

if __name__ == "__main__":
    demo.launch(
        theme=custom_theme,
        css=custom_css,
        share=True,
    )