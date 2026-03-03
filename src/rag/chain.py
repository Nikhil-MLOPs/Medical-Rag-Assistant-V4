class RagChain:

    def build_prompt(self, query: str, results, memory_context: str = "") -> str:

        context = "\n\n".join([r.text for r in results])

        prompt = f"""
You are a medical assistant.

Conversation History:
{memory_context}

Context:
{context}

Question:
{query}

Answer strictly using the provided context.
"""

        return prompt.strip()