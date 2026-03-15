import ollama


class QueryRewriter:

    def __init__(self, model="qwen2.5:1.5b"):
        self.model = model

    def rewrite(self, query: str, history: str, topic: str | None):

        pronouns = ["it", "they", "this", "that", "these", "those"]

        if not any(p in query.lower().split() for p in pronouns):
            return query

        prompt = f"""
You rewrite follow-up medical questions into standalone queries.

Rules:
1. Use ONLY the most recent medical condition from the conversation.
2. Do NOT mix multiple diseases.
3. Replace pronouns like "it" or "this" with the correct condition.
4. If the topic cannot be determined, keep the question unchanged.

Current medical topic:
{topic}

Conversation history:
{history}

User question:
{query}

Return ONLY the rewritten standalone query.
"""

        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0}
        )

        rewritten = response["message"]["content"].strip()

        if len(rewritten) < 5:
            return query

        return rewritten