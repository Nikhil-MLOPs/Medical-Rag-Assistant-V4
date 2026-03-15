class RagChain:

    def build_prompt(self, query, retrieved_chunks, memory_context=""):

        context_blocks = []

        for i, r in enumerate(retrieved_chunks):

            topic = r.metadata.get("topic", "Unknown")
            section = r.metadata.get("section", "Unknown")
            page = r.metadata.get("page", "Unknown")

            block = f"""
[Source {i+1}]
Topic: {topic}
Section: {section}
Page: {page}

Content:
{r.text}
"""
            context_blocks.append(block.strip())

        context_block = "\n\n-----\n\n".join(context_blocks)

        prompt = f"""
You are a careful medical assistant.

You must answer the question ONLY using the provided sources.

Rules:
1. Use ONLY the information from the retrieved sources.
2. Extract the answer directly from the source text whenever possible.
3. If multiple sources mention the answer, summarize them briefly.
4. Always cite sources using [Source #].
5. Only say "The provided documents do not contain enough information"
   if NONE of the sources mention the answer.
6. Keep the answer concise (2–3 sentences).

Conversation History:
{memory_context}

Retrieved Sources:
{context_block}

User Question:
{query}

Answer (use the sources above):

Step 1: Identify which source contains the answer.
Step 2: Extract the relevant sentence.
Step 3: Answer the question with citations.

Final Answer:
"""

        return prompt.strip()