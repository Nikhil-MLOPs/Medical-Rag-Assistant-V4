import re


class ConversationMemory:

    def __init__(self):
        self.history = []
        self.active_topic = None

    # -----------------------------
    # Detect topic from user query
    # -----------------------------
    def detect_topic(self, query: str):

        q = query.lower()

        match = re.search(
            r"(what is|define|explain|tell me about)\s+([a-zA-Z\s\-]+)", q
        )

        if not match:
            return None

        topic = match.group(2).strip()

        # normalize common medical names
        topic_map = {
            "diabetes": "diabetes mellitus",
            "high blood pressure": "hypertension",
            "radiotherapy": "radiation therapy",
        }

        return topic_map.get(topic, topic)

    # -----------------------------
    # Add interaction to memory
    # -----------------------------
    def add(self, query: str, answer: str, topic: str | None = None):

        # detect explicit topic in user query
        detected_topic = self.detect_topic(query)

        if detected_topic:
            self.active_topic = detected_topic.lower()

        # fallback: use retriever-detected topic
        elif topic:
            self.active_topic = topic.lower()

        self.history.append({
            "query": query,
            "answer": answer
        })

    # -----------------------------
    # Build context for prompt
    # -----------------------------
    def get_context(self):

        if not self.history:
            return ""

        context = ""

        # use only last 4 turns
        recent_history = self.history[-4:]

        for item in recent_history:

            # filter history by active topic
            if self.active_topic and self.active_topic not in item["query"].lower():
                continue

            context += f"User: {item['query']}\n"
            context += f"Assistant: {item['answer']}\n"

        return context.strip()

    # -----------------------------
    # Current active topic
    # -----------------------------
    def get_active_topic(self):
        return self.active_topic