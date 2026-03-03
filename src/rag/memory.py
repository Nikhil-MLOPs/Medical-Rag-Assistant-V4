class ConversationMemory:

    def __init__(self):
        self.history = []

    def add(self, query: str, answer: str):
        self.history.append({"query": query, "answer": answer})

    def get_context(self) -> str:
        context = ""
        for item in self.history[-5:]:
            context += f"User: {item['query']}\nAssistant: {item['answer']}\n"
        return context.strip()