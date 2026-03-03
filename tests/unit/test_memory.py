from src.rag.memory import ConversationMemory


def test_memory_add_and_context():

    mem = ConversationMemory()

    mem.add("Q1", "A1")
    mem.add("Q2", "A2")

    context = mem.get_context()

    assert "Q1" in context
    assert "A2" in context