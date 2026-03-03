def generate_explanation(results):

    topics = {
        r.metadata.get("topic", "Unknown")
        for r in results
    }

    return f"Answer generated using documents from topics: {', '.join(topics)}"