from src.retrieval.retriever import RetrieverService


def main():
    retriever = RetrieverService()

    query = "How is diabetes caused?"

    response = retriever.retrieve(query)

    print("\n=========== FINAL RESULTS ===========\n")

    for i, r in enumerate(response.results):
        print(f"[{i}] Score: {r.score}")
        print(r.text[:300])
        print(r.metadata)
        print()


if __name__ == "__main__":
    main()