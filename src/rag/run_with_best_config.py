import yaml
from src.rag.rag_service import RagService

CONFIG = "configs/best_config.yaml"


def cast_types(params):

    int_keys = [
        "top_k_dense",
        "top_k_sparse",
        "top_k_final",
    ]

    float_keys = [
        "hybrid_alpha",
        "temperature",
        "reranker_boost",
    ]

    for k in int_keys:
        if k in params:
            params[k] = int(params[k])

    for k in float_keys:
        if k in params:
            params[k] = float(params[k])

    return params


def main():

    with open(CONFIG) as f:
        best = yaml.safe_load(f)

    best = cast_types(best)

    rag = RagService(config_override=best)

    print("Running RAG with best configuration")

    while True:

        query = input("\nQuestion: ")

        if query == "exit":
            break

        response = rag.ask(query)

        print("\nAnswer:\n", response.answer)


if __name__ == "__main__":
    main()