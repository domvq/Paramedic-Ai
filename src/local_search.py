import pickle
from pathlib import Path

from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INDEX_PATH = (
    PROJECT_ROOT
    / "data"
    / "knowledge"
    / "index"
    / "knowledge.pkl"
)


def load_index():

    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Knowledge index not found: {INDEX_PATH}"
        )

    with open(
        INDEX_PATH,
        "rb",
    ) as file:
        return pickle.load(file)


def search_local(question, top_k=3):

    if not question or not question.strip():
        return []

    index = load_index()

    vectorizer = index["vectorizer"]
    matrix = index["matrix"]
    metadata = index["metadata"]

    query_vector = vectorizer.transform(
        [question]
    )

    scores = cosine_similarity(
        query_vector,
        matrix,
    )[0]

    ranked = scores.argsort()[::-1]

    results = []

    for index_number in ranked[:top_k]:

        result = dict(
            metadata[index_number]
        )

        result["score"] = round(
            float(scores[index_number]),
            4,
        )

        results.append(result)

    return results


if __name__ == "__main__":

    results = search_local(
        "airway assessment"
    )

    if not results:
        print("NO RESULTS")
    else:
        for result in results:
            print(
                f"{result['score']:.4f} - "
                f"{result['title']}"
            )
