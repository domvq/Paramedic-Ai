import json
import pickle
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

INDEX_DIR = (
    PROJECT_ROOT
    / "data"
    / "knowledge"
    / "index"
)

REGISTRY_PATH = (
    PROJECT_ROOT
    / "data"
    / "knowledge"
    / "sources.json"
)

INDEX_PATH = (
    INDEX_DIR
    / "knowledge.pkl"
)


def load_registry():

    if not REGISTRY_PATH.exists():
        return {}

    with open(
        REGISTRY_PATH,
        "r",
        encoding="utf-8-sig",
    ) as file:

        return json.load(file)


def build_index():

    files = list(
        PROCESSED_DIR.glob("*.txt")
    )

    if not files:

        raise RuntimeError(
            "No processed documents found."
        )

    documents = []
    metadata = []

    registry = load_registry()

    for path in files:

        text = path.read_text(
            encoding="utf-8-sig"
        )

        if not text.strip():
            continue

        documents.append(text)

        source_name = (
            path.stem + ".docx"
        )

        source_metadata = registry.get(
            source_name,
            {},
        )

        metadata.append(
            {
                "source": source_name,
                "title": source_metadata.get(
                    "title",
                    path.stem,
                ),
                "jurisdiction": source_metadata.get(
                    "jurisdiction",
                    "UNSPECIFIED",
                ),
                "document_type": source_metadata.get(
                    "document_type",
                    "UNKNOWN",
                ),
                "effective_date": source_metadata.get(
                    "effective_date",
                    "UNKNOWN",
                ),
                "version": source_metadata.get(
                    "version",
                    "UNKNOWN",
                ),
                "status": source_metadata.get(
                    "status",
                    "UNKNOWN",
                ),
                "authority": source_metadata.get(
                    "authority",
                    "UNKNOWN",
                ),
                "review_required": source_metadata.get(
                    "review_required",
                    True,
                ),
                "text": text,
            }
        )

    if not documents:

        raise RuntimeError(
            "No readable documents found."
        )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
    )

    matrix = vectorizer.fit_transform(
        documents
    )

    INDEX_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    index = {
        "vectorizer": vectorizer,
        "matrix": matrix,
        "metadata": metadata,
    }

    with open(
        INDEX_PATH,
        "wb",
    ) as file:

        pickle.dump(
            index,
            file,
        )

    print(
        f"INDEX CREATED: {INDEX_PATH}"
    )

    print(
        f"DOCUMENTS INDEXED: {len(metadata)}"
    )


if __name__ == "__main__":

    build_index()

