import json
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

REGISTRY_PATH = (
    PROJECT_ROOT
    / "data"
    / "knowledge"
    / "sources.json"
)


KNOWLEDGE_AVAILABLE = INDEX_PATH.exists()

KNOWLEDGE_ERROR = None


def load_source_registry():

    if not REGISTRY_PATH.exists():
        return {}

    try:

        with open(
            REGISTRY_PATH,
            "r",
            encoding="utf-8-sig",
        ) as file:

            return json.load(file)

    except Exception as error:

        return {}


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


def search_knowledge(
    question,
    top_k=3,
):

    if not question or not question.strip():
        return []

    try:

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

    except Exception as error:

        global KNOWLEDGE_ERROR

        KNOWLEDGE_ERROR = str(error)

        return []


def format_context(results):

    if not results:
        return ""

    context_parts = []

    for number, result in enumerate(
        results,
        start=1,
    ):

        context_parts.append(
            f"""
SOURCE {number}

Title: {result.get('title', '')}
File: {result.get('source', '')}
Jurisdiction: {result.get('jurisdiction', 'UNSPECIFIED')}
Document type: {result.get('document_type', 'UNKNOWN')}
Effective date: {result.get('effective_date', 'UNKNOWN')}
Version: {result.get('version', 'UNKNOWN')}
Status: {result.get('status', 'UNKNOWN')}
Authority: {result.get('authority', 'UNKNOWN')}
Retrieval score: {result.get('score', '')}

TEXT:
{result.get('text', '')}
"""
        )

    return "\n".join(context_parts)


def format_references(results):

    if not results:
        return []

    references = []

    seen = set()

    for result in results:

        source = result.get(
            "source",
            "Unknown",
        )

        if source in seen:
            continue

        seen.add(source)

        title = result.get(
            "title",
            source,
        )

        jurisdiction = result.get(
            "jurisdiction",
            "UNSPECIFIED",
        )

        document_type = result.get(
            "document_type",
            "UNKNOWN",
        )

        status = result.get(
            "status",
            "UNKNOWN",
        )

        effective_date = result.get(
            "effective_date",
            "UNKNOWN",
        )

        version = result.get(
            "version",
            "UNKNOWN",
        )

        score = result.get("score")

        score_text = ""

        if isinstance(
            score,
            (int, float),
        ):

            score_text = (
                f" • match {score:.2f}"
            )

        references.append(
            f"- **{title}** — "
            f"{document_type} • "
            f"{jurisdiction} • "
            f"{status} • "
            f"effective: {effective_date} • "
            f"version: {version}"
            f"{score_text}"
        )

    return references


def summarize_sources(results):

    if not results:

        return {
            "count": 0,
            "active": 0,
            "unknown": 0,
            "inactive": 0,
            "review_required": 0,
        }

    active = 0
    unknown = 0
    inactive = 0
    review_required = 0

    for result in results:

        status = str(
            result.get(
                "status",
                "UNKNOWN",
            )
        ).lower()

        if status == "active":

            active += 1

        elif status in {
            "inactive",
            "retired",
            "superseded",
        }:

            inactive += 1

        else:

            unknown += 1

        if result.get(
            "review_required",
            False,
        ):

            review_required += 1

    return {
        "count": len(results),
        "active": active,
        "unknown": unknown,
        "inactive": inactive,
        "review_required": review_required,
    }
