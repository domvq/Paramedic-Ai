import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_TOOLS_PATH = Path(r"C:\OllamaAI\tools")

OLLAMA_TOOLS = Path(
    os.getenv(
        "OLLAMA_TOOLS",
        str(DEFAULT_TOOLS_PATH),
    )
)

SOURCE_REGISTRY = (
    PROJECT_ROOT
    / "data"
    / "knowledge"
    / "sources.json"
)


if str(OLLAMA_TOOLS) not in sys.path:
    sys.path.insert(0, str(OLLAMA_TOOLS))


try:
    from knowledge_chat import search as _search

    KNOWLEDGE_AVAILABLE = True
    KNOWLEDGE_ERROR = None

except Exception as error:
    _search = None
    KNOWLEDGE_AVAILABLE = False
    KNOWLEDGE_ERROR = str(error)


def load_source_registry():
    """Load local source metadata."""

    if not SOURCE_REGISTRY.exists():
        return {}

    try:

        with open(
            SOURCE_REGISTRY,
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(file)

    except Exception:
        return {}


def enrich_result(result):
    """Apply local source metadata to a search result."""

    registry = load_source_registry()

    source = result.get(
        "source",
        "",
    )

    metadata = registry.get(
        source,
        {},
    )

    enriched = dict(result)

    for key, value in metadata.items():

        if value is not None:

            enriched[key] = value

    return enriched


def search_knowledge(question):
    """Search the EMS knowledge base."""

    if not KNOWLEDGE_AVAILABLE:
        return []

    if not question or not question.strip():
        return []

    try:

        results = _search(question)

        if results is None:
            return []

        return [
            enrich_result(result)
            for result in results
        ]

    except Exception:
        return []


def format_context(results):
    """Convert knowledge results into AI context."""

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
    """Create clean references for display."""

    if not results:
        return []

    references = []

    for result in results:

        title = result.get(
            "title",
            result.get(
                "source",
                "Unknown",
            ),
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

        score = result.get(
            "score"
        )

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
    """Return source statistics."""

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
