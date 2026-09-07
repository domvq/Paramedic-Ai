import json
import shutil
from pathlib import Path

from docx import Document


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REGISTRY_PATH = (
    PROJECT_ROOT
    / "data"
    / "knowledge"
    / "sources.json"
)


def extract_docx_text(path):
    """Extract readable text from a Word document."""

    document = Document(path)

    paragraphs = []

    for paragraph in document.paragraphs:

        text = paragraph.text.strip()

        if text:
            paragraphs.append(text)

    return "\n\n".join(paragraphs)


def load_registry():

    if not REGISTRY_PATH.exists():
        return {}

    with open(
        REGISTRY_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


def save_registry(registry):

    REGISTRY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        REGISTRY_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            registry,
            file,
            indent=2,
        )


def ingest_document(path):

    suffix = path.suffix.lower()

    if suffix != ".docx":

        raise ValueError(
            "Currently supported format: .docx"
        )

    text = extract_docx_text(path)

    if not text.strip():

        raise ValueError(
            "The document contains no readable text."
        )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        PROCESSED_DIR
        / f"{path.stem}.txt"
    )

    output_path.write_text(
        text,
        encoding="utf-8",
    )

    registry = load_registry()

    filename = path.name

    if filename not in registry:

        registry[filename] = {
            "title": path.stem,
            "jurisdiction": "UNSPECIFIED",
            "document_type": "educational_reference",
            "effective_date": None,
            "version": None,
            "status": "active",
            "authority": "Unspecified",
            "review_required": True,
        }

    save_registry(registry)

    return output_path


def main():

    DOCUMENTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    documents = list(
        DOCUMENTS_DIR.glob("*.docx")
    )

    if not documents:

        print(
            "No .docx documents found in:"
        )

        print(DOCUMENTS_DIR)

        return

    for document in documents:

        try:

            output = ingest_document(
                document
            )

            print(
                f"INGESTED: {document.name}"
            )

            print(
                f"TEXT: {output}"
            )

        except Exception as error:

            print(
                f"ERROR: {document.name}: {error}"
            )


if __name__ == "__main__":

    main()
