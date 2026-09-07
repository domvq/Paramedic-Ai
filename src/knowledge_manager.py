import json
import subprocess
import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent

DOCUMENTS_DIR = (
    PROJECT_ROOT
    / "data"
    / "documents"
)

REGISTRY_PATH = (
    PROJECT_ROOT
    / "data"
    / "knowledge"
    / "sources.json"
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


def ingest_uploaded_file(
    uploaded_file,
):

    DOCUMENTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        DOCUMENTS_DIR
        / uploaded_file.name
    )

    output_path.write_bytes(
        uploaded_file.getbuffer()
    )

    return output_path


def run_ingestion():

    result = subprocess.run(
        [
            sys.executable,
            str(
                PROJECT_ROOT
                / "src"
                / "ingest.py"
            ),
        ],
        capture_output=True,
        text=True,
    )

    return result


def run_index_build():

    result = subprocess.run(
        [
            sys.executable,
            str(
                PROJECT_ROOT
                / "src"
                / "build_index.py"
            ),
        ],
        capture_output=True,
        text=True,
    )

    return result


# ==================================================
# KNOWLEDGE MANAGEMENT
# ==================================================

st.header("📚 Knowledge Manager")

st.caption(
    "Add authorized EMS reference documents to the "
    "project knowledge base."
)


uploaded_file = st.file_uploader(
    "Upload an EMS reference document",
    type=["pdf", "docx"],
)


if uploaded_file:

    file_path = ingest_uploaded_file(
        uploaded_file
    )

    st.success(
        f"Uploaded: {uploaded_file.name}"
    )


    registry = load_registry()

    existing = registry.get(
        uploaded_file.name,
        {},
    )


    st.subheader("Source Metadata")

    title = st.text_input(
        "Title",
        value=existing.get(
            "title",
            Path(
                uploaded_file.name
            ).stem,
        ),
    )


    jurisdiction = st.text_input(
        "Jurisdiction",
        value=existing.get(
            "jurisdiction",
            "UNSPECIFIED",
        ),
        help=(
            "Example: California, Sacramento County, "
            "Agency name, etc."
        ),
    )


    document_type = st.selectbox(
        "Document type",
        [
            "educational_reference",
            "agency_protocol",
            "medical_director_order",
            "manufacturer_reference",
            "other",
        ],
        index=0,
    )


    effective_date = st.text_input(
        "Effective date",
        value=existing.get(
            "effective_date",
            "",
        ) or "",
        placeholder="YYYY-MM-DD",
    )


    version = st.text_input(
        "Version",
        value=existing.get(
            "version",
            "",
        ) or "",
    )


    status = st.selectbox(
        "Status",
        [
            "active",
            "draft",
            "inactive",
            "retired",
            "superseded",
        ],
    )


    authority = st.text_input(
        "Authority / issuing organization",
        value=existing.get(
            "authority",
            "",
        ) or "",
    )


    review_required = st.checkbox(
        "Requires review",
        value=existing.get(
            "review_required",
            True,
        ),
    )


    if st.button(
        "💾 Save Metadata",
        type="primary",
    ):

        registry[
            uploaded_file.name
        ] = {

            "title": title,

            "jurisdiction": jurisdiction,

            "document_type": document_type,

            "effective_date": (
                effective_date
                if effective_date
                else None
            ),

            "version": (
                version
                if version
                else None
            ),

            "status": status,

            "authority": authority,

            "review_required": review_required,
        }


        save_registry(
            registry
        )


        st.success(
            "Metadata saved."
        )


    if st.button(
        "🔄 Ingest & Rebuild Index",
    ):

        with st.spinner(
            "Processing document..."
        ):

            ingestion = run_ingestion()

            if ingestion.returncode != 0:

                st.error(
                    "Document ingestion failed."
                )

                if ingestion.stderr:

                    st.code(
                        ingestion.stderr
                    )

            else:

                st.success(
                    "Document ingestion complete."
                )


                index_result = (
                    run_index_build()
                )


                if (
                    index_result.returncode
                    != 0
                ):

                    st.error(
                        "Index build failed."
                    )

                    if index_result.stderr:

                        st.code(
                            index_result.stderr
                        )

                else:

                    st.success(
                        "Knowledge index rebuilt."
                    )


                    if index_result.stdout:

                        st.code(
                            index_result.stdout
                        )


# ==================================================
# CURRENT LIBRARY
# ==================================================

st.divider()

st.subheader(
    "Registered Sources"
)


registry = load_registry()


if not registry:

    st.info(
        "No sources registered."
    )

else:

    for filename, metadata in registry.items():

        title = metadata.get(
            "title",
            filename,
        )

        document_type = metadata.get(
            "document_type",
            "UNKNOWN",
        )

        jurisdiction = metadata.get(
            "jurisdiction",
            "UNSPECIFIED",
        )

        status = metadata.get(
            "status",
            "UNKNOWN",
        )

        review_required = metadata.get(
            "review_required",
            True,
        )


        with st.expander(
            f"📄 {title}"
        ):

            st.write(
                f"**File:** {filename}"
            )

            st.write(
                f"**Type:** {document_type}"
            )

            st.write(
                f"**Jurisdiction:** {jurisdiction}"
            )

            st.write(
                f"**Status:** {status}"
            )


            if review_required:

                st.warning(
                    "⚠️ Review required"
                )

            else:

                st.success(
                    "✓ Review complete"
                )
