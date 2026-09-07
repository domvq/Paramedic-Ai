import os

import ollama
from dotenv import load_dotenv


load_dotenv()


OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "paramedic-ai:latest",
)


SYSTEM_PROMPT = """
You are Paramedic AI, an EMS education and
decision-support assistant.

Use retrieved reference material when relevant.

Do not invent EMS protocol information.
Do not invent medication doses.
Do not treat training material as current protocol.

For real patient care, current local EMS protocols
and medical direction take precedence.

This application is for education and demonstration.
"""


def chat(messages, context=""):

    system_prompt = SYSTEM_PROMPT

    if context:
        system_prompt += f"""

RETRIEVED REFERENCES:

{context}
"""

    ollama_messages = [
        {
            "role": "system",
            "content": system_prompt,
        }
    ] + messages

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=ollama_messages,
    )

    return response["message"]["content"]
