import os

from groq import Groq


# ============================================================
# GROQ CONFIGURATION
# ============================================================

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "llama-3.1-8b-instant",
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not configured.")

client = Groq(
    api_key=GROQ_API_KEY
)


# ============================================================
# CHAT
# ============================================================

def chat(messages, temperature=0.2, max_tokens=1024):
    """
    Send a conversation to Groq and return the assistant response.
    """

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Copilot error: {e}"

