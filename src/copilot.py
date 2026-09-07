import os

from groq import Groq


# ============================================================
# GROQ CONFIGURATION
# ============================================================

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-20b",
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

def chat(
    messages,
    context=None,
    temperature=0.2,
    max_tokens=1024,
):
    """
    Send a conversation to Groq.

    context is optional so existing app.py calls that provide
    context=... continue to work.
    """

    try:
        # Add knowledge context to the conversation when provided.
        if context:
            messages = list(messages)

            messages.insert(
                0,
                {
                    "role": "system",
                    "content": (
                        "Use the following medical/knowledge-base context "
                        "when relevant. Do not invent information that is "
                        "not supported by the context.\n\n"
                        f"{context}"
                    ),
                },
            )

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Copilot error: {e}"
