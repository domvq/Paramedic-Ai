
import os

from groq import Groq


# ============================================================
# GROQ CONFIGURATION
# ============================================================

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
 ,
)

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are Paramedic AI, an EMS education and decision-support
assistant.

Your purpose is to help users understand EMS concepts,
protocols, assessment principles, and educational material.

Use the provided knowledge context when answering.

Important:
- Do not claim to diagnose patients.
- Do not replace a medical director or local EMS protocol.
- Clearly identify uncertainty.
- Encourage users to follow current local protocols,
  scope of practice, medical direction, and manufacturer
  instructions.
- When the provided knowledge context contains relevant
  information, prioritize it.
"""


# ============================================================
# CHAT
# ============================================================

def chat(messages, context=""):
    """
    Send the conversation and retrieved knowledge context
    to the Groq model.
    """

    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not configured."
        )

    # --------------------------------------------------------
    # Build messages
    # --------------------------------------------------------

    groq_messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]

    # Add retrieved knowledge
    if context:
        groq_messages.append(
            {
                "role": "system",
                "content": (
                    "Use the following authorized EMS "
                    "knowledge context when relevant:\n\n"
                    f"{context}"
                ),
            }
        )

    # Add conversation history
    for message in messages:

        role = message.get("role")

        content = message.get("content", "")

        if role not in ["user", "assistant"]:
            continue

        groq_messages.append(
            {
                "role": role,
                "content": content,
            }
        )

    # --------------------------------------------------------
    # Groq API call
    # --------------------------------------------------------

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=groq_messages,
        temperature=0.2,
    )

    # --------------------------------------------------------
    # Return answer
    # --------------------------------------------------------

    return response.choices[0].message.content

