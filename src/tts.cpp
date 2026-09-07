
import sys
import pyttsx3


def speak_text(text):
    if not text:
        print("Text is empty.")
        return

    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 170)
        engine.setProperty("volume", 1.0)

        engine.say(text)
        engine.runAndWait()

        print("Speech completed.")

    except Exception as error:
        print(f"TTS error: {error}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tts.py <text>")
        sys.exit(1)

    text = " ".join(sys.argv[1:])
    speak_text(text)

