import json
import os
import threading
import tkinter as tk
from tkinter import messagebox, scrolledtext

import ollama

import sys

sys.path.insert(
    0,
    r"C:\OllamaAI\tools"
)

from knowledge_chat import search

BASE = r"C:\OllamaAI"
SESSION_DIR = os.path.join(BASE, "sessions")
MODEL = "paramedic-ai:latest"


def ensure_sessions():
    os.makedirs(SESSION_DIR, exist_ok=True)


def session_path(name):
    safe = "".join(
        c for c in name
        if c.isalnum() or c in "_-"
    )

    return os.path.join(
        SESSION_DIR,
        safe + ".json"
    )


def save_session(name, messages):

    ensure_sessions()

    with open(
        session_path(name),
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                "session": name,
                "messages": messages
            },
            f,
            indent=2,
            ensure_ascii=False
        )


def load_session(name):

    path = session_path(name)

    if not os.path.exists(path):
        return []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    return data.get("messages", [])


def list_sessions():

    ensure_sessions()

    sessions = []

    for filename in os.listdir(SESSION_DIR):

        if filename.endswith(".json"):

            sessions.append(
                filename[:-5]
            )

    return sorted(sessions)


class SessionApp:

    def __init__(self, root):

        self.root = root

        self.root.title(
            "Paramedic AI — Sessions"
        )

        self.root.geometry(
            "1000x700"
        )

        self.session_name = "default"
        self.messages = []

        self.build_ui()

    def build_ui(self):

        header = tk.Frame(
            self.root,
            bg="#17232d"
        )

        header.pack(fill="x")

        tk.Label(
            header,
            text="PARAMEDIC AI",
            font=("Segoe UI", 20, "bold"),
            fg="white",
            bg="#17232d"
        ).pack(
            side="left",
            padx=20,
            pady=15
        )

        self.session_label = tk.Label(
            header,
            text="Session: default",
            fg="#45d483",
            bg="#17232d"
        )

        self.session_label.pack(
            side="right",
            padx=20
        )

        self.chat = scrolledtext.ScrolledText(
            self.root,
            wrap=tk.WORD,
            font=("Segoe UI", 11),
            bg="#0d141a",
            fg="white",
            insertbackground="white",
            padx=15,
            pady=15
        )

        self.chat.pack(
            fill="both",
            expand=True,
            padx=15,
            pady=15
        )

        bottom = tk.Frame(
            self.root,
            bg="#101820"
        )

        bottom.pack(
            fill="x",
            padx=15,
            pady=(0, 15)
        )

        self.entry = tk.Entry(
            bottom,
            font=("Segoe UI", 12),
            bg="#1b2833",
            fg="white",
            insertbackground="white"
        )

        self.entry.pack(
            side="left",
            fill="x",
            expand=True,
            ipady=10
        )

        self.entry.bind(
            "<Return>",
            self.send
        )

        tk.Button(
            bottom,
            text="Send",
            command=self.send,
            bg="#1976d2",
            fg="white",
            padx=18,
            pady=8
        ).pack(side="left", padx=4)

        tk.Button(
            bottom,
            text="New",
            command=self.new_session,
            bg="#455a64",
            fg="white",
            padx=18,
            pady=8
        ).pack(side="left", padx=4)

        tk.Button(
            bottom,
            text="Save",
            command=self.save,
            bg="#2e7d32",
            fg="white",
            padx=18,
            pady=8
        ).pack(side="left", padx=4)

        tk.Button(
            bottom,
            text="Load",
            command=self.load,
            bg="#6a1b9a",
            fg="white",
            padx=18,
            pady=8
        ).pack(side="left", padx=4)

    def send(self, event=None):

        question = self.entry.get().strip()

        if not question:
            return

        self.entry.delete(0, tk.END)

        self.messages.append({
            "role": "user",
            "content": question
        })

        self.chat.insert(
            tk.END,
            f"You:\n{question}\n\n"
        )

        self.chat.insert(
            tk.END,
            "Paramedic AI:\nThinking...\n\n"
        )

        threading.Thread(
            target=self.ask_ai,
            daemon=True
        ).start()

    def ask_ai(self):

        try:

            question = self.messages[-1]["content"]

            results = search(
                question
            )

            context_parts = []

            for number, result in enumerate(
                results,
                start=1
            ):

                context_parts.append(
                    f"""
SOURCE {number}
Title: {result['title']}
File: {result['source']}
Jurisdiction: {result['jurisdiction']}
Document type: {result['document_type']}
Effective date: {result['effective_date']}
Status: {result['status']}

TEXT:
{result['text']}
"""
                )

            context = "\n".join(
                context_parts
            )

            if context:

                rag_message = {
                    "role": "system",
                    "content": f"""
You are Paramedic AI, an EMS education
and decision-support assistant.

Use the retrieved reference material
below when it is relevant.

Do not invent protocol information.
Do not invent medication doses.
Do not treat TRAINING material as
current protocol.

For real patient care, current local
EMS protocols and medical direction
take precedence.

RETRIEVED REFERENCES:

{context}
"""
                }

                messages_for_ai = [
                    rag_message
                ] + self.messages

            else:

                messages_for_ai = (
                    self.messages
                )

            response = ollama.chat(
                model=MODEL,
                messages=messages_for_ai
            )

            answer = response[
                "message"
            ]["content"]

            if results:

                seen_sources = set()

                references = []

                for result in results:

                    source = result[
                        "source"
                    ]

                    if source in seen_sources:
                        continue

                    seen_sources.add(
                        source
                    )

                    references.append(
                        f"- "
                        f"{result['title']} "
                        f"({result['status']})"
                    )

                answer += (
                    "\n\nReferences used:\n"
                    + "\n".join(
                        references
                    )
                )

            self.messages.append({
                "role": "assistant",
                "content": answer
            })

            save_session(
                self.session_name,
                self.messages
            )

            self.root.after(
                0,
                self.display_answer,
                answer
            )

        except Exception as error:

            self.root.after(
                0,
                self.display_answer,
                f"ERROR:\n{error}"
            )

    def display_answer(self, answer):

        current = self.chat.get(            "1.0",
            tk.END
        )

        marker = (
            "Paramedic AI:\n"
            "Thinking...\n\n"
        )

        position = current.rfind(marker)

        if position != -1:

            self.chat.delete(
                f"1.0+{position}c",
                f"1.0+"
                f"{position + len(marker)}c"
            )

        self.chat.insert(
            tk.END,
            f"Paramedic AI:\n"
            f"{answer}\n\n"
        )

        self.chat.see(tk.END)

    def new_session(self):

        name = self.ask_name(
            "New Session"
        )

        if not name:
            return

        self.session_name = name
        self.messages = []

        self.chat.delete(
            "1.0",
            tk.END
        )

        self.session_label.config(
            text=f"Session: {name}"
        )

        self.chat.insert(
            tk.END,
            f"New session: {name}\n\n"
        )

    def save(self):

        save_session(
            self.session_name,
            self.messages
        )

        messagebox.showinfo(
            "Session",
            f"Saved: {self.session_name}"
        )

    def load(self):

        sessions = list_sessions()

        if not sessions:

            messagebox.showinfo(
                "Sessions",
                "No saved sessions."
            )

            return

        dialog = tk.Toplevel(
            self.root
        )

        dialog.title(
            "Load Session"
        )

        dialog.geometry(
            "350x300"
        )

        tk.Label(
            dialog,
            text="Select a session:"
        ).pack(pady=10)

        listbox = tk.Listbox(
            dialog
        )

        listbox.pack(
            fill="both",
            expand=True,
            padx=20
        )

        for session in sessions:
            listbox.insert(
                tk.END,
                session
            )

        def choose():

            selection = listbox.curselection()

            if not selection:
                return

            name = listbox.get(
                selection[0]
            )

            self.load_named_session(
                name
            )

            dialog.destroy()

        tk.Button(
            dialog,
            text="Load",
            command=choose
        ).pack(pady=10)

    def load_named_session(self, name):

        self.session_name = name

        self.messages = load_session(
            name
        )

        self.chat.delete(
            "1.0",
            tk.END
        )

        self.session_label.config(
            text=f"Session: {name}"
        )

        for message in self.messages:

            label = (
                "You"
                if message["role"] == "user"
                else "Paramedic AI"
            )

            self.chat.insert(
                tk.END,
                f"{label}:\n"
                f"{message['content']}\n\n"
            )

        self.chat.see(tk.END)

    def ask_name(self, title):

        dialog = tk.Toplevel(
            self.root
        )

        dialog.title(title)

        dialog.geometry(
            "400x150"
        )

        tk.Label(
            dialog,
            text="Session name:"
        ).pack(pady=10)

        entry = tk.Entry(
            dialog,
            width=40
        )

        entry.pack()

        result = {
            "value": None
        }

        def confirm():

            value = entry.get().strip()

            if value:
                result["value"] = value
                dialog.destroy()

        tk.Button(
            dialog,
            text="Create",
            command=confirm
        ).pack(pady=15)

        entry.focus()

        self.root.wait_window(
            dialog
        )

        return result["value"]


def main():

    ensure_sessions()

    root = tk.Tk()

    SessionApp(root)

    root.mainloop()


if __name__ == "__main__":
    main()