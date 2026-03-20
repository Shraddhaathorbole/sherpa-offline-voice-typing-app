import tkinter as tk
from tkinter import messagebox

from app.audio import AudioCapture
from app.engine import SpeechEngine


class VoiceTypingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Offline Voice Typing")
        self.root.geometry("1000x600")
        self.root.configure(bg="#2d2d2d")

        self.running = False
        self.text_data = ""

        self.engine = SpeechEngine(self.on_text)
        self.audio = AudioCapture(self.on_audio)

        self.build_ui()

    def build_ui(self):
        tk.Label(
            self.root,
            text="OFFLINE VOICE TYPING",
            font=("Arial", 24, "bold"),
            bg="#2d2d2d",
            fg="white",
        ).pack(pady=10)

        self.status_label = tk.Label(
            self.root,
            text="Status: Idle",
            font=("Arial", 14),
            bg="#2d2d2d",
            fg="white",
        )
        self.status_label.pack()

        self.preview_label = tk.Label(
            self.root,
            text="",
            font=("Arial", 12),
            bg="#2d2d2d",
            fg="#cfcfcf",
        )
        self.preview_label.pack()

        self.textbox = tk.Text(
            self.root,
            font=("Arial", 16),
            bg="#111",
            fg="white",
            insertbackground="white",
        )
        self.textbox.pack(fill="both", expand=True, padx=20, pady=20)

        frame = tk.Frame(self.root, bg="#2d2d2d")
        frame.pack()

        tk.Button(frame, text="Start", command=self.start, width=10).pack(side="left", padx=10)
        tk.Button(frame, text="Stop", command=self.stop, width=10).pack(side="left", padx=10)
        tk.Button(frame, text="Clear", command=self.clear, width=10).pack(side="left", padx=10)

    def on_audio(self, audio):
        if self.running:
            try:
                self.engine.add_audio(audio)
            except Exception as e:
                print("Engine error:", e)

    def on_text(self, text):
        self.root.after(0, lambda: self.update_text(text))

    def update_text(self, new_text):
        if not new_text:
            return

        new_text = new_text.strip()

        if self.text_data:
            self.text_data = (self.text_data + " " + new_text).strip()
        else:
            self.text_data = new_text

        self.preview_label.config(text=new_text)

        self.textbox.delete("1.0", tk.END)
        self.textbox.insert(tk.END, self.text_data)
        self.textbox.see(tk.END)

    def start(self):
        if self.running:
            return

        self.engine.reset()
        self.audio.stop()
        self.audio.start()
        self.running = True
        self.status_label.config(text="Status: Listening")

    def stop(self):
        if not self.running:
            return

        self.running = False
        self.audio.stop()
        self.engine.reset()
        self.status_label.config(text="Status: Idle")
        self.preview_label.config(text="")

    def clear(self):
        self.text_data = ""
        self.preview_label.config(text="")
        self.textbox.delete("1.0", tk.END)
        self.engine.reset()