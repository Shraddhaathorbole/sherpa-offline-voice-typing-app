import tkinter as tk
from tkinter import messagebox
import threading
import time

from app.audio import AudioCapture
from app.engine import SpeechEngine
from app.whisper_finalizer import WhisperFinalizer


class VoiceTypingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Offline Voice Typing")
        self.root.geometry("1000x600")
        self.root.configure(bg="#2f2f2f")

        self.running = False
        self.processing = False
        self.committed_text = ""
        self.live_text = ""
        self.last_final_text = ""
        self.last_audio_time = time.time()

        try:
            self.engine = SpeechEngine(self.on_live_text)
            self.finalizer = WhisperFinalizer()
        except Exception as e:
            messagebox.showerror("Initialization Error", str(e))
            raise

        self.audio = AudioCapture(self.on_audio)
        self.build_ui()

    def build_ui(self):
        self.main_frame = tk.Frame(self.root, bg="#2f2f2f")
        self.main_frame.pack(fill="both", expand=True, padx=16, pady=14)

        self.title_label = tk.Label(
            self.main_frame,
            text="OFFLINE VOICE TYPING",
            font=("Arial", 24, "bold"),
            fg="white",
            bg="#2f2f2f",
        )
        self.title_label.pack(pady=(8, 8))

        self.status_label = tk.Label(
            self.main_frame,
            text="Status: Idle",
            font=("Arial", 14, "bold"),
            fg="white",
            bg="#2f2f2f",
        )
        self.status_label.pack()

        self.preview_label = tk.Label(
            self.main_frame,
            text="",
            font=("Arial", 11),
            fg="#d9d9d9",
            bg="#2f2f2f",
        )
        self.preview_label.pack(pady=(6, 10))

        self.textbox = tk.Text(
            self.main_frame,
            font=("Arial", 16),
            bg="#050505",
            fg="white",
            insertbackground="white",
            wrap="word",
            relief="solid",
            borderwidth=2,
            highlightthickness=1,
            highlightbackground="white",
            highlightcolor="white",
        )
        self.textbox.pack(fill="both", expand=True, pady=(8, 18))

        button_frame = tk.Frame(self.main_frame, bg="#2f2f2f")
        button_frame.pack(pady=(2, 0))

        self.start_btn = tk.Button(
            button_frame,
            text="Start",
            command=self.start,
            width=12,
            font=("Arial", 12),
            bg="#e9e9e9",
            fg="black",
            relief="raised",
            bd=2,
        )
        self.start_btn.pack(side="left", padx=14)

        self.stop_btn = tk.Button(
            button_frame,
            text="Stop",
            command=self.stop,
            width=12,
            font=("Arial", 12),
            bg="#e9e9e9",
            fg="black",
            relief="raised",
            bd=2,
        )
        self.stop_btn.pack(side="left", padx=14)

        self.clear_btn = tk.Button(
            button_frame,
            text="Clear",
            command=self.clear,
            width=12,
            font=("Arial", 12),
            bg="#e9e9e9",
            fg="black",
            relief="raised",
            bd=2,
        )
        self.clear_btn.pack(side="left", padx=14)

    def on_audio(self, audio):
        print("AUDIO CALLBACK HIT", type(audio), len(audio) if audio is not None else None)

        if not self.running or self.processing:
            return

        self.engine.add_audio(audio)

        energy = float(abs(audio).mean())
        print("ENERGY:", energy)

        if energy > 0.003:
            self.last_audio_time = time.time()

        silence = time.time() - self.last_audio_time
        if silence > 0.8:
            self.processing = True
            self.audio.stop()
            audio_data = self.audio.get_audio()

            threading.Thread(
                target=self.process_final,
                args=(audio_data,),
                daemon=True
            ).start()

    def on_live_text(self, text):
        self.root.after(0, lambda: self.update_live(text))

    def update_live(self, text):
        if not self.running:
            return

        self.live_text = text.strip()
        self.preview_label.config(text=self.live_text)
        self.refresh_textbox()

    def process_final(self, audio):
        final_text = self.finalizer.transcribe(audio)
        self.root.after(0, lambda: self.commit_final(final_text))

    def commit_final(self, text):
        text = text.strip()
        self.processing = False

        if not text:
            self.live_text = ""
            self.engine.reset()
            if self.running:
                self.audio.start()
            self.refresh_textbox()
            return

        if text.lower() == self.last_final_text.lower():
            self.live_text = ""
            self.engine.reset()
            if self.running:
                self.audio.start()
            self.refresh_textbox()
            return

        if self.committed_text:
            self.committed_text = f"{self.committed_text} {text}".strip()
        else:
            self.committed_text = text

        self.last_final_text = text
        self.live_text = ""

        self.engine.reset()
        if self.running:
            self.audio.start()

        self.refresh_textbox()
        self.preview_label.config(text="")

    def refresh_textbox(self):
        combined = self.committed_text
        if self.live_text:
            combined = f"{combined} {self.live_text}".strip()

        self.textbox.delete("1.0", tk.END)
        self.textbox.insert(tk.END, combined)
        self.textbox.see(tk.END)

    def start(self):
        if self.running:
            return

        self.running = True
        self.processing = False
        self.live_text = ""
        self.last_audio_time = time.time()

        self.engine.reset()
        self.audio.start()
        self.status_label.config(text="Status: Listening")
        self.preview_label.config(text="")
        self.refresh_textbox()

    def stop(self):
        if not self.running:
            return

        self.running = False
        self.audio.stop()

        audio = self.audio.get_audio()
        final_text = self.finalizer.transcribe(audio)
        self.commit_final(final_text)

        self.status_label.config(text="Status: Idle")
        self.preview_label.config(text="")

    def clear(self):
        self.committed_text = ""
        self.live_text = ""
        self.last_final_text = ""
        self.preview_label.config(text="")
        self.refresh_textbox()
        self.engine.reset()