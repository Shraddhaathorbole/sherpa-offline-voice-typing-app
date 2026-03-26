import logging
import threading
import time
import queue
from tkinter import filedialog
import tkinter as tk
from tkinter import messagebox
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from app.audio import AudioCapture
from app.engine import SpeechEngine
from app.whisper_finalizer import WhisperFinalizer

logger = logging.getLogger(__name__)

# Energy threshold above which a frame is considered "voice".
_ENERGY_THRESHOLD = 0.003
# Seconds of silence before auto-finalising the current segment.
_SILENCE_TIMEOUT = 0.8


class VoiceTypingGUI:
    """Top-level application window."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Offline Voice Typing")
        self.root.geometry("1000x620")
        self.root.configure(bg="#2f2f2f")
        self.root.resizable(True, True)

        # ---- Application state ----------------------------------------
        self._running = False       # True while session is active
        self._processing = False    # True while Whisper is transcribing
        self._committed_text = ""   # Finalised text shown in the textbox
        self._live_text = ""        # Current partial hypothesis (preview)
        self._last_final_text = ""  # Duplicate-suppression guard
        self._last_audio_time = time.monotonic()
        self._segment_start_time = None
        self._speech_detected = False

        # Audio Queue and Consumer Thread for low-latency non-blocking Engine feed
        self._audio_queue = queue.Queue()
        self._consumer_thread = None
        self._consumer_running = False

        # ---- Initialise back-end components ---------------------------
        try:
            self._engine = SpeechEngine(self._on_live_text)
            self._finalizer = WhisperFinalizer()
        except Exception as exc:
            messagebox.showerror("Initialisation Error", str(exc))
            raise

        self._audio = AudioCapture(self._on_audio)

        # ---- Build UI -------------------------------------------------
        self._build_ui()
        self._sync_button_states()

        # Intercept window-close so we can stop the audio stream cleanly.
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ==================================================================
    # UI Construction
    # ==================================================================

    def _build_ui(self) -> None:
        pad = {"padx": 16, "pady": 14}
        self._main = tk.Frame(self.root, bg="#2f2f2f")
        self._main.pack(fill="both", expand=True, **pad)

        # Title
        tk.Label(
            self._main,
            text="OFFLINE VOICE TYPING",
            font=("Arial", 22, "bold"),
            fg="white",
            bg="#2f2f2f",
        ).pack(pady=(4, 4))

        # Status row
        self._status_var = tk.StringVar(value="Status: Idle")
        tk.Label(
            self._main,
            textvariable=self._status_var,
            font=("Arial", 13, "bold"),
            fg="#aad4f5",
            bg="#2f2f2f",
        ).pack()

        # Live-preview label
        self._preview_var = tk.StringVar(value="")
        tk.Label(
            self._main,
            textvariable=self._preview_var,
            font=("Arial", 11, "italic"),
            fg="#d9d9d9",
            bg="#2f2f2f",
            wraplength=920,
            justify="left",
        ).pack(pady=(6, 8), fill="x", padx=4)

        # Main text area + scrollbar
        text_frame = tk.Frame(self._main, bg="#2f2f2f")
        text_frame.pack(fill="both", expand=True, pady=(0, 12))

        scrollbar = tk.Scrollbar(text_frame)
        scrollbar.pack(side="right", fill="y")

        self._textbox = tk.Text(
            text_frame,
            font=("Arial", 15),
            bg="#0a0a0a",
            fg="white",
            insertbackground="white",
            wrap="word",
            relief="solid",
            borderwidth=1,
            highlightthickness=1,
            highlightbackground="#444",
            highlightcolor="#aad4f5",
            yscrollcommand=scrollbar.set,
            undo=True,
        )
        self._textbox.pack(side="left", fill="both", expand=True)
        # Configure a special grey-italic tag for real-time live preview words
        self._textbox.tag_configure("preview", foreground="#888", font=("Arial", 15, "italic"))
        scrollbar.config(command=self._textbox.yview)

        # Word-count row
        self._wc_var = tk.StringVar(value="Words: 0")
        tk.Label(
            self._main,
            textvariable=self._wc_var,
            font=("Arial", 10),
            fg="#888",
            bg="#2f2f2f",
            anchor="w",
        ).pack(fill="x", padx=4)

        # Button row
        btn_frame = tk.Frame(self._main, bg="#2f2f2f")
        btn_frame.pack(pady=(4, 2))

        # Language dropdown
        self._language_var = tk.StringVar(value="auto")
        lang_frame = tk.Frame(btn_frame, bg="#2f2f2f")
        lang_frame.pack(side="left", padx=(0, 10))
        
        tk.Label(lang_frame, text="Lang:", font=("Arial", 12, "bold"), fg="white", bg="#2f2f2f").pack(side="left")
        
        # We can use OptionMenu for nice native-looking dropdown
        lang_dropdown = tk.OptionMenu(lang_frame, self._language_var, "auto", "en", "mr", "hi", command=self._on_language_change)
        lang_dropdown.config(font=("Arial", 12), bg="#e9e9e9", width=4)
        lang_dropdown.pack(side="left", padx=4)

        btn_cfg = dict(width=10, font=("Arial", 12), fg="black", relief="raised", bd=2)

        self._start_btn = tk.Button(
            btn_frame, text="▶  Start", command=self.start,
            bg="#7ed67e", **btn_cfg
        )
        self._start_btn.pack(side="left", padx=10)

        self._stop_btn = tk.Button(
            btn_frame, text="■  Stop", command=self.stop,
            bg="#e07070", **btn_cfg
        )
        self._stop_btn.pack(side="left", padx=10)

        self._copy_btn = tk.Button(
            btn_frame, text="⎘  Copy", command=self._copy_to_clipboard,
            bg="#e9e9e9", **btn_cfg
        )
        self._copy_btn.pack(side="left", padx=10)

        self._clear_btn = tk.Button(
            btn_frame, text="✕  Clear", command=self.clear,
            bg="#e9e9e9", **btn_cfg
        )
        self._clear_btn.pack(side="left", padx=10)

        self._save_btn = tk.Button(
        btn_frame, text="💾 Save TXT", command=self._save_txt,
        bg="#e9e9e9", **btn_cfg)
        self._save_btn.pack(side="left", padx=10)

        self._export_btn = tk.Button(
        btn_frame, text="📄 Export PDF", command=self._export_pdf,
        bg="#e9e9e9", **btn_cfg)
        self._export_btn.pack(side="left", padx=10)

    # ==================================================================
    # Audio callback  (runs in the sounddevice thread)
    # ==================================================================

    def _on_audio(self, audio) -> None:
        if not self._running or self._processing:
            return

        # Energy-based voice-activity detection
        energy = float(abs(audio).mean())
        if energy > _ENERGY_THRESHOLD:
            self._last_audio_time = time.monotonic()
            if not self._speech_detected:
                self._speech_detected = True
                self._segment_start_time = time.monotonic()
            
            # Feed live engine for partial results via Queue (if English)
            if self._language_var.get() == "en" and self._consumer_running:
                self._audio_queue.put(audio)

        # Commit text to screen on natural breath pauses
        silence = time.monotonic() - self._last_audio_time

        if self._speech_detected and silence > _SILENCE_TIMEOUT:
            self._speech_detected = False
            self._segment_start_time = None
            
            # PortAudio heavily deadlocks if `stream.stop()` is called from inside its own callback.
            # Marshal the finalise order securely onto the Tkinter Main Thread to safely abort the mic.
            self.root.after(0, self._trigger_finalisation)
        elif not self._speech_detected and silence > _SILENCE_TIMEOUT:
            self._last_audio_time = time.monotonic()

    def _trigger_finalisation(self) -> None:
        """Stop streaming, grab buffered audio, launch Whisper in background."""
        self._processing = True
        self._consumer_running = False
        self._audio.stop()
        audio_data = self._audio.get_audio()

        threading.Thread(
            target=self._process_final,
            args=(audio_data,),
            daemon=True,
            name="whisper-finaliser",
        ).start()

    def _audio_consumer(self) -> None:
        """Pulls audio from the queue and feeds the engine without blocking."""
        while self._consumer_running:
            try:
                audio = self._audio_queue.get(timeout=0.1)
                self._engine.add_audio(audio)
            except queue.Empty:
                pass

    # ==================================================================
    # Live text  (called from audio thread → marshalled to Tk thread)
    # ==================================================================

    def _on_live_text(self, text: str) -> None:
        self.root.after(0, self._update_live, text)

    def _update_live(self, text: str) -> None:
        if not self._running:
            return
        self._live_text = text.strip()
        lang = self._language_var.get()
        if lang in ["mr", "hi"]:
            self._preview_var.set(f"(Live preview disabled for {lang.upper()})")
        else:
            self._preview_var.set(f"…{self._live_text}")
        self._refresh_textbox()

    def _on_language_change(self, new_lang: str) -> None:
        """Dynamically update the UI preview label when dropdown is changed."""
        if self._running:
            if new_lang in ["mr", "hi", "auto"]:
                self._preview_var.set(f"(Live preview disabled for {new_lang.upper()})")
            else:
                self._preview_var.set(f"…{self._live_text}" if self._live_text else "")

    def _save_txt(self):
        text = self._textbox.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Empty", "No text to save.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt")]
        )

        if file_path:
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(text)
                messagebox.showinfo("Success", "File saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", str(e))
    

    def _export_pdf(self):
        text = self._textbox.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Empty", "No text to export.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF Files", "*.pdf")]
        )

        if file_path:
            try:
                doc = SimpleDocTemplate(file_path)
                styles = getSampleStyleSheet()
                content = [Paragraph(text, styles["Normal"])]
                doc.build(content)

                messagebox.showinfo("Success", "PDF exported successfully!")
            except Exception as e:
                messagebox.showerror("Error", str(e))

    # ==================================================================
    # Whisper finalisation  (background thread → Tk thread)
    # ==================================================================

    def _process_final(self, audio) -> None:
        lang = self._language_var.get()
        final_text = self._finalizer.transcribe(audio, language=lang)
        self.root.after(0, self._commit_final, final_text)

    def _commit_final(self, text: str) -> None:
        text = text.strip()
        self._processing = False
        self._live_text = ""
        
        lang = self._language_var.get()
        if self._running and lang in ["mr", "hi"]:
            self._preview_var.set(f"(Live preview disabled for {lang.upper()})")
        else:
            self._preview_var.set("")

        def _restart_if_running():
            self._engine.reset()
            if self._running:
                self._last_audio_time = time.monotonic()
                self._speech_detected = False
                # Restart the consumer
                while not self._audio_queue.empty():
                    self._audio_queue.get_nowait()
                self._consumer_running = True
                self._consumer_thread = threading.Thread(target=self._audio_consumer, daemon=True)
                self._consumer_thread.start()
                try:
                    self._audio.start()
                except Exception:
                    logger.exception("Could not restart audio stream after finalisation.")
            self._sync_button_states()

        # Discard empty or duplicate results silently
        if not text or text.lower() == self._last_final_text.lower():
            self._refresh_textbox()
            _restart_if_running()
            return

        # Append with a space separator safely to the textbox
        prefix = " " if self._committed_text else ""
        if self._committed_text:
            self._committed_text = f"{self._committed_text} {text}"
        else:
            self._committed_text = text

        self._last_final_text = text
        
        # O(1) appending directly onto Tkinter UI to eliminate massive full-document refreshes
        self._delete_preview_tags()
        self._textbox.insert(tk.END, prefix + text)
        self._textbox.see(tk.END)
        self._update_word_count()
        
        _restart_if_running()

        # Update status back to listening after commit, or Idle if stopped
        if self._running:
            self._status_var.set("Status: Listening…")
        else:
            self._status_var.set("Status: Idle")
            self._sync_button_states()

    # ==================================================================
    # Textbox helpers
    # ==================================================================

    def _delete_preview_tags(self) -> None:
        """Silently purge any grey transient preview words before inserting final text."""
        ranges = self._textbox.tag_ranges("preview")
        if ranges:
            self._textbox.delete(ranges[0], ranges[1])
            
    def _update_word_count(self) -> None:
        word_count = len(self._committed_text.split()) if self._committed_text.strip() else 0
        self._wc_var.set(f"Words: {word_count}")

    def _refresh_textbox(self) -> None:
        """Render live partial words natively via tagged regions without destroying the parent document."""
        self._delete_preview_tags()
        
        if self._live_text:
            prefix = " " if self._committed_text else ""
            self._textbox.insert(tk.END, prefix + self._live_text, "preview")
            
        self._textbox.see(tk.END)

    # ==================================================================
    # Button commands
    # ==================================================================

    def start(self) -> None:
        if self._running or self._processing:
            return

        self._running = True
        self._processing = False
        self._live_text = ""
        self._last_audio_time = time.monotonic()
        self._segment_start_time = None
        self._speech_detected = False

        self._engine.reset()
        
        while not self._audio_queue.empty():
            self._audio_queue.get_nowait()
        self._consumer_running = True
        self._consumer_thread = threading.Thread(target=self._audio_consumer, daemon=True)
        self._consumer_thread.start()
        
        try:
            self._audio.start()
        except Exception as exc:
            self._running = False
            messagebox.showerror("Audio Error", f"Could not open microphone:\n{exc}")
            self._sync_button_states()
            return

        self._status_var.set("Status: Listening…")
        self._preview_var.set("(Live preview disabled for Marathi)" if self._language_var.get() == "mr" else "")
        self._refresh_textbox()
        self._sync_button_states()

    def stop(self) -> None:
        if not self._running:
            return

        self._running = False
        self._consumer_running = False
        
        # If Whisper is already processing a chunk from auto-silence, wait for it!
        if self._processing:
            self._status_var.set("Status: Processing final segment…")
            self._sync_button_states()
            return

        self._processing = True
        self._status_var.set("Status: Processing final segment…")
        self._sync_button_states()

        self._audio.stop()
        audio_data = self._audio.get_audio()

        if audio_data is not None and len(audio_data) > 0:
            self._processing = True
            threading.Thread(
                target=self._process_final_and_idle,
                args=(audio_data,),
                daemon=True,
                name="whisper-stop-finaliser",
            ).start()
        else:
            self._processing = False
            self._status_var.set("Status: Idle")
            self._sync_button_states()

    def _process_final_and_idle(self, audio) -> None:
        """Like _process_final but sets status to Idle when done."""
        lang = self._language_var.get()
        final_text = self._finalizer.transcribe(audio, language=lang)
        self.root.after(0, self._commit_final_idle, final_text)

    def _commit_final_idle(self, text: str) -> None:
        self._commit_final(text)
        if not self._running:
            self._status_var.set("Status: Idle")
            self._sync_button_states()

    def clear(self) -> None:
        if self._running:
            self._running = False
            self._consumer_running = False
            try:
                self._audio.stop()
            except Exception:
                pass
            self._processing = False
            self._sync_button_states()
            
        self._committed_text = ""
        self._live_text = ""
        self._last_final_text = ""
        lang = self._language_var.get()
        self._preview_var.set(f"(Live preview disabled for {lang.upper()})" if self._running and lang in ["mr", "hi"] else "")
        self._textbox.delete("1.0", tk.END)
        self._wc_var.set("Words: 0")
        self._consumer_running = False
        self._speech_detected = False
        self._segment_start_time = None
        self._engine.reset()

    def _copy_to_clipboard(self) -> None:
        text = self._textbox.get("1.0", tk.END).strip()
        if text:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.root.update()
            # Brief visual feedback
            self._copy_btn.config(text="✓  Copied!")
            self.root.after(1500, lambda: self._copy_btn.config(text="⎘  Copy"))

    # ==================================================================
    # Button-state management
    # ==================================================================

    def _sync_button_states(self) -> None:
        """Enable/disable buttons to reflect the current application state."""
        busy = self._processing
        if self._running:
            self._start_btn.config(state="disabled")
            self._stop_btn.config(state="normal" if not busy else "disabled")
        else:
            self._start_btn.config(state="normal" if not busy else "disabled")
            self._stop_btn.config(state="disabled")

    # ==================================================================
    # Window close
    # ==================================================================

    def _on_close(self) -> None:
        """Gracefully shut everything down before destroying the window."""
        self._running = False
        self._consumer_running = False
        try:
            self._audio.stop()
        except Exception:
            pass
        self.root.destroy()
