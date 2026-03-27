import logging
import threading
import time
import queue
from tkinter import filedialog
import tkinter as tk
from tkinter import messagebox
import customtkinter as ctk
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from app.audio import AudioCapture
from app.engine import SpeechEngine
from app.whisper_finalizer import WhisperFinalizer
from app.vosk_finalizer import VoskFinalizer
from app.ai4bharat_finalizer import MarathiAI4BharatFinalizer

logger = logging.getLogger(__name__)

# Energy threshold above which a frame is considered "voice".
_ENERGY_THRESHOLD = 0.002
# Seconds of silence before auto-finalising the current segment.
_SILENCE_TIMEOUT = 0.8
_SILENCE_TIMEOUT_INDIC = 1.1


class VoiceTypingGUI:
    """Top-level application window."""

    def __init__(self, root):
        self.root = root
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.root.title("Offline Voice Typing Pro")
        self.root.geometry("1050x650")
        self.root.minsize(800, 500)
        self.root.configure(bg="#1E1E1E")
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
            self._vosk_hi = VoskFinalizer()  # Vosk used exclusively for Hindi
            self._ai4b_mr = MarathiAI4BharatFinalizer() # AI4Bharat used exclusively for Marathi
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
        pad = {"padx": 24, "pady": 18}
        self._main = ctk.CTkFrame(self.root, fg_color="transparent")
        self._main.pack(fill="both", expand=True, **pad)

        header_frame = ctk.CTkFrame(self._main, fg_color="transparent")
        header_frame.pack(fill="x", pady=(0, 6))

        # Title
        ctk.CTkLabel(
            header_frame,
            text="OFFLINE VOICE TYPING PRO",
            font=ctk.CTkFont(family="Inter", size=26, weight="bold"),
            text_color="#FFFFFF",
        ).pack(side="left", anchor="w")

        # Status row
        self._status_var = tk.StringVar(value="Status: Idle")
        ctk.CTkLabel(
            header_frame,
            textvariable=self._status_var,
            font=ctk.CTkFont(family="Inter", size=15, weight="bold"),
            text_color="#3B8ED0",
        ).pack(side="right", anchor="e")

        # Live-preview label
        self._preview_var = tk.StringVar(value="")
        ctk.CTkLabel(
            self._main,
            textvariable=self._preview_var,
            font=ctk.CTkFont(family="Inter", size=14, slant="italic"),
            text_color="#A0A0A0",
            wraplength=1000,
            justify="left",
            anchor="w"
        ).pack(pady=(0, 12), fill="x", padx=2)

        # Main text area + scrollbar
        text_frame = ctk.CTkFrame(self._main, corner_radius=8, border_width=1, border_color="#3A3A3A")
        text_frame.pack(fill="both", expand=True, pady=(0, 16))

        # Internal styling for tk.Text to match CustomTkinter
        self._textbox = tk.Text(
            text_frame,
            font=("Inter", 16),
            bg="#141414",
            fg="white",
            insertbackground="white",
            wrap="word",
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
            undo=True,
            padx=14,
            pady=14
        )
        self._textbox.pack(side="left", fill="both", expand=True, padx=2, pady=2)
        
        # Custom Scrollbar mimicking CTk Scrollbar
        scrollbar = ctk.CTkScrollbar(text_frame, command=self._textbox.yview)
        scrollbar.pack(side="right", fill="y", padx=2, pady=2)
        self._textbox.configure(yscrollcommand=scrollbar.set)

        # Configure a special grey-italic tag for real-time live preview words
        self._textbox.tag_configure("preview", foreground="#888888", font=("Inter", 16, "italic"))

        # Word-count row
        self._wc_var = tk.StringVar(value="Words: 0")
        ctk.CTkLabel(
            self._main,
            textvariable=self._wc_var,
            font=ctk.CTkFont(family="Inter", size=13),
            text_color="#888888",
            anchor="w",
        ).pack(fill="x", padx=4, pady=(0, 6))

        # Button row
        btn_frame = ctk.CTkFrame(self._main, fg_color="transparent")
        btn_frame.pack(pady=(0, 4), fill="x")

        # Language dropdown
        self._language_var = ctk.StringVar(value="en")
        
        ctk.CTkLabel(
            btn_frame, text="Lang:", font=ctk.CTkFont(family="Inter", size=15, weight="bold"), text_color="white"
        ).pack(side="left", padx=(0, 8))
        
        lang_dropdown = ctk.CTkOptionMenu(
            btn_frame, 
            variable=self._language_var, 
            values=["en", "mr", "hi"], 
            command=self._on_language_change,
            width=70,
            font=ctk.CTkFont(family="Inter", size=14, weight="bold"),
            dropdown_font=ctk.CTkFont(family="Inter", size=14)
        )
        lang_dropdown.pack(side="left", padx=(0, 24))

        # Modern button stylings
        btn_cfg = dict(width=100, height=38, font=ctk.CTkFont(family="Inter", size=14, weight="bold"), corner_radius=6)

        self._start_btn = ctk.CTkButton(
            btn_frame, text="▶ Start", command=self.start,
            fg_color="#2FA572", hover_color="#25835A", **btn_cfg
        )
        self._start_btn.pack(side="left", padx=6)

        self._stop_btn = ctk.CTkButton(
            btn_frame, text="■ Stop", command=self.stop,
            fg_color="#C94A4A", hover_color="#A33B3B", **btn_cfg
        )
        self._stop_btn.pack(side="left", padx=6)

        self._copy_btn = ctk.CTkButton(
            btn_frame, text="⎘ Copy", command=self._copy_to_clipboard,
            fg_color="#4A4A4A", hover_color="#3A3A3A", **btn_cfg
        )
        self._copy_btn.pack(side="left", padx=6)

        self._clear_btn = ctk.CTkButton(
            btn_frame, text="✕ Clear", command=self.clear,
            fg_color="#4A4A4A", hover_color="#3A3A3A", **btn_cfg
        )
        self._clear_btn.pack(side="left", padx=6)

        # Right-aligned buttons via spacer
        spacer = ctk.CTkFrame(btn_frame, fg_color="transparent")
        spacer.pack(side="left", fill="x", expand=True)

        self._save_btn = ctk.CTkButton(
            btn_frame, text="💾 Save TXT", command=self._save_txt,
            fg_color="#3B8ED0", hover_color="#2A6B9C", **btn_cfg
        )
        self._save_btn.pack(side="right", padx=6)

        self._export_btn = ctk.CTkButton(
            btn_frame, text="📄 Export PDF", command=self._export_pdf,
            fg_color="#D08E3B", hover_color="#9C6B2A", **btn_cfg
        )
        self._export_btn.pack(side="right", padx=6)

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
        silence_timeout = self._current_silence_timeout()

        if self._speech_detected and silence > silence_timeout:
            self._speech_detected = False
            self._segment_start_time = None
            
            # PortAudio heavily deadlocks if `stream.stop()` is called from inside its own callback.
            # Marshal the finalise order securely onto the Tkinter Main Thread to safely abort the mic.
            self.root.after(0, self._trigger_finalisation)
        elif not self._speech_detected and silence > silence_timeout:
            self._last_audio_time = time.monotonic()

    def _current_silence_timeout(self) -> float:
        lang = self._language_var.get()
        return _SILENCE_TIMEOUT_INDIC if lang in ["hi", "mr"] else _SILENCE_TIMEOUT

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
            if new_lang in ["mr", "hi"]:
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
    # Whisper / Vosk finalisation  (background thread → Tk thread)
    # ==================================================================

    def _process_final(self, audio) -> None:
        lang = self._language_var.get()
        
        # Hindi should use Vosk exclusively and nothing else
        if lang == "hi":
            try:
                final_text = self._vosk_hi.transcribe(audio, language=lang)
            except Exception as e:
                logger.error(f"Vosk error for Hindi: {e}")
                final_text = ""
        # Marathi exclusively runs via AI4Bharat model
        elif lang == "mr":
            try:
                final_text = self._ai4b_mr.transcribe(audio, language=lang)
            except Exception as e:
                logger.error(f"AI4Bharat error for Marathi: {e}")
                final_text = ""
        else:
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

        merged_text = self._merge_with_overlap(self._committed_text, text)
        if merged_text.strip().lower() == self._committed_text.strip().lower():
            self._refresh_textbox()
            _restart_if_running()
            return
        self._committed_text = merged_text

        self._last_final_text = text
        
        # Re-render committed text after overlap-aware merge to avoid duplicate phrase tails.
        self._delete_preview_tags()
        self._textbox.delete("1.0", tk.END)
        self._textbox.insert("1.0", self._committed_text)
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

    def _merge_with_overlap(self, committed: str, incoming: str, max_overlap_words: int = 8, max_trim_words: int = 2) -> str:
        committed = committed.strip()
        incoming = incoming.strip()
        if not committed:
            return incoming
        if not incoming:
            return committed

        cw = committed.split()
        nw = incoming.split()

        best = (0, 0)  # (overlap_words, trim_words)
        max_k = min(max_overlap_words, len(cw), len(nw))

        for trim in range(0, min(max_trim_words, len(cw)) + 1):
            base = cw[: len(cw) - trim] if trim else cw
            if not base:
                continue
            local_max = min(max_k, len(base), len(nw))
            for k in range(local_max, 0, -1):
                if [w.casefold() for w in base[-k:]] == [w.casefold() for w in nw[:k]]:
                    if (k, -trim) > (best[0], -best[1]):
                        best = (k, trim)
                    break

        overlap, trim = best
        base = cw[: len(cw) - trim] if trim else cw
        tail = nw[overlap:]
        return " ".join(base + tail).strip()

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
        lang = self._language_var.get()
        self._preview_var.set(f"(Live preview disabled for {lang.upper()})" if lang in ["mr", "hi"] else "")
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
        
        if lang == "hi":
            try:
                final_text = self._vosk_hi.transcribe(audio, language=lang)
            except Exception as e:
                logger.error(f"Vosk error for Hindi: {e}")
                final_text = ""
        elif lang == "mr":
            try:
                final_text = self._ai4b_mr.transcribe(audio, language=lang)
            except Exception as e:
                logger.error(f"AI4Bharat error for Marathi: {e}")
                final_text = ""
        else:
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
            self._copy_btn.configure(text="✓ Copied!")
            self.root.after(1500, lambda: self._copy_btn.configure(text="⎘ Copy"))

    # ==================================================================
    # Button-state management
    # ==================================================================

    def _sync_button_states(self) -> None:
        """Enable/disable buttons to reflect the current application state."""
        busy = self._processing
        if self._running:
            self._start_btn.configure(state="disabled")
            self._stop_btn.configure(state="normal" if not busy else "disabled")
        else:
            self._start_btn.configure(state="normal" if not busy else "disabled")
            self._stop_btn.configure(state="disabled")

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
