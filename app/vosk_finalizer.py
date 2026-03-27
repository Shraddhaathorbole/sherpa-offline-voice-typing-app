import logging
import os
import json
import threading
import numpy as np

logger = logging.getLogger(__name__)

class VoskFinalizer:
    """
    Hindi final-segment transcription using Vosk.
    The user explicitly requested Vosk exclusively for Hindi.
    """

    def __init__(self, model_path="models/vosk_hi"):
        self.model_path = model_path
        self._model = None
        self._load_lock = threading.Lock()

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            try:
                import vosk
            except ImportError:
                logger.error("Vosk library not installed. Please run `pip install vosk`")
                raise RuntimeError("vosk is required for Hindi transcription.")

            vosk.SetLogLevel(-1)

            if not os.path.exists(self.model_path):
                logger.error(f"Vosk model not found at '{self.model_path}'.")
                raise RuntimeError(
                    f"Please download the Vosk Hindi model (e.g., vosk-model-hi-0.22) "
                    f"and extract it into the directory: {self.model_path}"
                )

            logger.info("Loading Vosk model from '%s'…", self.model_path)
            try:
                self._model = vosk.Model(self.model_path)
                logger.info("Vosk model loaded successfully.")
            except Exception:
                logger.exception("Failed to load Vosk model.")
                raise

    def transcribe(self, audio: "np.ndarray | None", language: str = "hi") -> str:
        if language != "hi":
            logger.warning(f"VoskFinalizer called with language='{language}' but it's meant for Hindi.")
            
        if audio is None or len(audio) == 0:
            return ""

        self._ensure_model()
        import vosk

        # Vosk expects 16-bit PCM mono
        audio_int16 = (audio * 32767.0).astype(np.int16)

        rec = vosk.KaldiRecognizer(self._model, 16000)
        
        # We process the whole chunk at once
        rec.AcceptWaveform(audio_int16.tobytes())
        res = json.loads(rec.FinalResult())
        
        text = res.get("text", "").strip()

        # Danda conversion for Hindi if necessary
        if text:
            # Vosk usually doesn't add punctuation to Hindi. If it did, normalize.
            text = text.replace(".", "।")
            if text[-1] not in ("।", "॥", "?", "!"):
                text = text + " ।"
        
        return text
