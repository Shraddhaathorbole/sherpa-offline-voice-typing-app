import os
import logging

import numpy as np
import sherpa_onnx

logger = logging.getLogger(__name__)


class SpeechEngine:
    """
    Wraps a Sherpa-ONNX online transducer recogniser for low-latency,
    streaming speech recognition.

    on_live_text(str) is called whenever the partial hypothesis changes.
    """

    _MODEL_DIR = os.path.join("models", "sherpa")

    def __init__(self, on_live_text):
        self.on_live_text = on_live_text

        self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            encoder=os.path.join(self._MODEL_DIR, "encoder.onnx"),
            decoder=os.path.join(self._MODEL_DIR, "decoder.onnx"),
            joiner=os.path.join(self._MODEL_DIR, "joiner.onnx"),
            tokens=os.path.join(self._MODEL_DIR, "tokens.txt"),
            num_threads=2,
            sample_rate=16000,
            feature_dim=80,
        )
        logger.info("Sherpa-ONNX recogniser loaded from '%s'.", self._MODEL_DIR)

        self._stream = None
        self._last_text = ""
        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Discard current partial hypothesis and start a fresh stream."""
        self._stream = self.recognizer.create_stream()
        self._last_text = ""

    def add_audio(self, audio: np.ndarray) -> None:
        """
        Feed a block of float32 PCM audio (16 kHz, mono) to the recogniser.
        Calls on_live_text only when the hypothesis actually changes.
        """
        if audio is None or len(audio) == 0:
            return

        audio = np.asarray(audio, dtype=np.float32)

        try:
            self._stream.accept_waveform(16000, audio)

            while self.recognizer.is_ready(self._stream):
                self.recognizer.decode_stream(self._stream)

            result = self.recognizer.get_result(self._stream).strip()
        except Exception:
            logger.exception("Error during streaming decode – resetting stream.")
            self.reset()
            return

        if result and result != self._last_text:
            self._last_text = result
            try:
                self.on_live_text(result)
            except Exception:
                logger.exception("on_live_text callback raised an exception.")
