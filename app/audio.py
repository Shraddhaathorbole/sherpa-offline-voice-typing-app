import sounddevice as sd
import numpy as np
import logging
import threading

logger = logging.getLogger(__name__)

class AudioCapture:

    def __init__(self, callback, sample_rate: int = 16000, block_size: int = 1600):
        self.callback = callback
        self.sample_rate = sample_rate
        self.block_size = block_size

        self._stream = None          
        self._lock = threading.Lock()
        self._recorded: list = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open and start the audio input stream."""
        with self._lock:
            if self._stream is not None:
                logger.warning("AudioCapture.start() called while stream already active – ignoring.")
                return

            self._recorded = []

            try:
                self._stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype="float32",
                    blocksize=self.block_size,
                    callback=self._sd_callback,
                )
                self._stream.start()
                logger.info("Audio stream started (rate=%d, blocksize=%d).", self.sample_rate, self.block_size)
            except Exception:
                self._stream = None
                logger.exception("Failed to open audio input stream.")
                raise

    def stop(self) -> None:
        """Stop and close the audio input stream."""
        with self._lock:
            stream = self._stream
            self._stream = None

        if stream is None:
            return

        try:
            stream.stop()
            stream.close()
            logger.info("Audio stream stopped.")
        except Exception:
            logger.exception("Error while stopping audio stream.")

    def get_audio(self):
        """
        Return all audio recorded since the last start() as a single float32
        array, or None if nothing was captured yet.
        """
        with self._lock:
            chunks = list(self._recorded)

        if not chunks:
            return None

        return np.concatenate(chunks, axis=0).astype(np.float32)

    @property
    def is_active(self) -> bool:
        """True while the audio stream is open."""
        with self._lock:
            return self._stream is not None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _sd_callback(self, indata, frames, time, status) -> None:
        if status:
            logger.warning("Audio stream status: %s", status)

        audio = np.squeeze(indata).copy().astype(np.float32)

        with self._lock:
            self._recorded.append(audio)

        try:
            self.callback(audio)
        except Exception:
            logger.exception("Unhandled exception inside audio callback consumer.")
