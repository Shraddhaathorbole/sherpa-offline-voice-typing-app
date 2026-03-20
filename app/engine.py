import os
import sherpa_onnx
import numpy as np

from app.text_post import TextPostProcessor


class SpeechEngine:
    def __init__(self, callback):
        self.callback = callback
        self.post = TextPostProcessor()

        model_dir = os.path.join("models", "sherpa")

        required = ["encoder.onnx", "decoder.onnx", "joiner.onnx", "tokens.txt"]
        for f in required:
            if not os.path.exists(os.path.join(model_dir, f)):
                raise RuntimeError(f"Missing model file: {f}")

        self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            encoder=os.path.join(model_dir, "encoder.onnx"),
            decoder=os.path.join(model_dir, "decoder.onnx"),
            joiner=os.path.join(model_dir, "joiner.onnx"),
            tokens=os.path.join(model_dir, "tokens.txt"),
            num_threads=2,
            sample_rate=16000,
            feature_dim=80,
        )

        self.reset()

    def reset(self):
        self.stream = self.recognizer.create_stream()
        self.last_raw_text = ""
        self.last_emitted_text = ""
        self.silence_counter = 0

    def add_audio(self, audio: np.ndarray):
        if audio is None or len(audio) == 0:
            return

        # simple VAD-like gate using energy
        energy = float(np.mean(np.abs(audio)))
        if energy < 0.003:
            self.silence_counter += 1
        else:
            self.silence_counter = 0

        # skip long silence blocks
        if self.silence_counter > 25:
            return

        self.stream.accept_waveform(16000, audio)

        while self.recognizer.is_ready(self.stream):
            self.recognizer.decode_stream(self.stream)

        result = self.recognizer.get_result(self.stream).strip()
        if not result:
            return

        if result == self.last_raw_text:
            return

        if self.last_raw_text and result.startswith(self.last_raw_text):
            new_text = result[len(self.last_raw_text):].strip()
        else:
            new_text = result

        self.last_raw_text = result

        new_text = self._clean_increment(new_text)
        if not new_text:
            return

        processed = self.post.process(new_text)
        if not processed:
            return

        if processed == self.last_emitted_text:
            return

        self.last_emitted_text = processed
        self.callback(processed)

    def _clean_increment(self, text: str) -> str:
        if not text:
            return ""

        text = text.strip()

        # remove repeated adjacent words
        words = text.split()
        cleaned = []
        for w in words:
            if not cleaned or cleaned[-1].lower() != w.lower():
                cleaned.append(w)

        text = " ".join(cleaned).strip()

        # reject tiny garbage
        if len(text) < 2:
            return ""

        return text