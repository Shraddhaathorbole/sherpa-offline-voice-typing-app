import os
import sherpa_onnx
import numpy as np


class SpeechEngine:
    def __init__(self, on_live_text):
        self.on_live_text = on_live_text

        model_dir = os.path.join("models", "sherpa")

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
        self.last_text = ""

    def add_audio(self, audio):
        if audio is None:
            return

        audio = audio.astype(np.float32)

        self.stream.accept_waveform(16000, audio)

        while self.recognizer.is_ready(self.stream):
            self.recognizer.decode_stream(self.stream)

        result = self.recognizer.get_result(self.stream).strip()

        if not result:
            return

        if result != self.last_text:
            self.last_text = result
            self.on_live_text(result)