import sounddevice as sd
import numpy as np


class AudioCapture:
    def __init__(self, callback, sample_rate=16000, block_size=1600):
        self.callback = callback
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.stream = None
        self.recorded = []

    def _callback(self, indata, frames, time, status):
        if status:
            print(status)

        audio = np.squeeze(indata).astype(np.float32)

        # store for whisper
        self.recorded.append(audio.copy())

        self.callback(audio)

    def start(self):
        self.recorded = []

        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.block_size,
            callback=self._callback,
        )
        self.stream.start()

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def get_audio(self):
        if not self.recorded:
            return None
        return np.concatenate(self.recorded)