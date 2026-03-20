import sounddevice as sd
import numpy as np
import noisereduce as nr


class AudioCapture:
    def __init__(self, callback, sample_rate=16000):
        self.callback = callback
        self.sample_rate = sample_rate
        self.stream = None
        self.noise_profile = None
        self.collect_noise_frames = 20
        self.noise_frames = []

    def _callback(self, indata, frames, time, status):
        if status:
            print("Audio status:", status)

        audio = np.squeeze(indata).astype(np.float32)

        # collect a tiny startup noise profile
        if len(self.noise_frames) < self.collect_noise_frames:
            self.noise_frames.append(audio.copy())
            if len(self.noise_frames) == self.collect_noise_frames:
                self.noise_profile = np.concatenate(self.noise_frames)
        else:
            if self.noise_profile is not None:
                try:
                    audio = nr.reduce_noise(
                        y=audio,
                        sr=self.sample_rate,
                        y_noise=self.noise_profile,
                        stationary=True,
                        prop_decrease=0.6,
                    )
                except Exception:
                    pass

        self.callback(audio)

    def start(self):
        if self.stream:
            return

        self.noise_profile = None
        self.noise_frames = []

        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=1600,
            callback=self._callback,
        )
        self.stream.start()

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None