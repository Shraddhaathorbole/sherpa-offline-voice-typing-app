import logging
import os
import re
import unicodedata
import numpy as np
import sherpa_onnx

logger = logging.getLogger(__name__)
class SherpaFinalizer:
    """
    Hindi final-segment transcription using Sherpa + Sherpa VAD.
    Prefers Sherpa Whisper models (language locked to hi), with Sherpa CTC fallback.
    """

    _MODEL_ROOT = os.path.join("models")
    _VAD_BUFFER_SECONDS = 30.0
    _VAD_SEGMENT_PADDING_SAMPLES = 3200  # 200 ms
    _MIN_VAD_SPEECH_SAMPLES = 1600       # 100 ms

    def __init__(self) -> None:
        self._init_vad()
        self._backend = "ctc"
        self._recognizer = self._build_primary_recognizer()

    def _build_primary_recognizer(self):
        whisper_paths = self._resolve_whisper_paths(self._MODEL_ROOT)
        if whisper_paths is not None:
            encoder, decoder, tokens = whisper_paths
            logger.info("Hindi finalizer using Sherpa Whisper model from '%s'.", os.path.dirname(encoder))
            self._backend = "whisper"
            return sherpa_onnx.OfflineRecognizer.from_whisper(
                encoder=encoder,
                decoder=decoder,
                tokens=tokens,
                language="hi",
                task="transcribe",
            )

        model_path, tokens_path = self._resolve_ctc_paths(self._MODEL_ROOT)
        logger.info("Hindi finalizer using Sherpa CTC model='%s'.", model_path)
        self._backend = "ctc"
        return sherpa_onnx.OfflineRecognizer.from_omnilingual_asr_ctc(
            model=model_path,
            tokens=tokens_path,
        )

    def transcribe(self, audio: "np.ndarray | None") -> str:
        if audio is None or len(audio) == 0:
            return ""
        pcm = np.asarray(audio, dtype=np.float32)

        try:
            chunks = self._extract_speech_chunks(pcm)
            text_vad = self._decode(chunks) if chunks else ""
            text_raw = self._decode([pcm])

            cands = []
            for t in (text_vad, text_raw):
                if not t:
                    continue
                p = self._post_process_hi(t)
                if p:
                    cands.append(p)
            if not cands:
                return ""
            return sorted(cands, key=lambda x: (-self._score_hi(x), -len(x), x))[0]
        except Exception:
            logger.exception("Sherpa Hindi transcription failed.")
            return ""

    def _decode(self, chunks: list[np.ndarray]) -> str:
        parts = []
        for chunk in chunks:
            if chunk is None or len(chunk) == 0:
                continue
            stream = self._recognizer.create_stream()
            stream.accept_waveform(16000, chunk)
            self._recognizer.decode_stream(stream)
            t = stream.result.text.strip()
            if t:
                parts.append(t)
        return " ".join(parts).strip()

    def _init_vad(self) -> None:
        self._vad_cfg = None
        vad_model = self._resolve_vad_model_path(self._MODEL_ROOT)
        if not vad_model:
            logger.warning("Hindi VAD model not found under '%s'. Running without VAD.", self._MODEL_ROOT)
            return
        cfg = sherpa_onnx.VadModelConfig()
        cfg.sample_rate = 16000
        cfg.silero_vad.model = vad_model
        cfg.silero_vad.threshold = 0.40
        cfg.silero_vad.min_speech_duration = 0.10
        cfg.silero_vad.min_silence_duration = 0.20
        cfg.silero_vad.max_speech_duration = 30.0
        self._vad_cfg = cfg
        logger.info("Hindi VAD enabled with model '%s'.", vad_model)

    def _extract_speech_chunks(self, pcm: np.ndarray) -> list[np.ndarray]:
        if self._vad_cfg is None:
            return [pcm]
        vad = sherpa_onnx.VoiceActivityDetector(self._vad_cfg, self._VAD_BUFFER_SECONDS)
        step = int(self._vad_cfg.silero_vad.window_size) or 512
        for i in range(0, len(pcm), step):
            vad.accept_waveform(pcm[i:i + step])
        vad.flush()

        chunks = []
        while not vad.empty():
            seg = vad.front
            start = max(0, int(seg.start) - self._VAD_SEGMENT_PADDING_SAMPLES)
            end = min(len(pcm), int(seg.start) + len(seg.samples) + self._VAD_SEGMENT_PADDING_SAMPLES)
            if end > start:
                part = np.asarray(pcm[start:end], dtype=np.float32)
                if len(part) >= self._MIN_VAD_SPEECH_SAMPLES:
                    chunks.append(part)
            vad.pop()
        return chunks or [pcm]

    @staticmethod
    def _resolve_vad_model_path(model_root: str) -> str | None:
        if not os.path.isdir(model_root):
            return None
        cands = []
        for root, _, files in os.walk(model_root):
            for f in files:
                lf = f.lower()
                if lf.endswith(".onnx") and "vad" in lf and "rknn" not in lf:
                    p = os.path.join(root, f)
                    cands.append((os.path.getsize(p), p))
        if not cands:
            return None
        return sorted(cands, key=lambda x: (-x[0], x[1]))[0][1]

    @staticmethod
    def _resolve_whisper_paths(model_root: str) -> tuple[str, str, str] | None:
        if not os.path.isdir(model_root):
            return None
        candidates = []
        for root, _, files in os.walk(model_root):
            enc = [os.path.join(root, f) for f in files if f.endswith(".onnx") and "encoder" in f.lower()]
            dec = [os.path.join(root, f) for f in files if f.endswith(".onnx") and "decoder" in f.lower()]
            tok = [os.path.join(root, f) for f in files if f.lower().endswith("tokens.txt")]
            if not enc or not dec or not tok:
                continue
            for e in enc:
                for d in dec:
                    score = os.path.getsize(e) + os.path.getsize(d)
                    if ".int8." in e.lower():
                        score += 10
                    if ".int8." in d.lower():
                        score += 10
                    candidates.append((score, e, d, sorted(tok)[0]))
        if not candidates:
            return None
        _, e, d, t = sorted(candidates, key=lambda x: (-x[0], x[1], x[2], x[3]))[0]
        return e, d, t

    @staticmethod
    def _resolve_ctc_paths(model_root: str) -> tuple[str, str]:
        if not os.path.isdir(model_root):
            raise RuntimeError("No models directory found.")
        cands = []
        for root, _, files in os.walk(model_root):
            toks = [os.path.join(root, f) for f in files if f.lower().endswith("tokens.txt")]
            models = [
                os.path.join(root, f)
                for f in files
                if f.endswith(".onnx") and "model" in f.lower() and "vad" not in f.lower()
            ]
            if not toks or not models:
                continue
            tok = sorted(toks)[0]
            for m in models:
                cands.append((os.path.getsize(m), m, tok))
        if not cands:
            raise RuntimeError("No Sherpa model*.onnx + tokens.txt found under models/.")
        _, m, t = sorted(cands, key=lambda x: (-x[0], x[1], x[2]))[0]
        return m, t

    @staticmethod
    def _post_process_hi(text: str) -> str:
        if not text:
            return ""
        text = unicodedata.normalize("NFC", text)
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"(^|\s)[\u093A-\u094D\u0951-\u0957]+", r"\1", text)
        text = re.sub(r"([\u0915-\u0939])\u094d([\u093E-\u094C])", r"\1\2", text)
        text = re.sub(r"([^\s]+)(?:\s+\1\b)+", r"\1", text)
        return text.strip()

    @staticmethod
    def _score_hi(text: str) -> int:
        dev = len(re.findall(r"[\u0900-\u097F]", text))
        latin = len(re.findall(r"[A-Za-z]", text))
        words = len(text.split())
        return (dev * 2) + words - (latin * 2)
