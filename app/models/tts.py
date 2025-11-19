from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import onnxruntime as ort
import soundfile as sf

from ..config import TTSConfig


@dataclass
class TTSResult:
    audio_path: Path
    subtitles: List[Tuple[float, float, str]]


class OnnxTTSEngine:
    """
    Simple ONNXRuntime-based TTS wrapper.

    This expects a pre-exported TTS model that takes text input and outputs
    a waveform. The exact inputs/outputs depend on the model you choose.
    For production, you would adapt this class to your specific model.
    """

    def __init__(self, cfg: TTSConfig) -> None:
        self.cfg = cfg
        self._sess: ort.InferenceSession | None = None

    def _load(self):
        if self._sess is None and self.cfg.onnx_model_path is not None:
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if self.cfg.device != "cpu"
                else ["CPUExecutionProvider"]
            )
            self._sess = ort.InferenceSession(
                str(self.cfg.onnx_model_path), providers=providers
            )

    def synthesize(
        self,
        text: str,
        output_path: Path,
    ) -> Path:
        """
        Synthesise speech and save it to `output_path`.

        NOTE: This is a placeholder, since ONNX models differ widely.
        In a real setup, you'd implement tokenisation and calls to the model
        based on its documented interface.
        """
        self._load()
        if self._sess is None:
            raise RuntimeError("ONNX TTS model is not configured.")

        # Placeholder: generate 1 second of silence.
        # Replace this with real model inference.
        duration_sec = 1.0
        samples = np.zeros(int(duration_sec * self.cfg.sample_rate), dtype=np.float32)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), samples, self.cfg.sample_rate)
        return output_path


class FallbackTTSEngine:
    """
    Extremely simple CPU-only TTS fallback using a dummy waveform.

    This keeps the pipeline runnable even without a real TTS model.
    Replace this with something like `pyttsx3` or a small offline TTS library.
    """

    def __init__(self, cfg: TTSConfig) -> None:
        self.cfg = cfg

    def synthesize(self, text: str, output_path: Path) -> Path:
        # For demo: generate a short sine wave as "audio".
        duration_sec = max(3.0, len(text.split()) * 0.35)
        t = np.linspace(0, duration_sec, int(duration_sec * self.cfg.sample_rate))
        freq = 220.0
        samples = 0.1 * np.sin(2 * np.pi * freq * t)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), samples, self.cfg.sample_rate)
        return output_path


class TTSPipeline:
    """
    TTS pipeline that uses ONNXRuntime if configured, otherwise falls back.
    Also generates simple subtitles aligned roughly with words.
    """

    def __init__(self, cfg: TTSConfig) -> None:
        self.cfg = cfg
        self.onnx_engine = (
            OnnxTTSEngine(cfg) if cfg.onnx_model_path is not None else None
        )
        self.fallback_engine = (
            FallbackTTSEngine(cfg) if cfg.use_fallback_tts else None
        )

    def synthesize_with_subtitles(
        self, script: str, output_path: Path
    ) -> TTSResult:
        # 1) Generate audio
        if self.onnx_engine is not None:
            audio_path = self.onnx_engine.synthesize(script, output_path)
        elif self.fallback_engine is not None:
            audio_path = self.fallback_engine.synthesize(script, output_path)
        else:
            raise RuntimeError(
                "No TTS backend configured. Provide an ONNX model path or enable fallback."
            )

        # 2) Create naive word-aligned subtitles based on speech rate
        words = script.split()
        words_per_second = 2.3  # heuristic
        subtitles: List[Tuple[float, float, str]] = []

        current_time = 0.0
        window_words: List[str] = []
        for w in words:
            window_words.append(w)
            # Emit a subtitle every ~8 words
            if len(window_words) >= 8:
                start = current_time
                duration = len(window_words) / words_per_second
                end = start + duration
                subtitles.append((start, end, " ".join(window_words)))
                current_time = end
                window_words = []

        if window_words:
            start = current_time
            duration = len(window_words) / words_per_second
            end = start + duration
            subtitles.append((start, end, " ".join(window_words)))

        return TTSResult(audio_path=audio_path, subtitles=subtitles)


