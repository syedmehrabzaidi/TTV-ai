from enum import Enum
from pathlib import Path
from pydantic import BaseModel
import os


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
STORYBOARD_DIR = DATA_DIR / "storyboards"
AUDIO_DIR = DATA_DIR / "audio"
VIDEO_DIR = DATA_DIR / "videos"
TMP_DIR = DATA_DIR / "tmp"


class DevicePreset(str, Enum):
    CPU = "cpu"
    GPU = "gpu"


class LLMConfig(BaseModel):
    model_name: str
    device: str
    max_new_tokens: int = 256


class StoryboardConfig(BaseModel):
    model_name: str
    device: str
    image_size: int = 512
    num_inference_steps: int = 20


class TTSConfig(BaseModel):
    # For a small ONNX TTS, this would be a directory with model.onnx + config
    onnx_model_path: Path | None = None
    device: str = "cpu"
    sample_rate: int = 22050
    # Fallback TTS if ONNX is not configured (e.g. simple CPU-only TTS)
    use_fallback_tts: bool = True


class PipelineConfig(BaseModel):
    preset: DevicePreset = DevicePreset.CPU
    llm: LLMConfig
    storyboard: StoryboardConfig
    tts: TTSConfig


def build_config(preset: DevicePreset = DevicePreset.CPU) -> PipelineConfig:
    """
    Build a PipelineConfig for CPU or GPU.

    CPU-first defaults:
      - Small LLM (GPT-2) for script + scene breakdown.
      - Tiny SD-like model for storyboard.
      - ONNXRuntime on CPU (if model configured) or simple fallback TTS.

    GPU preset:
      - Larger instruction-tuned LLM.
      - Heavier SD model (e.g. SDXL).
      - ONNXRuntime on CUDA for TTS.
    """
    if preset == DevicePreset.GPU:
        llm_cfg = LLMConfig(
            # Example heavy GPU instruction model (requires enough vRAM)
            model_name=os.environ.get(
                "LLM_MODEL_GPU", "meta-llama/Llama-3.1-8B-Instruct"
            ),
            device="cuda",
            max_new_tokens=512,
        )
        storyboard_cfg = StoryboardConfig(
            model_name=os.environ.get(
                "STORYBOARD_MODEL_GPU",
                "stabilityai/stable-diffusion-xl-base-1.0",
            ),
            device="cuda",
            image_size=768,
            num_inference_steps=30,
        )
        tts_cfg = TTSConfig(
            onnx_model_path=Path(
                os.environ.get("TTS_ONNX_MODEL_GPU", "")
            )
            if os.environ.get("TTS_ONNX_MODEL_GPU")
            else None,
            device="cuda",
            sample_rate=22050,
            use_fallback_tts=True,
        )
    else:
        # CPU-friendly defaults
        llm_cfg = LLMConfig(
            model_name=os.environ.get("LLM_MODEL_CPU", "gpt2"),
            device="cpu",
            max_new_tokens=256,
        )
        storyboard_cfg = StoryboardConfig(
            # A small Stable Diffusion-like model; still heavy but manageable on CPU
            model_name=os.environ.get(
                "STORYBOARD_MODEL_CPU", "Segmind/tiny-sd"
            ),
            device="cpu",
            image_size=512,
            num_inference_steps=15,
        )
        tts_cfg = TTSConfig(
            onnx_model_path=Path(
                os.environ.get("TTS_ONNX_MODEL_CPU", "")
            )
            if os.environ.get("TTS_ONNX_MODEL_CPU")
            else None,
            device="cpu",
            sample_rate=22050,
            use_fallback_tts=True,
        )

    return PipelineConfig(
        preset=preset,
        llm=llm_cfg,
        storyboard=storyboard_cfg,
        tts=tts_cfg,
    )


def ensure_directories() -> None:
    """Create data directories if they don't exist."""
    for d in [DATA_DIR, STORYBOARD_DIR, AUDIO_DIR, VIDEO_DIR, TMP_DIR]:
        d.mkdir(parents=True, exist_ok=True)


