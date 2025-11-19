from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

from .config import (
    PipelineConfig,
    build_config,
    DevicePreset,
    VIDEO_DIR,
)
from .helper import build_run_paths, generate_run_id, build_video_from_storyboard
from .models.llm import LLMEngine, ScriptAndScenes
from .models.storyboard import StoryboardEngine, StoryboardResult
from .models.tts import TTSPipeline, TTSResult


@dataclass
class PipelineArtifacts:
    run_id: str
    script: str
    scenes: List[str]
    storyboard_images: List[Path]
    audio_path: Path
    subtitles: List
    video_path: Path


class TextToVideoPipeline:
    """
    End-to-end text-to-video orchestration.

    Step 1: LLM → script generation
    Step 2: LLM → scene breakdown
    Step 3: SD-mini → storyboard images
    Step 4: TTS → narration audio
    Step 5/6: MoviePy → combine images + audio → final video
    """

    def __init__(self, cfg: PipelineConfig | None = None) -> None:
        self.cfg = cfg or build_config(DevicePreset.CPU)
        self.llm = LLMEngine(self.cfg.llm)
        self.storyboard = StoryboardEngine(self.cfg.storyboard)
        self.tts = TTSPipeline(self.cfg.tts)

    def run(
        self,
        user_prompt: str,
        target_duration_sec: int = 20,
        run_id: str | None = None,
    ) -> PipelineArtifacts:
        run_id = run_id or generate_run_id()
        paths = build_run_paths(run_id)

        # Step 1 + 2: script + scenes
        script_and_scenes: ScriptAndScenes = self.llm.generate_script_and_scenes(
            user_prompt, target_duration_sec=target_duration_sec
        )

        # Step 3: storyboard images
        storyboard_result: StoryboardResult = self.storyboard.generate_storyboard(
            script_and_scenes.scenes, paths["storyboard_dir"]
        )

        # Step 4: TTS narration
        audio_output_path = paths["audio_dir"] / "narration.wav"
        tts_result: TTSResult = self.tts.synthesize_with_subtitles(
            script_and_scenes.script, audio_output_path
        )

        # Step 5/6: combine images + audio into final video
        video_output_path = (
            paths["video_dir"] / f"{run_id}_video.mp4"
        )

        video_path = build_video_from_storyboard(
            storyboard_result.scene_image_paths,
            tts_result.audio_path,
            tts_result.subtitles,
            video_output_path,
        )

        return PipelineArtifacts(
            run_id=run_id,
            script=script_and_scenes.script,
            scenes=script_and_scenes.scenes,
            storyboard_images=storyboard_result.scene_image_paths,
            audio_path=tts_result.audio_path,
            subtitles=tts_result.subtitles,
            video_path=video_path,
        )


def create_pipeline(preset: DevicePreset = DevicePreset.CPU) -> TextToVideoPipeline:
    """
    Factory to build a pipeline with a given device preset.
    """
    cfg = build_config(preset)
    return TextToVideoPipeline(cfg)


