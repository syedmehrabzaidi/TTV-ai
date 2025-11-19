from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import uuid
import json
import math

from moviepy.editor import (
    ImageClip,
    AudioFileClip,
    concatenate_videoclips,
)

from .config import STORYBOARD_DIR, AUDIO_DIR, VIDEO_DIR, TMP_DIR


def generate_run_id() -> str:
    return uuid.uuid4().hex


def build_run_paths(run_id: str) -> dict:
    """
    Return all paths for a given pipeline run.
    """
    run_storyboard_dir = STORYBOARD_DIR / run_id
    run_audio_dir = AUDIO_DIR / run_id
    run_video_dir = VIDEO_DIR / run_id
    run_tmp_dir = TMP_DIR / run_id

    for d in [run_storyboard_dir, run_audio_dir, run_video_dir, run_tmp_dir]:
        d.mkdir(parents=True, exist_ok=True)

    return {
        "storyboard_dir": run_storyboard_dir,
        "audio_dir": run_audio_dir,
        "video_dir": run_video_dir,
        "tmp_dir": run_tmp_dir,
    }


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_video_from_storyboard(
    scene_image_paths: List[Path],
    audio_path: Path,
    subtitles: List[Tuple[float, float, str]],
    output_path: Path,
    target_fps: int = 24,
) -> Path:
    """
    Combine storyboard images + audio into a final video.

    - scene_image_paths: ordered list of per-scene images.
    - audio_path: narration audio file path.
    - subtitles: list of (start_sec, end_sec, text) tuples.
    - output_path: where to save the resulting video.
    - Adds simple text overlays (subtitles) and cross-fade transitions.
    """
    audio_clip = AudioFileClip(str(audio_path))
    total_audio_duration = float(audio_clip.duration)

    # Distribute time across scenes based on narration duration.
    num_scenes = max(1, len(scene_image_paths))
    base_scene_duration = total_audio_duration / num_scenes

    scene_clips = []
    for idx, img_path in enumerate(scene_image_paths):
        clip = ImageClip(str(img_path)).set_duration(base_scene_duration)

        # Simple fade-in/out for transitions.
        clip = clip.crossfadein(0.3).crossfadeout(0.3)
        scene_clips.append(clip)

    if not scene_clips:
        raise ValueError("No storyboard scene images provided.")

    video = concatenate_videoclips(scene_clips, method="compose")

    # Align total duration with the audio duration.
    if video.duration and not math.isclose(video.duration, total_audio_duration, rel_tol=0.05):
        video = video.set_duration(total_audio_duration)

    video = video.set_audio(audio_clip)

    # NOTE: For simplicity, we don't burn subtitles into frames here.
    # In a production setup, you can use MoviePy's TextClip or ffmpeg filters
    # to overlay subtitles. Here we just save an SRT file next to the video.
    srt_path = output_path.with_suffix(".srt")
    save_srt(subtitles, srt_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    video.write_videofile(
        str(output_path),
        fps=target_fps,
        codec="libx264",
        audio_codec="aac",
        verbose=False,
        logger=None,
    )

    audio_clip.close()
    for c in scene_clips:
        c.close()

    return output_path


def save_srt(subtitles: List[Tuple[float, float, str]], path: Path) -> None:
    """
    Save subtitles as an SRT file.
    """
    def format_time(seconds: float) -> str:
        millis = int((seconds - int(seconds)) * 1000)
        seconds = int(seconds)
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

    lines = []
    for idx, (start, end, text) in enumerate(subtitles, start=1):
        lines.append(str(idx))
        lines.append(f"{format_time(start)} --> {format_time(end)}")
        lines.append(text)
        lines.append("")  # blank line between entries

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


