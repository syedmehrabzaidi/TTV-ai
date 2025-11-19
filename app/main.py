from __future__ import annotations

from enum import Enum
from typing import Optional, Dict

from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel

from .config import (
    DevicePreset,
    ensure_directories,
    VIDEO_DIR,
)
from .pipeline import create_pipeline, TextToVideoPipeline, PipelineArtifacts
from .helper import generate_run_id


class GenerationStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class GenerateVideoRequest(BaseModel):
    prompt: str
    target_duration_sec: int = 20
    device_preset: DevicePreset = DevicePreset.CPU


class GenerateVideoResponse(BaseModel):
    run_id: str
    status: GenerationStatus
    video_path: Optional[str] = None
    detail: Optional[str] = None


class StatusResponse(BaseModel):
    run_id: str
    status: GenerationStatus
    video_path: Optional[str] = None
    detail: Optional[str] = None


app = FastAPI(title="Text-to-Video GEN-AI API", version="0.1.0")

# In-memory status store. For production, replace with Redis / DB.
RUN_STATUSES: Dict[str, StatusResponse] = {}


def _run_pipeline_background(
    run_id: str,
    pipeline: TextToVideoPipeline,
    prompt: str,
    target_duration_sec: int,
) -> None:
    try:
        RUN_STATUSES[run_id] = StatusResponse(
            run_id=run_id,
            status=GenerationStatus.RUNNING,
        )
        artifacts: PipelineArtifacts = pipeline.run(
            user_prompt=prompt,
            target_duration_sec=target_duration_sec,
            run_id=run_id,
        )
        RUN_STATUSES[run_id] = StatusResponse(
            run_id=run_id,
            status=GenerationStatus.COMPLETED,
            video_path=str(artifacts.video_path),
        )
    except Exception as exc:  # noqa: BLE001
        RUN_STATUSES[run_id] = StatusResponse(
            run_id=run_id,
            status=GenerationStatus.FAILED,
            detail=str(exc),
        )


@app.on_event("startup")
def on_startup() -> None:
    ensure_directories()


@app.post("/generate_video", response_model=GenerateVideoResponse)
async def generate_video(
    payload: GenerateVideoRequest, background_tasks: BackgroundTasks
) -> GenerateVideoResponse:
    """
    Trigger the text-to-video pipeline.

    - `prompt`: e.g. "Make a 20-second video about AI robots helping doctors."
    - `device_preset`: "cpu" or "gpu" to switch model configuration.
    """
    preset = payload.device_preset
    pipeline = create_pipeline(preset)

    # Create a run ID and mark as pending.
    run_id = generate_run_id()

    RUN_STATUSES[run_id] = StatusResponse(
        run_id=run_id,
        status=GenerationStatus.PENDING,
    )

    background_tasks.add_task(
        _run_pipeline_background,
        run_id,
        pipeline,
        payload.prompt,
        payload.target_duration_sec,
    )

    return GenerateVideoResponse(
        run_id=run_id,
        status=GenerationStatus.PENDING,
    )


@app.get("/status/{run_id}", response_model=StatusResponse)
async def status(run_id: str) -> StatusResponse:
    """
    Check the status of a video generation run.
    """
    status_obj = RUN_STATUSES.get(run_id)
    if status_obj is None:
        return StatusResponse(
            run_id=run_id,
            status=GenerationStatus.FAILED,
            detail="Run ID not found.",
        )
    return status_obj


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "video_output_dir": str(VIDEO_DIR)}


