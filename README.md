## Text-to-Video GEN-AI FastAPI Service

CPU-first, production-structured FastAPI project for a text-to-video pipeline:

1. LLM → script + scene breakdown.
2. Stable Diffusion → storyboard images.
3. TTS → narration audio.
4. MoviePy → combine images + audio into a final MP4.

### Folder structure

- `app/`
  - `main.py` – FastAPI app + endpoints.
  - `config.py` – CPU/GPU model configs.
  - `helper.py` – file/video utilities, subtitles.
  - `pipeline.py` – orchestration of the full pipeline.
  - `models/`
    - `llm.py` – script + scene generation.
    - `storyboard.py` – Stable Diffusion storyboard images.
    - `tts.py` – TTS + subtitles.
- `data/`
  - `storyboards/` – per-run storyboard images.
  - `audio/` – narration audio per run.
  - `videos/` – final MP4 outputs.
  - `tmp/` – scratch/intermediate files.

### Install & run

```bash
cd /home/syedmehrab/Ai_training_projects/TTV-ai
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

uvicorn app.main:app --reload
```

### API usage

- **POST** `/generate_video`

```bash
curl -X POST "http://127.0.0.1:8000/generate_video" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Make a 20-second video about AI robots helping doctors.",
    "target_duration_sec": 20,
    "device_preset": "cpu"
  }'
```

Response:

```json
{
  "run_id": "<id>",
  "status": "PENDING"
}
```

- **GET** `/status/{run_id}` – poll to retrieve status and final video path.

### CPU vs GPU

- Default is **CPU** with:
  - LLM: `gpt2`
  - Storyboard: `Segmind/tiny-sd`
  - TTS: ONNX if configured, otherwise a simple fallback.

- Set environment variables + use `device_preset="gpu"` to enable GPU-heavy models:
  - `LLM_MODEL_GPU`, `STORYBOARD_MODEL_GPU`, `TTS_ONNX_MODEL_GPU`.

Models are downloaded once by `transformers`/`diffusers` and then cached under `~/.cache/huggingface/`.


