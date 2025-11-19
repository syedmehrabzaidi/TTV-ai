from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
from diffusers import StableDiffusionPipeline

from ..config import StoryboardConfig


@dataclass
class StoryboardResult:
    scene_image_paths: List[Path]


class StoryboardEngine:
    """
    Stable Diffusion-based storyboard generator.

    CPU-first default: a tiny SD model (`Segmind/tiny-sd`).
    GPU option: heavier models like `stabilityai/stable-diffusion-xl-base-1.0`.
    """

    def __init__(self, cfg: StoryboardConfig) -> None:
        self.cfg = cfg
        self._pipe: StableDiffusionPipeline | None = None

    def _load(self):
        if self._pipe is None:
            self._pipe = StableDiffusionPipeline.from_pretrained(
                self.cfg.model_name,
                torch_dtype=torch.float16 if self.cfg.device != "cpu" else torch.float32,
            )
            if self.cfg.device != "cpu":
                self._pipe = self._pipe.to("cuda")
            else:
                self._pipe = self._pipe.to("cpu")
                self._pipe.enable_attention_slicing()

    def generate_storyboard(
        self,
        scene_descriptions: List[str],
        output_dir: Path,
    ) -> StoryboardResult:
        """
        Step 3: Generate one image per scene description.
        """
        self._load()
        output_dir.mkdir(parents=True, exist_ok=True)

        scene_image_paths: List[Path] = []
        for idx, desc in enumerate(scene_descriptions):
            prompt = desc
            image = self._pipe(
                prompt,
                num_inference_steps=self.cfg.num_inference_steps,
                height=self.cfg.image_size,
                width=self.cfg.image_size,
            ).images[0]

            out_path = output_dir / f"scene_{idx+1:02}.png"
            image.save(out_path)
            scene_image_paths.append(out_path)

        return StoryboardResult(scene_image_paths=scene_image_paths)


