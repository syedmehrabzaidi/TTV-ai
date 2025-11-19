from __future__ import annotations

from dataclasses import dataclass
from typing import List

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from ..config import LLMConfig


@dataclass
class ScriptAndScenes:
    script: str
    scenes: List[str]


class LLMEngine:
    """
    Lightweight wrapper around a text-generation LLM.

    CPU-first default: GPT-2 (small, widely available).
    GPU option: larger instruction-tuned models (e.g. Llama-3 family).
    """

    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg
        self._pipe = None

    def _load(self):
        if self._pipe is None:
            tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name)
            model = AutoModelForCausalLM.from_pretrained(
                self.cfg.model_name,
                device_map="auto" if self.cfg.device != "cpu" else None,
            )
            self._pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=0 if self.cfg.device != "cpu" else -1,
            )

    def _generate(self, prompt: str) -> str:
        self._load()
        out = self._pipe(
            prompt,
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )[0]["generated_text"]
        return out[len(prompt) :].strip()

    def generate_script_and_scenes(self, user_prompt: str, target_duration_sec: int = 20) -> ScriptAndScenes:
        """
        Step 1 + 2: generate a script and a simple scene breakdown.

        The scene breakdown is returned as a list of textual scene descriptions.
        """
        prompt = (
            "You are a helpful assistant that writes short video scripts.\n"
            f"User request: {user_prompt}\n\n"
            f"Write a concise narration script for a ~{target_duration_sec}-second video.\n"
            "Then break the video into 4-6 numbered visual scenes with 1 line descriptions.\n\n"
            "Format:\n"
            "SCRIPT:\n"
            "<narration text>\n"
            "SCENES:\n"
            "1) <scene one description>\n"
            "2) <scene two description>\n"
            "...\n"
        )
        result = self._generate(prompt)
        print("--------result-----------generate_script_and_scenes---------------------------------",result)

        # Parse script and scenes.
        script = ""
        scenes: List[str] = []

        if "SCENES:" in result:
            script_part, scenes_part = result.split("SCENES:", maxsplit=1)
            # Strip the leading 'SCRIPT:' if present.
            script = script_part.replace("SCRIPT:", "").strip()

            for line in scenes_part.splitlines():
                line = line.strip()
                if not line:
                    continue
                if line[0].isdigit():
                    # Assume format like "1) description" or "1. description"
                    parts = line.split(")", 1)
                    if len(parts) == 1:
                        parts = line.split(".", 1)
                    if len(parts) == 2:
                        scenes.append(parts[1].strip())
                    else:
                        scenes.append(line)
        else:
            script = result.strip()
            scenes = [user_prompt]

        if not scenes:
            scenes = [user_prompt]

        return ScriptAndScenes(script=script, scenes=scenes)


