#!/usr/bin/env python3
"""
Z8 HSNAI image generator + ImageMagick finishing layer.

Text-to-image:
    python3 generator.py any-description.txt output.png

Image-to-image:
    python3 z8.txt output.png --input source.jpg \
        --prompt "preserve the subject, transform it into a gothic engraving"

The first positional argument is a TXT description file; the second is the output image.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import torch
from PIL import Image, ImageOps
from PIL.PngImagePlugin import PngInfo
from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image

PHI = (1.0 + math.sqrt(5.0)) / 2.0


@dataclass
class HSNConfig:
    heidegger: float = 1.2
    spinoza: float = 1.0
    nietzsche: float = 1.5

    def normalized(self) -> "HSNConfig":
        values = [max(0.0, self.heidegger), max(0.0, self.spinoza), max(0.0, self.nietzsche)]
        total = sum(values)
        if total <= 0:
            return HSNConfig(1 / 3, 1 / 3, 1 / 3)
        return HSNConfig(*(v / total for v in values))


class HSNEngine:
    def __init__(self, config: HSNConfig):
        self.config = config.normalized()

    @staticmethod
    def emphasize(text: str, weight: float) -> str:
        if weight >= 0.45:
            return text + " Strongly emphasize this quality."
        if weight >= 0.25:
            return text
        if weight > 0:
            return "Subtly incorporate this quality: " + text
        return ""

    def build_prompt(self, concept: str, style: str) -> str:
        h = (
            "Give the subject tactile material presence, coherent environmental placement, "
            "physical weight, subtle wear, and grounded spatial relationships."
        )
        s = (
            "Create visual unity through coherent materials, continuous forms, smooth transitions, "
            "consistent illumination, and seamless interaction with the environment."
        )
        n = (
            "Use decisive silhouettes, concentrated energy, bold contrast, expressive movement, "
            "and dramatic directional lighting."
        )
        pieces = [
            concept.strip(),
            self.emphasize(h, self.config.heidegger),
            self.emphasize(s, self.config.spinoza),
            self.emphasize(n, self.config.nietzsche),
            style.strip(),
        ]
        return " ".join(p for p in pieces if p)

    def parameters(self) -> dict[str, Any]:
        c = self.config
        guidance = round(5.5 + 2.5 * c.nietzsche, 2)
        steps = round(28 + 8 * c.heidegger + 6 * c.spinoza)
        angles = {"Heidegger": 0.0, "Spinoza": 2 * math.pi / 3, "Nietzsche": 4 * math.pi / 3}
        weights = {"Heidegger": c.heidegger, "Spinoza": c.spinoza, "Nietzsche": c.nietzsche}
        circle = {
            name: {
                "angle_radians": angles[name],
                "x": math.cos(angles[name]) * weight,
                "y": math.sin(angles[name]) * weight,
                "field_intensity": weight,
            }
            for name, weight in weights.items()
        }
        return {"guidance_scale": guidance, "steps": steps, "zero_circle": circle}


DEFAULT_PROMPT = "A visually compelling subject with a clear focal point and complete composition"
DEFAULT_STYLE = (
    "highly detailed, polished composition, convincing anatomy and perspective, "
    "rich material texture, cinematic lighting, no empty space"
)
DEFAULT_NEGATIVE = (
    "text, letters, caption, watermark, logo, malformed anatomy, extra fingers, "
    "missing fingers, duplicate subject, cropped head, blurry, low detail, low resolution"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HSNAI SDXL generator with ImageMagick output finishing")
    p.add_argument("description_file", help="Any .txt filename containing the image description")
    p.add_argument("output", help="Any output image filename, such as output.png")
    p.add_argument("--prompt", default=None, help="Optional override; otherwise the .txt file contents are the prompt")
    p.add_argument("--input", "--image", dest="input", default=None, help="Optional source image for image-to-image mode")
    p.add_argument("--strength", type=float, default=0.55, help="Image-to-image change strength, 0 < value <= 1")
    p.add_argument("--negative", default=DEFAULT_NEGATIVE, help="Negative prompt")
    p.add_argument("--style", default=DEFAULT_STYLE, help="Additional visual style")
    p.add_argument("--model", default="stabilityai/stable-diffusion-xl-base-1.0")
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--guidance", type=float, default=None)
    p.add_argument("--heidegger", type=float, default=1.2)
    p.add_argument("--spinoza", type=float, default=1.0)
    p.add_argument("--nietzsche", type=float, default=1.5)
    p.add_argument("--cpu-offload", action="store_true")
    p.add_argument("--sequential-cpu-offload", action="store_true")
    p.add_argument("--imagemagick", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--auto-level", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--sharpen", default="0x0.6", help="ImageMagick sharpen geometry; use 0 to disable")
    p.add_argument("--quality", type=int, default=95)
    return p.parse_args()


def select_device() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


def configure_pipe(pipe, device: str, cpu_offload: bool, sequential: bool):
    if sequential and cpu_offload:
        raise ValueError("Choose only one of --cpu-offload and --sequential-cpu-offload")
    if device == "cuda" and sequential:
        pipe.enable_sequential_cpu_offload()
    elif device == "cuda" and cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    pipe.enable_vae_slicing()
    return pipe


def load_source(path: str, width: int, height: int) -> Image.Image:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input image not found: {source}")
    with Image.open(source) as im:
        image = ImageOps.exif_transpose(im).convert("RGB")
    # Fill the requested canvas without distortion.
    return ImageOps.fit(image, (width, height), method=Image.Resampling.LANCZOS)


def find_imagemagick() -> list[str]:
    magick = shutil.which("magick")
    if magick:
        return [magick]
    convert = shutil.which("convert")
    if convert:
        return [convert]
    raise FileNotFoundError("ImageMagick was not found. Install it with: sudo apt install imagemagick")


def finish_with_imagemagick(source: Path, output: Path, args: argparse.Namespace) -> None:
    command = find_imagemagick() + [str(source), "-auto-orient", "-colorspace", "sRGB"]
    if args.auto_level:
        command.append("-auto-level")
    if args.sharpen and args.sharpen != "0":
        command += ["-sharpen", args.sharpen]
    command += ["-quality", str(max(1, min(100, args.quality))), str(output)]
    subprocess.run(command, check=True)


def main() -> int:
    args = parse_args()
    if args.width % 8 or args.height % 8:
        raise ValueError("Width and height must be divisible by 8")
    if args.input and not (0.0 < args.strength <= 1.0):
        raise ValueError("--strength must be greater than 0 and no greater than 1")

    description_file = Path(args.description_file).expanduser().resolve()
    if not description_file.is_file():
        raise FileNotFoundError(f"Description file not found: {description_file}")
    file_prompt = description_file.read_text(encoding="utf-8").strip()
    if not file_prompt:
        raise ValueError("Description file is empty")
    concept = args.prompt.strip() if args.prompt and args.prompt.strip() else file_prompt

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    seed = args.seed if args.seed is not None else random.SystemRandom().randint(0, 2**31 - 1)

    hsn = HSNEngine(HSNConfig(args.heidegger, args.spinoza, args.nietzsche))
    prompt = hsn.build_prompt(concept, args.style)
    derived = hsn.parameters()
    guidance = args.guidance if args.guidance is not None else derived["guidance_scale"]
    steps = args.steps if args.steps is not None else derived["steps"]

    device, dtype = select_device()
    load_options: dict[str, Any] = {"torch_dtype": dtype, "use_safetensors": True}
    if dtype == torch.float16:
        load_options["variant"] = "fp16"

    if args.input:
        pipe = AutoPipelineForImage2Image.from_pretrained(args.model, **load_options)
    else:
        pipe = AutoPipelineForText2Image.from_pretrained(args.model, **load_options)
    pipe = configure_pipe(pipe, device, args.cpu_offload, args.sequential_cpu_offload)

    generator_device = "cuda" if device == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(seed)
    call: dict[str, Any] = {
        "prompt": prompt,
        "negative_prompt": args.negative,
        "guidance_scale": float(guidance),
        "num_inference_steps": int(steps),
        "generator": generator,
    }
    if args.input:
        call["image"] = load_source(args.input, args.width, args.height)
        call["strength"] = args.strength
    else:
        call["width"] = args.width
        call["height"] = args.height

    with torch.inference_mode():
        image = pipe(**call).images[0]

    metadata = {
        "engine": "Z8-HSNAI-SDXL-ImageMagick",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "mode": "image-to-image" if args.input else "text-to-image",
        "description_file": str(description_file),
        "input": str(Path(args.input).expanduser().resolve()) if args.input else None,
        "output": str(output),
        "device": device,
        "dtype": str(dtype),
        "seed": seed,
        "prompt": prompt,
        "negative_prompt": args.negative,
        "width": args.width,
        "height": args.height,
        "strength": args.strength if args.input else None,
        "guidance_scale": guidance,
        "steps": steps,
        "hsn": asdict(hsn.config),
        "zero_circle": derived["zero_circle"],
        "golden_ratio": PHI,
        "imagemagick": bool(args.imagemagick),
    }

    with tempfile.TemporaryDirectory(prefix="z8_") as tmp:
        raw = Path(tmp) / "raw.png"
        info = PngInfo()
        info.add_text("Z8-HSNAI", json.dumps(metadata, ensure_ascii=False))
        image.save(raw, format="PNG", pnginfo=info)
        if args.imagemagick:
            finish_with_imagemagick(raw, output, args)
        else:
            shutil.copy2(raw, output)

    metadata_path = output.with_suffix(output.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Image: {output}")
    print(f"Metadata: {metadata_path}")
    print(f"Seed: {seed}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Cancelled.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"z8 error: {exc}", file=sys.stderr)
        raise SystemExit(1)
