from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from diffusers import AutoPipelineForText2Image
from PIL import Image
from PIL.PngImagePlugin import PngInfo


# ============================================================
# Mathematical constants
# ============================================================

PHI = (1.0 + math.sqrt(5.0)) / 2.0
GOLDEN_LOWER = 1.0 / PHI
GOLDEN_UPPER = PHI


# ============================================================
# HSN prompt-conditioning layer
# ============================================================

@dataclass
class HSNConfig:
    """
    Relative influence of each philosophical prompt lens.

    Values are normalized automatically before use.
    """

    heidegger_weight: float = 1.0
    spinoza_weight: float = 1.0
    nietzsche_weight: float = 1.0

    def normalized(self) -> "HSNConfig":
        values = [
            max(0.0, float(self.heidegger_weight)),
            max(0.0, float(self.spinoza_weight)),
            max(0.0, float(self.nietzsche_weight)),
        ]

        total = sum(values)

        if total <= 0.0:
            return HSNConfig(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)

        return HSNConfig(
            heidegger_weight=values[0] / total,
            spinoza_weight=values[1] / total,
            nietzsche_weight=values[2] / total,
        )


class HeideggerModule:
    """Phenomenological and environmental articulation."""

    def describe(self, concept: str) -> Dict[str, Any]:
        return {
            "lens": "Heidegger",
            "focus": "being-in-the-world",
            "keywords": [
                "presence",
                "worldhood",
                "material familiarity",
                "tactile reality",
                "environmental placement",
            ],
            "prompt_fragment": (
                f"Present {concept} as materially present within a coherent world. "
                "Emphasize tactile surfaces, environmental context, subtle wear, "
                "physical weight, grounded spatial relationships, and a convincing "
                "sense of presence."
            ),
        }


class SpinozaModule:
    """Continuity, unity, and immanent-causality articulation."""

    def describe(self, concept: str) -> Dict[str, Any]:
        return {
            "lens": "Spinoza",
            "focus": "substance-and-modes",
            "keywords": [
                "unity",
                "continuity",
                "immanent causality",
                "extension",
                "visual coherence",
            ],
            "prompt_fragment": (
                f"Render {concept} as part of one visually continuous world. "
                "Use unified forms, coherent materials, smooth transitions, "
                "consistent illumination, and seamless interaction between the "
                "subject, environment, smoke, light, and surrounding objects."
            ),
        }


class NietzscheModule:
    """Force, intensity, contrast, and movement articulation."""

    def describe(self, concept: str) -> Dict[str, Any]:
        return {
            "lens": "Nietzsche",
            "focus": "force-and-becoming",
            "keywords": [
                "force",
                "intensity",
                "contrast",
                "becoming",
                "overflowing energy",
            ],
            "prompt_fragment": (
                f"Infuse {concept} with force and intense visual presence. "
                "Use bold contrasts, dramatic directional lighting, expressive "
                "posture, concentrated energy, decisive silhouette, and dynamic "
                "atmospheric movement."
            ),
        }


class HSNEngine:
    """
    Produces a structured HSN decomposition and a diffusion-friendly prompt.

    Philosophical labels are retained in metadata, but are not inserted as
    bracketed tokens into the final image prompt. This avoids wasting SDXL
    tokenizer capacity on abstract labels.
    """

    def __init__(self, config: Optional[HSNConfig] = None):
        self.config = (config or HSNConfig()).normalized()
        self.heidegger = HeideggerModule()
        self.spinoza = SpinozaModule()
        self.nietzsche = NietzscheModule()

    @staticmethod
    def _repeat_by_weight(fragment: str, weight: float) -> str:
        """
        Convert a normalized weight into modest textual emphasis.

        SDXL does not natively understand arbitrary numeric lens weights.
        Repetition is deliberately limited to avoid prompt degradation.
        """
        if weight >= 0.45:
            return f"{fragment} Strongly emphasize this quality."
        if weight >= 0.25:
            return fragment
        if weight > 0.0:
            return f"Subtly incorporate the following quality: {fragment}"
        return ""

    def generate(self, concept: str) -> Dict[str, Any]:
        concept = concept.strip()

        if not concept:
            raise ValueError("The concept cannot be empty.")

        h_desc = self.heidegger.describe(concept)
        s_desc = self.spinoza.describe(concept)
        n_desc = self.nietzsche.describe(concept)

        decomposition = {
            "heidegger": h_desc,
            "spinoza": s_desc,
            "nietzsche": n_desc,
        }

        fragments = [
            self._repeat_by_weight(
                h_desc["prompt_fragment"],
                self.config.heidegger_weight,
            ),
            self._repeat_by_weight(
                s_desc["prompt_fragment"],
                self.config.spinoza_weight,
            ),
            self._repeat_by_weight(
                n_desc["prompt_fragment"],
                self.config.nietzsche_weight,
            ),
        ]

        combined_prompt = " ".join(
            fragment for fragment in fragments if fragment
        )

        return {
            "concept": concept,
            "config": asdict(self.config),
            "decomposition": decomposition,
            "combined_prompt": combined_prompt,
        }


# ============================================================
# ZCMS layer
# ============================================================

class ZCMSLayer:
    """
    Maps the normalized HSN weights onto three points around a circle.

    These values are used as conditioning metadata and as bounded influences
    on guidance scale and inference-step selection.
    """

    ANGLES = {
        "Heidegger": 0.0,
        "Spinoza": 2.0 * math.pi / 3.0,
        "Nietzsche": 4.0 * math.pi / 3.0,
    }

    def __init__(self, radius: float = 1.0):
        if radius <= 0.0:
            raise ValueError("ZCMS radius must be greater than zero.")

        self.radius = float(radius)

    def map_to_circle(
        self,
        lens: str,
        weight: float,
    ) -> Dict[str, Any]:
        angle = self.ANGLES.get(lens, 0.0)

        return {
            "lens": lens,
            "zone": f"ZC-{lens[0]}",
            "angle_radians": angle,
            "coordinates": {
                "x": self.radius * math.cos(angle) * weight,
                "y": self.radius * math.sin(angle) * weight,
            },
            "field_intensity": weight,
        }

    def apply(self, hsn_output: Dict[str, Any]) -> Dict[str, Any]:
        config = hsn_output["config"]

        mapped = {
            "Heidegger": self.map_to_circle(
                "Heidegger",
                config["heidegger_weight"],
            ),
            "Spinoza": self.map_to_circle(
                "Spinoza",
                config["spinoza_weight"],
            ),
            "Nietzsche": self.map_to_circle(
                "Nietzsche",
                config["nietzsche_weight"],
            ),
        }

        return {
            **hsn_output,
            "zero_circle_map": mapped,
        }

    @staticmethod
    def generation_parameters(
        zcms_output: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Derive restrained SDXL parameters from the normalized lens weights.

        The ranges are intentionally conservative:
          guidance_scale: approximately 5.5 to 8.0
          steps: 28 to 42
        """
        config = zcms_output["config"]

        h = config["heidegger_weight"]
        s = config["spinoza_weight"]
        n = config["nietzsche_weight"]

        # Nietzsche weight increases visual force and prompt adherence.
        guidance_scale = 5.5 + (2.5 * n)

        # Heidegger and Spinoza increase time spent resolving material context.
        steps = round(28 + 8 * h + 6 * s)

        return {
            "guidance_scale": round(guidance_scale, 2),
            "num_inference_steps": int(max(20, min(50, steps))),
        }


# ============================================================
# Prompt assembly
# ============================================================

@dataclass
class VisualStyle:
    positive: str = (
        "vintage nineteenth-century gothic storybook engraving, "
        "finely detailed pen-and-ink linework, engraved cross-hatching, "
        "dramatic red-orange firelight, deep smoky shadows, "
        "carefully controlled composition, realistic anatomy, "
        "highly detailed illustration"
    )

    negative: str = (
        "text, letters, typography, signature, caption, watermark, logo, "
        "modern clothing, modern furniture, science fiction, anime, chibi, "
        "cute mascot, cheerful expression, comedy, gore, blood, exposed organs, "
        "extra arms, extra legs, extra fingers, missing fingers, fused fingers, "
        "malformed hands, malformed anatomy, duplicate subject, cropped head, "
        "out of frame, blurry, low detail, low resolution, jpeg artifacts"
    )


class PromptBuilder:
    def __init__(self, style: Optional[VisualStyle] = None):
        self.style = style or VisualStyle()

    def build(
        self,
        zcms_output: Dict[str, Any],
        additional_prompt: str = "",
        negative_prompt: Optional[str] = None,
    ) -> Dict[str, str]:
        concept = zcms_output["concept"]
        conditioned = zcms_output["combined_prompt"]

        positive_parts = [
            concept,
            conditioned,
            self.style.positive,
            additional_prompt.strip(),
        ]

        positive = ". ".join(
            part.strip(" .")
            for part in positive_parts
            if part and part.strip()
        )

        negative = (
            negative_prompt.strip()
            if negative_prompt and negative_prompt.strip()
            else self.style.negative
        )

        return {
            "positive": positive,
            "negative": negative,
        }


# ============================================================
# SDXL backend
# ============================================================

@dataclass
class SDXLConfig:
    model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"

    # Native SDXL resolution is 1024 x 1024.
    width: int = 1024
    height: int = 1024

    guidance_scale: Optional[float] = None
    num_inference_steps: Optional[int] = None

    # Memory-management options
    cpu_offload: bool = False
    sequential_cpu_offload: bool = False
    attention_slicing: bool = False
    vae_slicing: bool = True
    vae_tiling: bool = False

    # Optional optimization. Requires an available compatible backend.
    enable_xformers: bool = False

    # Use float32 on CPU. CUDA normally uses float16.
    force_float32: bool = False


class SDXLBackend:
    def __init__(self, config: Optional[SDXLConfig] = None):
        self.config = config or SDXLConfig()
        self.device = self._select_device()
        self.dtype = self._select_dtype()
        self.pipeline = self._load_pipeline()

    @staticmethod
    def _select_device() -> str:
        if torch.cuda.is_available():
            return "cuda"

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"

        return "cpu"

    def _select_dtype(self) -> torch.dtype:
        if self.config.force_float32:
            return torch.float32

        if self.device == "cuda":
            return torch.float16

        # MPS float16 can work, but float32 is generally more predictable.
        return torch.float32

    def _load_pipeline(self):
        print(
            f"Loading SDXL model '{self.config.model_id}' "
            f"on {self.device} with {self.dtype}..."
        )

        load_options: Dict[str, Any] = {
            "torch_dtype": self.dtype,
            "use_safetensors": True,
        }

        # variant="fp16" should only be requested for half-precision CUDA.
        if self.dtype == torch.float16:
            load_options["variant"] = "fp16"

        pipeline = AutoPipelineForText2Image.from_pretrained(
            self.config.model_id,
            **load_options,
        )

        if self.device == "cuda":
            if self.config.sequential_cpu_offload:
                pipeline.enable_sequential_cpu_offload()
            elif self.config.cpu_offload:
                pipeline.enable_model_cpu_offload()
            else:
                pipeline.to("cuda")
        else:
            pipeline.to(self.device)

        if self.config.attention_slicing:
            pipeline.enable_attention_slicing()

        if self.config.vae_slicing:
            pipeline.enable_vae_slicing()

        if self.config.vae_tiling:
            pipeline.enable_vae_tiling()

        if self.config.enable_xformers:
            try:
                pipeline.enable_xformers_memory_efficient_attention()
            except Exception as exc:
                print(
                    "Warning: xFormers could not be enabled. "
                    f"Generation will continue without it: {exc}"
                )

        # Disable the progress bar only if desired. Leaving it enabled gives
        # useful feedback during long generations.
        pipeline.set_progress_bar_config(disable=False)

        return pipeline

    def make_generator(self, seed: int) -> torch.Generator:
        """
        Create a deterministic generator on a supported device.

        CPU generators avoid some backend-specific generator incompatibilities.
        """
        generator_device = "cuda" if self.device == "cuda" else "cpu"
        return torch.Generator(device=generator_device).manual_seed(seed)

    def generate(
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
        guidance_scale: float,
        num_inference_steps: int,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> Image.Image:
        width = width or self.config.width
        height = height or self.config.height

        if width % 8 != 0 or height % 8 != 0:
            raise ValueError("Width and height must be divisible by 8.")

        generator = self.make_generator(seed)

        with torch.inference_mode():
            result = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
            )

        return result.images[0]


# ============================================================
# Unified HSNAI interface
# ============================================================

@dataclass
class GenerationResult:
    image: Image.Image
    output_path: Path
    metadata_path: Path
    seed: int
    prompt: str
    negative_prompt: str
    parameters: Dict[str, Any]


class HSNAI:
    def __init__(
        self,
        hsn_config: Optional[HSNConfig] = None,
        sdxl_config: Optional[SDXLConfig] = None,
        zcms_radius: float = 1.0,
        style: Optional[VisualStyle] = None,
    ):
        self.hsn = HSNEngine(hsn_config)
        self.zcms = ZCMSLayer(zcms_radius)
        self.prompt_builder = PromptBuilder(style)
        self.sdxl = SDXLBackend(sdxl_config)

    @staticmethod
    def _create_seed(seed: Optional[int]) -> int:
        if seed is not None:
            if seed < 0:
                raise ValueError("Seed must be zero or greater.")
            return int(seed)

        return random.SystemRandom().randint(0, 2**31 - 1)

    @staticmethod
    def _png_metadata(metadata: Dict[str, Any]) -> PngInfo:
        png_info = PngInfo()
        png_info.add_text("HSNAI", json.dumps(metadata, ensure_ascii=False))
        png_info.add_text("Prompt", metadata["prompt"])
        png_info.add_text("Negative Prompt", metadata["negative_prompt"])
        png_info.add_text("Seed", str(metadata["seed"]))
        return png_info

    def generate_image(
        self,
        concept: str,
        output_path: str | Path = "hsnai_output.png",
        seed: Optional[int] = None,
        additional_prompt: str = "",
        negative_prompt: Optional[str] = None,
        guidance_scale: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> GenerationResult:
        # 1. Philosophical decomposition
        hsn_output = self.hsn.generate(concept)

        # 2. ZCMS mapping
        zcms_output = self.zcms.apply(hsn_output)

        # 3. Diffusion prompt creation
        prompts = self.prompt_builder.build(
            zcms_output=zcms_output,
            additional_prompt=additional_prompt,
            negative_prompt=negative_prompt,
        )

        # 4. Derive bounded parameters from ZCMS
        derived = self.zcms.generation_parameters(zcms_output)

        final_guidance = (
            guidance_scale
            if guidance_scale is not None
            else self.sdxl.config.guidance_scale
        )
        if final_guidance is None:
            final_guidance = derived["guidance_scale"]

        final_steps = (
            num_inference_steps
            if num_inference_steps is not None
            else self.sdxl.config.num_inference_steps
        )
        if final_steps is None:
            final_steps = derived["num_inference_steps"]

        final_seed = self._create_seed(seed)

        final_width = width or self.sdxl.config.width
        final_height = height or self.sdxl.config.height

        # 5. Render
        image = self.sdxl.generate(
            prompt=prompts["positive"],
            negative_prompt=prompts["negative"],
            seed=final_seed,
            guidance_scale=float(final_guidance),
            num_inference_steps=int(final_steps),
            width=final_width,
            height=final_height,
        )

        # 6. Save image and complete generation record
        output_path = Path(output_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix.lower() != ".png":
            output_path = output_path.with_suffix(".png")

        metadata_path = output_path.with_suffix(".json")

        metadata = {
            "engine": "HSNAI-SDXL",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "model_id": self.sdxl.config.model_id,
            "device": self.sdxl.device,
            "dtype": str(self.sdxl.dtype),
            "seed": final_seed,
            "prompt": prompts["positive"],
            "negative_prompt": prompts["negative"],
            "parameters": {
                "width": final_width,
                "height": final_height,
                "guidance_scale": float(final_guidance),
                "num_inference_steps": int(final_steps),
            },
            "hsn": hsn_output,
            "zcms": zcms_output["zero_circle_map"],
        }

        png_info = self._png_metadata(metadata)
        image.save(output_path, format="PNG", pnginfo=png_info)

        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        return GenerationResult(
            image=image,
            output_path=output_path,
            metadata_path=metadata_path,
            seed=final_seed,
            prompt=prompts["positive"],
            negative_prompt=prompts["negative"],
            parameters=metadata["parameters"],
        )


# ============================================================
# Default devil concept
# ============================================================

DEFAULT_DEVIL_CONCEPT = (
    "A classic traditional devil from old European folklore, shown full body, "
    "with crimson-red skin, two large curved black horns, pointed ears, "
    "glowing golden-yellow eyes, a sharp black pointed goatee, and a sly, "
    "intimidating expression. He has black cloven hooves and a long red tail "
    "ending in a sharp arrow-shaped point. He holds a traditional black "
    "three-pronged pitchfork and stands beside an antique wooden writing desk. "
    "On the desk rests an aged parchment soul contract, a feather quill, "
    "a burning candle, and a red wax seal. The scene is theatrical and "
    "supernatural, menacing but not gruesome"
)


# ============================================================
# Command-line interface
# ============================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate images through the HSNAI prompt layer and SDXL."
    )

    parser.add_argument(
        "--concept",
        default=DEFAULT_DEVIL_CONCEPT,
        help="Base visual concept.",
    )
    parser.add_argument(
        "--output",
        default="hsnai_devil.png",
        help="PNG output path.",
    )
    parser.add_argument(
        "--model",
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Diffusers-compatible text-to-image model identifier.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Reproducible random seed. A random seed is used if omitted.",
    )
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)

    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Inference steps. If omitted, HSNAI derives the value.",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=None,
        help="CFG guidance scale. If omitted, HSNAI derives the value.",
    )

    parser.add_argument("--heidegger", type=float, default=1.2)
    parser.add_argument("--spinoza", type=float, default=1.0)
    parser.add_argument("--nietzsche", type=float, default=1.5)

    parser.add_argument(
        "--additional-prompt",
        default="",
        help="Extra positive prompt terms.",
    )
    parser.add_argument(
        "--negative-prompt",
        default=None,
        help="Override the default negative prompt.",
    )

    parser.add_argument(
        "--cpu-offload",
        action="store_true",
        help="Use model CPU offloading to reduce CUDA VRAM usage.",
    )
    parser.add_argument(
        "--sequential-cpu-offload",
        action="store_true",
        help="Use stronger sequential CPU offloading.",
    )
    parser.add_argument(
        "--attention-slicing",
        action="store_true",
        help="Enable attention slicing to reduce memory use.",
    )
    parser.add_argument(
        "--vae-tiling",
        action="store_true",
        help="Enable VAE tiling.",
    )
    parser.add_argument(
        "--xformers",
        action="store_true",
        help="Attempt to enable xFormers memory-efficient attention.",
    )
    parser.add_argument(
        "--float32",
        action="store_true",
        help="Force float32, including on CUDA.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    if args.cpu_offload and args.sequential_cpu_offload:
        raise ValueError(
            "Choose either --cpu-offload or --sequential-cpu-offload, "
            "not both."
        )

    hsn_config = HSNConfig(
        heidegger_weight=args.heidegger,
        spinoza_weight=args.spinoza,
        nietzsche_weight=args.nietzsche,
    )

    sdxl_config = SDXLConfig(
        model_id=args.model,
        width=args.width,
        height=args.height,
        guidance_scale=args.guidance,
        num_inference_steps=args.steps,
        cpu_offload=args.cpu_offload,
        sequential_cpu_offload=args.sequential_cpu_offload,
        attention_slicing=args.attention_slicing,
        vae_slicing=True,
        vae_tiling=args.vae_tiling,
        enable_xformers=args.xformers,
        force_float32=args.float32,
    )

    hsnai = HSNAI(
        hsn_config=hsn_config,
        sdxl_config=sdxl_config,
        zcms_radius=1.0,
    )

    result = hsnai.generate_image(
        concept=args.concept,
        output_path=args.output,
        seed=args.seed,
        additional_prompt=args.additional_prompt,
        negative_prompt=args.negative_prompt,
        guidance_scale=args.guidance,
        num_inference_steps=args.steps,
        width=args.width,
        height=args.height,
    )

    print("\nGeneration complete")
    print(f"Image:    {result.output_path}")
    print(f"Metadata: {result.metadata_path}")
    print(f"Seed:     {result.seed}")
    print(f"Settings: {result.parameters}")
    print("\nPrompt:")
    print(result.prompt)
    print("\nNegative prompt:")
    print(result.negative_prompt)


if __name__ == "__main__":
    main()
