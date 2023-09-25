import torch
import os
import sys
import argparse

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(__dir__))
from modules.shared import SDModel, device
from modules.basic.txt2img import txt2img

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint_path",
        default=None,
        type=str,
        required=True,
        help="Path to the stable diffusion model.",
    )

    parser.add_argument(
        "--prompt",
        default=None,
        type=str,
        required=True,
        help="Describe the image content with text",
    )

    parser.add_argument(
        "--output_path",
        default="./",
        type=str,
        required=True,
        help="Result folder path",
    )

    parser.add_argument(
        "--negative_prompt",
        default=None,
        type=str,
        required=False,
        help="Content which you don't want to show up with text",
    )

    parser.add_argument(
        "--diffusers_format",
        action="store_true",
        required=False,
        help="Model files in diffusers format",
    )

    parser.add_argument(
        "--seed", default=2023, required=False, type=int, help="Random seed"
    )

    args = parser.parse_args()

    sd_model = SDModel(device=device, requires_safety_checker=False)
    sd_model.load_models(args.checkpoint_path, diffusers_format=args.diffusers_format)

    images, status = txt2img(
        sd_model,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
    )

    os.makedirs(args.output_path, exist_ok=True)
    images[0].save(os.path.join(args.output_path, "txt2img.png"))
