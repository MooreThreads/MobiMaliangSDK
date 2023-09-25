import torch
import os
import sys
from PIL import Image
import argparse

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(__dir__))
from modules.wrappers import RealESRGANWrapper
from modules.shared import device

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint_path",
        default=None,
        type=str,
        required=True,
        help="Path to super resolution model.",
    )

    parser.add_argument(
        "--image_path",
        default=None,
        type=str,
        required=True,
        help="The input image path",
    )

    parser.add_argument(
        "--output_path",
        default="./",
        type=str,
        required=True,
        help="Result folder path",
    )

    args = parser.parse_args()

    realesrgan = RealESRGANWrapper(device=device)
    realesrgan.load_models(model_path=args.checkpoint_path)
    image = Image.open(args.image_path).convert("RGB")
    result = realesrgan(image)

    os.makedirs(args.output_path, exist_ok=True)
    result.save(os.path.join(args.output_path, "sr_img.png"))
