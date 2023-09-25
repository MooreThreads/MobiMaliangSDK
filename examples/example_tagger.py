
import os
import sys
import torch
from PIL import Image
import argparse

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(__dir__))
from modules.wrappers import DeepDanbooruWrapper
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

    args = parser.parse_args()

    tagger = DeepDanbooruWrapper(device=device)
    tagger.load_models(model_path=args.checkpoint_path)
    image = Image.open(args.image_path).convert("RGB")
    result = tagger(image)
    print(result)
