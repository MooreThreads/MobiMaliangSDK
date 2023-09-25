import torch
from PIL import Image
import os
import sys
import argparse

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__dir__)))
from modules.shared import SDModel, Controlnet, device
from modules.wrappers import HEDdetectorWrapper
from modules.controlnet.scribble2img import scribble2img

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
        "--scribble_controlnet_path",
        default=None,
        type=str,
        required=True,
        help="Path to the scribble controlnet model.",
    )
    
    parser.add_argument(
        "--scribble_processor_model_path",
        default=None,
        type=str,
        required=True,
        help="Path to the scribble processor model.",
    )

    parser.add_argument(
        "--image_path", default=None, type=str, required=True, help="The image path"
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

    image = Image.open(args.image_path).convert("RGB").resize((512, 512))
    sd_model = SDModel(device=device)
    controlnet_model = Controlnet(device=sd_model.device)
    processor = HEDdetectorWrapper(device=sd_model.device)
    sd_model.load_models(args.checkpoint_path, diffusers_format=args.diffusers_format)
    controlnet_model.load_models(args.scribble_controlnet_path, diffusers_format=True)
    processor.load_models(args.scribble_processor_model_path)

    images, _, scribble_image = scribble2img(
        sd_model,
        controlnet_model,
        processor,
        ori_image=image,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
    )

    os.makedirs(args.output_path, exist_ok=True)
    images[0].save(os.path.join(args.output_path, "scribble2img.png"))
    scribble_image.save(os.path.join(args.output_path, "scribble.png"))

# python examples/controlnets/example_scribble2img.py --checkpoint_path models/Stable-diffusion/v1-5-pruned-emaonly.ckpt --scribble_controlnet_path models/controlnet/scribble_v11 --scribble_processor_model_path models/controlnet/annotators --image_path data/person.jpg --prompt "a pretty girl" --output_path outputs