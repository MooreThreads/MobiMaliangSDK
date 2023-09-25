import torch
from PIL import Image
import os
import sys
import argparse

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__dir__)))
from modules.shared import SDModel, Controlnet, device
from modules.wrappers import PidiNetDetectorWrapper
from modules.controlnet.softedge2img import softedge2img

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
        "--softedge_controlnet_path",
        default=None,
        type=str,
        required=True,
        help="Path to the softedge controlnet model.",
    )
    
    parser.add_argument(
        "--softedge_processor_model_path",
        default=None,
        type=str,
        required=True,
        help="Path to the softedge processor model.",
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
    processor = PidiNetDetectorWrapper(device="cpu")
    sd_model.load_models(args.checkpoint_path, diffusers_format=args.diffusers_format)
    controlnet_model.load_models(args.softedge_controlnet_path, diffusers_format=True)
    processor.load_models(args.softedge_processor_model_path)

    images, _, softedge_image = softedge2img(
        sd_model,
        controlnet_model,
        processor,
        ori_image=image,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
    )

    os.makedirs(args.output_path, exist_ok=True)
    images[0].save(os.path.join(args.output_path, "softedge2img.png"))
    softedge_image.save(os.path.join(args.output_path, "softedge.png"))

# python examples/controlnets/example_softedge2img.py --checkpoint_path models/Stable-diffusion/v1-5-pruned-emaonly.ckpt --softedge_controlnet_path models/controlnet/softedge_v11 --softedge_processor_model_path models/controlnet/annotators --image_path data/person.jpg --prompt "a pretty girl" --output_path outputs