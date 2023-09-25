import torch
from PIL import Image
import os
import sys
import argparse

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__dir__)))
from modules.shared import SDModel, Controlnet, device
from modules.wrappers import ZoomConditionalImageWrapper
from modules.controlnet.tilehd2img import tilehd2img

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
        "--tile_controlnet_path",
        default=None,
        type=str,
        required=True,
        help="Path to the tile controlnet model.",
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
    processor = ZoomConditionalImageWrapper()
    sd_model.load_models(args.checkpoint_path, diffusers_format=args.diffusers_format)
    controlnet_model.load_models(args.tile_controlnet_path, diffusers_format=True)
    
    images, _, zoom_image = tilehd2img(
        sd_model, 
        controlnet_model, 
        processor, 
        prompt=args.prompt, 
        negative_prompt=args.negative_prompt,
        ori_image=image, 
        up_sampling_ratio=2,
        window_size=80,
        strength=0.35,
    )
    images[0].save("outputs/tilehd2img.png")
    zoom_image.save("outputs/zoom.png")
    
    # python examples/controlnets/example_tilehd2img.py --checkpoint_path models/Stable-diffusion/v1-5-pruned-emaonly.ckpt --tile_controlnet_path models/controlnet/tile_v11 --image_path data/person.jpg --prompt "(best quality, ultra detailed, master piece, sharp:1.3)" --output_path outputs
