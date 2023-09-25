import torch
from typing import Any, Callable, Dict, List, Optional, Union
from PIL import Image
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__dir__)))
from modules.shared import SDModel, Controlnet
from modules.controlnet.controlnet_img2img import controlnet_img2img
from modules.wrappers import ResizeForConditionalImageWrapper


def tile2img(
    sd_model: SDModel,
    controlnet_model: Controlnet,
    processor: ResizeForConditionalImageWrapper,
    ori_image: Image,
    is_tile_image: bool = False,
    down_sampling_ratio: float = 8.0,
    prompt: Union[str, List[str]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    strength: float = 1.0,
    num_inference_steps: int = 20,
    guidance_scale: float = 7.5,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    eta: float = 0.0,
    seed: Optional[Union[int, List[int]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
    callback_steps: int = 1,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    controlnet_conditioning_scale: Union[float, List[float]] = 0.8,
    guess_mode: bool = False,
    control_guidance_start: Union[float, List[float]] = 0.0,
    control_guidance_end: Union[float, List[float]] = 1.0,
    lora_model_name_or_path_or_dict: Optional[Union[str, List[str]]] = None,
    lora_alpha: Optional[float] = 0.75,
    text_inversion_model_name_or_path: Optional[
        Union[str, List[str], Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]
    ] = None,
):

    if is_tile_image:
        tile_image = ori_image
    else:
        tile_image = processor(
            input_image=ori_image, down_sampling_ratio=down_sampling_ratio
        )

    images, nsfw_content_detected = controlnet_img2img(
        sd_model=sd_model,
        controlnet_model=controlnet_model,
        prompt=prompt,
        image=tile_image,
        control_image=tile_image,
        height=height,
        width=width,
        strength=strength,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt,
        num_images_per_prompt=num_images_per_prompt,
        eta=eta,
        seed=seed,
        latents=latents,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        output_type=output_type,
        return_dict=return_dict,
        callback=callback,
        callback_steps=callback_steps,
        cross_attention_kwargs=cross_attention_kwargs,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        guess_mode=guess_mode,
        control_guidance_start=control_guidance_start,
        control_guidance_end=control_guidance_end,
        lora_model_name_or_path_or_dict=lora_model_name_or_path_or_dict,
        lora_alpha=lora_alpha,
        text_inversion_model_name_or_path=text_inversion_model_name_or_path,
    )

    return images, nsfw_content_detected, tile_image
