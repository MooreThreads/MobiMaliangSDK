import torch
from diffusers.utils import is_accelerate_available, is_accelerate_version
from typing import Any, Callable, Dict, List, Optional, Union
import gc
import PIL
import numpy as np
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(__dir__))
from shared import SDModel, Controlnet, MTGPU_DETECTION
from modules.pipelines.controlnet_tile_hd import (
    StableDiffusionControlNetTileImg2ImgPipeline,
)

# 精绘功能，将图像通过 prompt 控制，进行局部重绘的功能。可以作为加强版的 image to image 功能；也可以用作放大图像功能使用。

def controlnet_tileimg2img(
    sd_model: SDModel,
    controlnet_model: Controlnet,
    prompt: Union[str, List[str]] = None,
    image: Union[
        torch.FloatTensor,
        PIL.Image.Image,
        np.ndarray,
        List[torch.FloatTensor],
        List[PIL.Image.Image],
        List[np.ndarray],
    ] = None,
    control_image: Union[
        torch.FloatTensor,
        PIL.Image.Image,
        np.ndarray,
        List[torch.FloatTensor],
        List[PIL.Image.Image],
        List[np.ndarray],
    ] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    strength: float = 0.35,
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
    window_size: int = 80,
    lora_model_name_or_path_or_dict: Optional[Union[str, List[str]]] = None,
    lora_alpha: Optional[float] = 0.75,
    text_inversion_model_name_or_path: Optional[
        Union[str, List[str], Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]
    ] = None,
    fast_mode: bool = False,
):

    torch.musa.empty_cache() if MTGPU_DETECTION else torch.cuda.empty_cache()     
    gc.collect()

    tiled_controlnet_img2img_pipeline = StableDiffusionControlNetTileImg2ImgPipeline(
        vae=sd_model.vae,
        text_encoder=sd_model.text_encoder,
        tokenizer=sd_model.tokenizer,
        unet=sd_model.unet.unet,
        controlnet=controlnet_model.controlnet,
        scheduler=sd_model.scheduler,
        safety_checker=sd_model.safety_checker,
        feature_extractor=sd_model.feature_extractor,
        requires_safety_checker=sd_model.requires_safety_checker,
    )
    
    tiled_controlnet_img2img_pipeline.enable_attention_slicing()

    if lora_model_name_or_path_or_dict is not None:
        if not isinstance(lora_model_name_or_path_or_dict, list):
            lora_model_name_or_path_or_dict = [lora_model_name_or_path_or_dict]
        for lora_weight in lora_model_name_or_path_or_dict:
            sd_model.load_lora(lora_weight, alpha=lora_alpha)

    if text_inversion_model_name_or_path is not None:
        if not isinstance(text_inversion_model_name_or_path, list):
            text_inversion_model_name_or_path = [text_inversion_model_name_or_path]
        for text_inversion_weight in text_inversion_model_name_or_path:
            tiled_controlnet_img2img_pipeline.load_textual_inversion(
                text_inversion_weight
            )

    generator = None
    if seed is not None:
        if not isinstance(seed, list):
            seed = [seed]
        generator = [
            torch.Generator(sd_model.device).manual_seed(single_seed)
            for single_seed in seed
        ]

    sd_model.unet.oom = False
    
    results = tiled_controlnet_img2img_pipeline(
        prompt=prompt,
        image=image,
        control_image=control_image,
        height=height,
        width=width,
        strength=strength,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt,
        num_images_per_prompt=num_images_per_prompt,
        eta=eta,
        generator=generator,
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
        window_size=window_size,
    )
    images = results.images
    nsfw_content_detected = results.nsfw_content_detected

    torch.musa.empty_cache() if MTGPU_DETECTION else torch.cuda.empty_cache()    
    gc.collect()
    
    if (
        sd_model.enable_offload_cpu
        and is_accelerate_available()
        and is_accelerate_version(">=", "0.14.0")
    ):
        tiled_controlnet_img2img_pipeline.enable_model_cpu_offload()

    if lora_model_name_or_path_or_dict is not None:
        for lora_weight in lora_model_name_or_path_or_dict:
            sd_model.offload_lora(lora_weight, alpha=lora_alpha)

    return images, nsfw_content_detected
