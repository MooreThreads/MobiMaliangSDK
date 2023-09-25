import torch
from diffusers import StableDiffusionPipeline
from diffusers.utils import is_accelerate_available, is_accelerate_version
from typing import Any, Callable, Dict, List, Optional, Union
import gc
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(__dir__))
from shared import SDModel, MTGPU_DETECTION


def txt2img(
    sd_model: SDModel,
    prompt: Union[str, List[str]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
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
    guidance_rescale: float = 0.0,
    lora_model_name_or_path_or_dict: Optional[Union[str, List[str]]] = None,
    lora_alpha: Optional[float] = 0.75,
    text_inversion_model_name_or_path: Optional[
        Union[str, List[str], Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]
    ] = None,
    fast_mode: bool = False,
):

    torch.musa.empty_cache() if MTGPU_DETECTION else torch.cuda.empty_cache()    
    gc.collect()

    txt2img_pipeline = StableDiffusionPipeline(
        vae=sd_model.vae,
        text_encoder=sd_model.text_encoder,
        tokenizer=sd_model.tokenizer,
        unet=sd_model.unet,
        scheduler=sd_model.fast_scheduler if fast_mode else sd_model.scheduler,
        safety_checker=sd_model.safety_checker,
        feature_extractor=sd_model.feature_extractor,
        requires_safety_checker=sd_model.requires_safety_checker,
    )
    
    txt2img_pipeline.unet.oom = False

    if lora_model_name_or_path_or_dict is not None:
        if not isinstance(lora_model_name_or_path_or_dict, list):
            lora_model_name_or_path_or_dict = [lora_model_name_or_path_or_dict]
        for lora_weight in lora_model_name_or_path_or_dict:
            sd_model.load_lora(lora_weight, alpha=lora_alpha)

    if text_inversion_model_name_or_path is not None:
        if not isinstance(text_inversion_model_name_or_path, list):
            text_inversion_model_name_or_path = [text_inversion_model_name_or_path]
        for text_inversion_weight in text_inversion_model_name_or_path:
            txt2img_pipeline.load_textual_inversion(text_inversion_weight)

    generator = None
    if seed is not None:
        if not isinstance(seed, list):
            seed = [seed]
        generator = [
            torch.Generator(sd_model.device).manual_seed(single_seed)
            for single_seed in seed
        ]

    sd_model.unet.oom = False
    
    results = txt2img_pipeline(
        prompt=prompt,
        height=height,
        width=width,
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
        guidance_rescale=guidance_rescale,
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
        txt2img_pipeline.enable_model_cpu_offload()

    if lora_model_name_or_path_or_dict is not None:
        for lora_weight in lora_model_name_or_path_or_dict:
            sd_model.offload_lora(lora_weight, alpha=lora_alpha)

    return images, nsfw_content_detected
