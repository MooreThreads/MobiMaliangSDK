import torch
from diffusers import StableDiffusionControlNetImg2ImgPipeline
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from diffusers.models import AutoencoderKL, ControlNetModel, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import is_compiled_module, logging, replace_example_docstring

from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput


from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import PIL
import numpy as np
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__dir__)))
from hijack import (
    replace_encode_prompt,
    replace_sequential_cpu_offload_from_cuda_to_musa,
)

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> # !pip install opencv-python transformers accelerate
        >>> from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler
        >>> from diffusers.utils import load_image
        >>> import numpy as np
        >>> import torch

        >>> import cv2
        >>> from PIL import Image

        >>> # download an image
        >>> image = load_image(
        ...     "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
        ... )
        >>> input_image = image.convert("RGB")
        >>> w, h = input_image.size
        >>> W, H = int(w * up_sampling_ratio), int(h * up_sampling_ratio)
        >>> H = int(round(H / 8.0)) * 8
        >>> W = int(round(W / 8.0)) * 8
        >>> control_image = input_image.resize((W, H), resample=Image.LANCZOS)

        >>> # load control net and stable diffusion v1-5
        >>> controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-tile", torch_dtype=torch.float32)
        >>> pipe = StableDiffusionControlNetTileImg2ImgPipeline.from_pretrained(
        ...     "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float32
        ... )

        >>> # speed up diffusion process with faster scheduler and memory optimization
        >>> pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

        >>> # generate image
        >>> generator = torch.manual_seed(0)
        >>> image = pipe(
        ...     "best quarlity, masterpiece",
        ...     num_inference_steps=20,
        ...     generator=generator,
        ...     image=image,
        ...     control_image=control_image,
        ... ).images[0]
        ```
"""


class StableDiffusionControlNetTileImg2ImgPipeline(
    StableDiffusionControlNetImg2ImgPipeline
):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        controlnet: Union[
            ControlNetModel,
            List[ControlNetModel],
            Tuple[ControlNetModel],
            MultiControlNetModel,
        ],
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        requires_safety_checker: bool = True,
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            requires_safety_checker=requires_safety_checker,
        )

        # self._encode_prompt = replace_encode_prompt._encode_prompt
        # self.enable_sequential_cpu_offload = replace_sequential_cpu_offload_from_cuda_to_musa.enable_sequential_cpu_offload

    def get_views(self, height, width, window_size=64):
        # Here, we define the mappings F_i (see Eq. 7 in the MultiDiffusion paper https://arxiv.org/abs/2302.08113)
        window_size = int(window_size // 8 * 8)
        stride = window_size - 2
        if height is None or width is None:
            height = self.image_height
            width = self.image_width

        window_height_size = int(min(height / 8, window_size))
        window_width_size = int(min(width / 8, window_size))
        panorama_height = height / 8
        panorama_width = width / 8
        if panorama_height % stride == 0:
            num_blocks_height = int(max(1, panorama_height // stride))
        else:
            num_blocks_height = int(panorama_height // stride + 1)
        if panorama_width % stride == 0:
            num_blocks_width = int(max(1, panorama_width // stride))
        else:
            num_blocks_width = int(panorama_width // stride + 1)
        total_num_blocks = int(num_blocks_height * num_blocks_width)
        views = []
        for i in range(total_num_blocks):
            h_start = int((i // num_blocks_width) * stride)
            h_end = h_start + window_height_size
            if h_end > panorama_height:
                h_start = int(panorama_height - stride)
                h_end = int(panorama_height)
            w_start = int((i % num_blocks_width) * stride)
            w_end = w_start + window_width_size
            if w_end > panorama_width:
                w_start = int(panorama_width - stride)
                w_end = int(panorama_width)
            views.append((h_start, h_end, w_start, w_end))
        return views

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
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
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
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
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            image (`torch.FloatTensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.FloatTensor]`, `List[PIL.Image.Image]`, `List[np.ndarray]`,:
                    `List[List[torch.FloatTensor]]`, `List[List[np.ndarray]]` or `List[List[PIL.Image.Image]]`):
                The initial image will be used as the starting point for the image generation process. Can also accpet
                image latents as `image`, if passing latents directly, it will not be encoded again.
            control_image (`torch.FloatTensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.FloatTensor]`, `List[PIL.Image.Image]`, `List[np.ndarray]`,:
                    `List[List[torch.FloatTensor]]`, `List[List[np.ndarray]]` or `List[List[PIL.Image.Image]]`):
                The ControlNet input condition. ControlNet uses this input condition to generate guidance to Unet. If
                the type is specified as `Torch.FloatTensor`, it is passed to ControlNet as is. `PIL.Image.Image` can
                also be accepted as an image. The dimensions of the output image defaults to `image`'s dimensions. If
                height and/or width are passed, `image` is resized according to them. If multiple ControlNets are
                specified in init, images must be passed as a list such that each element of the list can be correctly
                batched for input to a single controlnet.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
            controlnet_conditioning_scale (`float` or `List[float]`, *optional*, defaults to 1.0):
                The outputs of the controlnet are multiplied by `controlnet_conditioning_scale` before they are added
                to the residual in the original unet. If multiple ControlNets are specified in init, you can set the
                corresponding scale as a list. Note that by default, we use a smaller conditioning scale for inpainting
                than for [`~StableDiffusionControlNetPipeline.__call__`].
            guess_mode (`bool`, *optional*, defaults to `False`):
                In this mode, the ControlNet encoder will try best to recognize the content of the input image even if
                you remove all prompts. The `guidance_scale` between 3.0 and 5.0 is recommended.
            control_guidance_start (`float` or `List[float]`, *optional*, defaults to 0.0):
                The percentage of total steps at which the controlnet starts applying.
            control_guidance_end (`float` or `List[float]`, *optional*, defaults to 1.0):
                The percentage of total steps at which the controlnet stops applying.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        controlnet = (
            self.controlnet._orig_mod
            if is_compiled_module(self.controlnet)
            else self.controlnet
        )

        # align format for control guidance
        if not isinstance(control_guidance_start, list) and isinstance(
            control_guidance_end, list
        ):
            control_guidance_start = len(control_guidance_end) * [
                control_guidance_start
            ]
        elif not isinstance(control_guidance_end, list) and isinstance(
            control_guidance_start, list
        ):
            control_guidance_end = len(control_guidance_start) * [control_guidance_end]
        elif not isinstance(control_guidance_start, list) and not isinstance(
            control_guidance_end, list
        ):
            mult = (
                len(controlnet.nets)
                if isinstance(controlnet, MultiControlNetModel)
                else 1
            )
            control_guidance_start, control_guidance_end = (
                mult * [control_guidance_start],
                mult * [control_guidance_end],
            )

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            control_image,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            controlnet_conditioning_scale,
            control_guidance_start,
            control_guidance_end,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        controlnet = (
            self.controlnet._orig_mod
            if is_compiled_module(self.controlnet)
            else self.controlnet
        )

        if isinstance(controlnet, MultiControlNetModel) and isinstance(
            controlnet_conditioning_scale, float
        ):
            controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(
                controlnet.nets
            )

        global_pool_conditions = (
            controlnet.config.global_pool_conditions
            if isinstance(controlnet, ControlNetModel)
            else controlnet.nets[0].config.global_pool_conditions
        )
        guess_mode = guess_mode or global_pool_conditions

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None)
            if cross_attention_kwargs is not None
            else None
        )
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )
        # 4. Prepare image
        image = self.image_processor.preprocess(image).to(dtype=torch.float32)

        # 5. Prepare controlnet_conditioning_image
        if isinstance(controlnet, ControlNetModel):
            control_image = self.prepare_control_image(
                image=control_image,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=controlnet.dtype,
                do_classifier_free_guidance=do_classifier_free_guidance,
                guess_mode=guess_mode,
            )
        elif isinstance(controlnet, MultiControlNetModel):
            control_images = []

            for control_image_ in control_image:
                control_image_ = self.prepare_control_image(
                    image=control_image_,
                    width=width,
                    height=height,
                    batch_size=batch_size * num_images_per_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    device=device,
                    dtype=controlnet.dtype,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    guess_mode=guess_mode,
                )

                control_images.append(control_image_)

            control_image = control_images
        else:
            assert False

        # 5. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.get_timesteps(
            num_inference_steps, strength, device
        )
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        # 6. Prepare latent variables
        latents = self.prepare_latents(
            image,
            latent_timestep,
            batch_size,
            num_images_per_prompt,
            prompt_embeds.dtype,
            device,
            generator,
        )

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7.1 Create tensor stating which controlnets to keep
        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(keeps[0] if len(keeps) == 1 else keeps)

        self.image_height = control_image.shape[2]
        self.image_width = control_image.shape[3]

        views = self.get_views(height, width, window_size=window_size)
        count = torch.zeros_like(latents)
        value = torch.zeros_like(latents)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                count.zero_()
                value.zero_()
                # generate views
                # Here, we iterate through different spatial crops of the latents and denoise them. These
                # denoised (latent) crops are then averaged to produce the final latent
                # for the current timestep via MultiDiffusion. Please see Sec. 4.1 in the
                # MultiDiffusion paper for more details: https://arxiv.org/abs/2302.08113
                for h_start, h_end, w_start, w_end in views:
                    # get the latents corresponding to the current view coordinates
                    latents_for_view = latents[:, :, h_start:h_end, w_start:w_end]
                    controlnet_conditioning_image_for_view = control_image[
                        :, :, h_start * 8 : h_end * 8, w_start * 8 : w_end * 8
                    ]
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = (
                        torch.cat([latents_for_view] * 2)
                        if do_classifier_free_guidance
                        else latents_for_view
                    )
                    latent_model_input = self.scheduler.scale_model_input(
                        latent_model_input, t
                    )

                    # controlnet(s) inference
                    if guess_mode and do_classifier_free_guidance:
                        # Infer ControlNet only for the conditional batch.
                        control_model_input = latents
                        control_model_input = self.scheduler.scale_model_input(
                            control_model_input, t
                        )
                        controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
                    else:
                        control_model_input = latent_model_input
                        controlnet_prompt_embeds = prompt_embeds

                    if isinstance(controlnet_keep[i], list):
                        cond_scale = [
                            c * s
                            for c, s in zip(
                                controlnet_conditioning_scale, controlnet_keep[i]
                            )
                        ]
                    else:
                        cond_scale = controlnet_conditioning_scale * controlnet_keep[i]
                    # compute the percentage of total steps we are at
                    current_sampling_percent = i / len(timesteps)

                    if (
                        current_sampling_percent < control_guidance_start[0]
                        or current_sampling_percent > control_guidance_end[0]
                    ):
                        # do not apply the controlnet
                        down_block_res_samples = None
                        mid_block_res_sample = None
                    else:
                        down_block_res_samples, mid_block_res_sample = self.controlnet(
                            control_model_input,
                            t,
                            encoder_hidden_states=controlnet_prompt_embeds,
                            controlnet_cond=controlnet_conditioning_image_for_view,
                            conditioning_scale=cond_scale,
                            guess_mode=guess_mode,
                            return_dict=False,
                        )

                        if guess_mode and do_classifier_free_guidance:
                            # Infered ControlNet only for the conditional batch.
                            # To apply the output of ControlNet to both the unconditional and conditional batches,
                            # add 0 to the unconditional batch to keep it unchanged.
                            down_block_res_samples = [
                                torch.cat([torch.zeros_like(d), d])
                                for d in down_block_res_samples
                            ]
                            mid_block_res_sample = torch.cat(
                                [
                                    torch.zeros_like(mid_block_res_sample),
                                    mid_block_res_sample,
                                ]
                            )

                    # predict the noise residual
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                        return_dict=False,
                    )[0]

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (
                            noise_pred_text - noise_pred_uncond
                        )

                    # compute the previous noisy sample x_t -> x_t-1
                    latents_view_denoised = self.scheduler.step(
                        noise_pred,
                        t,
                        latents_for_view,
                        **extra_step_kwargs,
                        return_dict=False
                    )[0]

                    value[:, :, h_start:h_end, w_start:w_end] += latents_view_denoised
                    count[:, :, h_start:h_end, w_start:w_end] += 1

                # take the MultiDiffusion step. Eq. 5 in MultiDiffusion paper: https://arxiv.org/abs/2302.08113
                latents = torch.where(count > 0, value / count, value)

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # If we do sequential model offloading, let's offload unet and controlnet
        # manually for max memory savings
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.unet.to("cpu")
            self.controlnet.to("cpu")
            try:
                torch.musa.empty_cache()
            except:
                torch.cuda.empty_cache()

        if not output_type == "latent":
            image = self.vae.decode(
                latents / self.vae.config.scaling_factor, return_dict=False
            )[0]
            image, has_nsfw_concept = self.run_safety_checker(
                image, device, prompt_embeds.dtype
            )
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(
            image, output_type=output_type, do_denormalize=do_denormalize
        )

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(
            images=image, nsfw_content_detected=has_nsfw_concept
        )
