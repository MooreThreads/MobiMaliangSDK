import torch
from diffusers import StableDiffusionPipeline, ControlNetModel, DDIMScheduler
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
    download_controlnet_from_original_ckpt,
)
from safetensors.torch import load_file
import gc

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)

from modules.tools import anti_oom, samplers

import hijack

hijack.replace_model_cpu_offload_from_cuda_to_musa()
hijack.replace_safety_checker_forward()
hijack.replace_safety_checker_postprocess()
hijack.replace_encode_prompt()
hijack.replace_convert_controlnet_checkpoint()
hijack.replace_from_single_file()

LORA_PREFIX_TEXT_ENCODER = "lora_te"
LORA_PREFIX_UNET = "lora_unet"

try:
    import torch_musa
    MTGPU_DETECTION = True
except:
    MTGPU_DETECTION = False

device = None

if MTGPU_DETECTION: 
    device = "musa"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"


class SDModel:
    def __init__(self, device, requires_safety_checker=False, enable_offload_cpu=False):

        self.device = device
        self.requires_safety_checker = requires_safety_checker
        self.enable_offload_cpu = enable_offload_cpu

        self.scheduler = None
        self.vae = None
        self.text_encoder = None
        self.tokenizer = None
        self.safety_checker = None
        self.feature_extractor = None
        self.unet = None

    def load_models(self, models_dir_or_single_file, diffusers_format=False):

        if diffusers_format:
            pipe = StableDiffusionPipeline.from_pretrained(
                models_dir_or_single_file, torch_dtype=torch.float32
            ).to(self.device)
        elif (
            os.path.isfile(models_dir_or_single_file)
            and models_dir_or_single_file.endswith(".safetensors")
            or models_dir_or_single_file.endswith(".ckpt")
        ):
            config_file_path = os.path.join(os.path.dirname(models_dir_or_single_file), "v1-inference.yaml")
            try:
                pipe = StableDiffusionPipeline.from_single_file(
                    models_dir_or_single_file,
                    torch_dtype=torch.float32,
                    load_safety_checker=False,
                    original_config_file=config_file_path
                ).to(self.device)
            except:
                pipe = StableDiffusionPipeline.from_single_file(
                    models_dir_or_single_file,
                    torch_dtype=torch.float32,
                    load_safety_checker=False,
                    original_config_file=None
                ).to(self.device)
        else:
            raise ValueError(
                "Make sure the model file ends up with .ckpt or .safetensors or a folder contains model files with diffusers flag which is true"
            )
        self.scheduler = DDIMScheduler(
            num_train_timesteps=pipe.scheduler.config.num_train_timesteps,
            beta_start=pipe.scheduler.config.beta_start,
            beta_end=pipe.scheduler.config.beta_end,
            beta_schedule=pipe.scheduler.config.beta_schedule,
            trained_betas=pipe.scheduler.config.trained_betas,
            clip_sample=pipe.scheduler.config.clip_sample,
            set_alpha_to_one=pipe.scheduler.config.set_alpha_to_one,
            steps_offset=pipe.scheduler.config.steps_offset,
            prediction_type=pipe.scheduler.config.prediction_type,
            thresholding=False,  # pipe.scheduler.config.thresholding,
            dynamic_thresholding_ratio=0.995,  # pipe.scheduler.config.dynamic_thresholding_ratio,
            clip_sample_range=1.0,  # pipe.scheduler.config.clip_sample_range,
            timestep_spacing=pipe.scheduler.config.timestep_spacing,
            rescale_betas_zero_snr=False,  # pipe.scheduler.config.rescale_betas_zero_snr,
        )
        self.fast_scheduler = samplers.BiDDIMScheduler(
            num_train_timesteps=pipe.scheduler.config.num_train_timesteps,
            beta_start=pipe.scheduler.config.beta_start,
            beta_end=pipe.scheduler.config.beta_end,
            beta_schedule=pipe.scheduler.config.beta_schedule,
            trained_betas=pipe.scheduler.config.trained_betas,
            clip_sample=pipe.scheduler.config.clip_sample,
            set_alpha_to_one=pipe.scheduler.config.set_alpha_to_one,
            steps_offset=pipe.scheduler.config.steps_offset,
            prediction_type=pipe.scheduler.config.prediction_type,
            thresholding=False,  # pipe.scheduler.config.thresholding,
            dynamic_thresholding_ratio=0.995,  # pipe.scheduler.config.dynamic_thresholding_ratio,
            clip_sample_range=1.0,  # pipe.scheduler.config.clip_sample_range,
            timestep_spacing=pipe.scheduler.config.timestep_spacing,
            rescale_betas_zero_snr=False,  # pipe.scheduler.config.rescale_betas_zero_snr,
        )

        self.vae = anti_oom.AntiOOMAutoencoderKL(pipe.vae)
        self.text_encoder = pipe.text_encoder
        self.tokenizer = pipe.tokenizer
        if self.requires_safety_checker:
            if pipe.safety_checker is None:
                print("There is no safety checker")
            self.safety_checker = pipe.safety_checker
        self.feature_extractor = pipe.feature_extractor
        self.unet = anti_oom.AntiOOMUNet2DConditionModel(pipe.unet)
        self.device = pipe.device

        torch.musa.empty_cache() if MTGPU_DETECTION else torch.cuda.empty_cache() 
        gc.collect()

    def offload_models(self):
        self.scheduler = None
        self.vae = None
        self.text_encoder = None
        self.tokenizer = None
        self.safety_checker = None
        self.feature_extractor = None
        self.unet = None
        
        torch.musa.empty_cache() if MTGPU_DETECTION else torch.cuda.empty_cache() 
        gc.collect()

    def load_lora(
        self,
        pretrained_model_name_or_path,
        LORA_PREFIX_UNET=LORA_PREFIX_UNET,
        LORA_PREFIX_TEXT_ENCODER=LORA_PREFIX_TEXT_ENCODER,
        alpha=0.75,
        is_path=True,
        skip_layers=[],
    ):
        assert self.unet is not None
        assert self.text_encoder is not None
        if is_path:
            state_dict = load_file(pretrained_model_name_or_path)
        else:
            state_dict = pretrained_model_name_or_path
        visited = []

        # directly update weight in diffusers model
        for key in state_dict:
            should_skip = False
            for skip_layer in skip_layers:
                if key.find(skip_layer) == 0:
                    should_skip = True
                    break

            if should_skip:
                continue

            # it is suggested to print out the key, it usually will be something like below
            # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

            # as we have set the alpha beforehand, so just skip
            if ".alpha" in key or key in visited:
                continue

            if "text" in key:
                layer_infos = (
                    key.split(".")[0]
                    .split(LORA_PREFIX_TEXT_ENCODER + "_")[-1]
                    .split("_")
                )
                curr_layer = self.text_encoder

            else:
                layer_infos = (
                    key.split(".")[0].split(LORA_PREFIX_UNET + "_")[-1].split("_")
                )
                
                if isinstance(self.unet, anti_oom.AntiOOMUNet2DConditionModel):
                    curr_layer = self.unet.unet
                else:
                    curr_layer = self.unet

            curr_device, curr_dtype = curr_layer.device, curr_layer.dtype

            # find the target layer
            temp_name = layer_infos.pop(0)
            not_found = False
            while len(layer_infos) > -1:
                try:
                    curr_layer = curr_layer.__getattr__(temp_name)
                    if len(layer_infos) > 0:
                        temp_name = layer_infos.pop(0)
                    elif len(layer_infos) == 0:
                        break
                except Exception:
                    if len(temp_name) > 0:
                        if len(layer_infos) == 0:
                            not_found = True
                            break
                        temp_name += "_" + layer_infos.pop(0)
                    else:
                        temp_name = layer_infos.pop(0)

            if not_found:
                continue

            pair_keys = []
            if "lora_down" in key:
                pair_keys.append(key.replace("lora_down", "lora_up"))
                pair_keys.append(key)
            else:
                pair_keys.append(key)
                pair_keys.append(key.replace("lora_up", "lora_down"))

            # update weight
            if len(state_dict[pair_keys[0]].shape) == 4:
                weight_up = (
                    state_dict[pair_keys[0]]
                    .squeeze(3)
                    .squeeze(2)
                    .to(curr_dtype)
                    .to(curr_device)
                )
                weight_down = (
                    state_dict[pair_keys[1]]
                    .squeeze(3)
                    .squeeze(2)
                    .to(curr_dtype)
                    .to(curr_device)
                )
                curr_layer.weight.data += alpha * torch.mm(
                    weight_up, weight_down
                ).unsqueeze(2).unsqueeze(3)
            else:
                weight_up = state_dict[pair_keys[0]].to(curr_dtype).to(curr_device)
                weight_down = state_dict[pair_keys[1]].to(curr_dtype).to(curr_device)
                curr_layer.weight.data = curr_layer.weight.data + alpha * torch.mm(
                    weight_up, weight_down
                )

            # update visited list
            for item in pair_keys:
                visited.append(item)

    def offload_lora(
        self,
        pretrained_model_name_or_path,
        LORA_PREFIX_UNET=LORA_PREFIX_UNET,
        LORA_PREFIX_TEXT_ENCODER=LORA_PREFIX_TEXT_ENCODER,
        alpha=0.75,
        is_path=True,
        skip_layers=[],
    ):
        assert self.unet is not None
        assert self.text_encoder is not None
        if is_path:
            state_dict = load_file(pretrained_model_name_or_path)
        else:
            state_dict = pretrained_model_name_or_path
        visited = []

        # directly update weight in diffusers model
        for key in state_dict:
            should_skip = False
            for skip_layer in skip_layers:
                if key.find(skip_layer) == 0:
                    should_skip = True
                    break

            if should_skip:
                continue

            # it is suggested to print out the key, it usually will be something like below
            # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

            # as we have set the alpha beforehand, so just skip
            if ".alpha" in key or key in visited:
                continue

            if "text" in key:
                layer_infos = (
                    key.split(".")[0]
                    .split(LORA_PREFIX_TEXT_ENCODER + "_")[-1]
                    .split("_")
                )
                curr_layer = self.text_encoder
            else:
                layer_infos = (
                    key.split(".")[0].split(LORA_PREFIX_UNET + "_")[-1].split("_")
                )
                if isinstance(self.unet, anti_oom.AntiOOMUNet2DConditionModel):
                    curr_layer = self.unet.unet
                else:
                    curr_layer = self.unet

            curr_device, curr_dtype = curr_layer.device, curr_layer.dtype

            # find the target layer
            temp_name = layer_infos.pop(0)
            not_found = False
            while len(layer_infos) > -1:
                try:
                    curr_layer = curr_layer.__getattr__(temp_name)
                    if len(layer_infos) > 0:
                        temp_name = layer_infos.pop(0)
                    elif len(layer_infos) == 0:
                        break
                except Exception:
                    if len(temp_name) > 0:
                        if len(layer_infos) == 0:
                            not_found = True
                            break
                        temp_name += "_" + layer_infos.pop(0)
                    else:
                        temp_name = layer_infos.pop(0)

            if not_found:
                continue

            pair_keys = []
            if "lora_down" in key:
                pair_keys.append(key.replace("lora_down", "lora_up"))
                pair_keys.append(key)
            else:
                pair_keys.append(key)
                pair_keys.append(key.replace("lora_up", "lora_down"))

            # update weight
            if len(state_dict[pair_keys[0]].shape) == 4:
                weight_up = (
                    state_dict[pair_keys[0]]
                    .squeeze(3)
                    .squeeze(2)
                    .to(curr_dtype)
                    .to(curr_device)
                )
                weight_down = (
                    state_dict[pair_keys[1]]
                    .squeeze(3)
                    .squeeze(2)
                    .to(curr_dtype)
                    .to(curr_device)
                )
                curr_layer.weight.data -= alpha * torch.mm(
                    weight_up, weight_down
                ).unsqueeze(2).unsqueeze(3)
            else:
                weight_up = state_dict[pair_keys[0]].to(curr_dtype).to(curr_device)
                weight_down = state_dict[pair_keys[1]].to(curr_dtype).to(curr_device)
                curr_layer.weight.data -= alpha * torch.mm(weight_up, weight_down)

            # update visited list
            for item in pair_keys:
                visited.append(item)


class Controlnet:
    def __init__(self, device):
        self.device = device
        self.controlnet = None

    def load_models(self, models_dir_or_single_file, diffusers_format=False):

        if diffusers_format:
            self.controlnet = ControlNetModel.from_pretrained(
                models_dir_or_single_file, torch_dtype=torch.float32
            ).to(self.device)
        elif models_dir_or_single_file.endswith(".pth"):
            model_name = os.path.splitext(os.path.split(models_dir_or_single_file)[1])[
                0
            ]
            yaml_file_path = os.path.join(
                os.path.dirname(models_dir_or_single_file), model_name + ".yaml"
            )
            assert (
                yaml_file_path
            ), "Please place the yaml file in the corresponding controlnet path"
            self.controlnet = download_controlnet_from_original_ckpt(
                checkpoint_path=models_dir_or_single_file,
                original_config_file=yaml_file_path,
            ).to(self.device)
        else:
            raise ValueError(
                "Make sure the model file ends up with .pth or a folder contains model files with diffusers flag which is not true"
            )
        torch.musa.empty_cache() if MTGPU_DETECTION else torch.cuda.empty_cache() 
        gc.collect()

    def offload_models(self):
        self.controlnet = None
        torch.musa.empty_cache() if MTGPU_DETECTION else torch.cuda.empty_cache() 
        gc.collect()
