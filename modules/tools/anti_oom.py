import torch
from diffusers.models import AutoencoderKL, UNet2DConditionModel
import inspect
from typing import Any, Dict, List, Optional, Tuple, Union
import gc
import os
import sys


try:
    import torch_musa
    MTGPU_DETECTION = True
except:
    MTGPU_DETECTION = False


class AntiOOMUNet2DConditionModel(torch.nn.Module):
    def __init__(self, unet: UNet2DConditionModel):
        super(AntiOOMUNet2DConditionModel, self).__init__()
        self.unet = unet
        self.oom = False
        self.config = unet.config
        self.down_blocks = unet.down_blocks
        self.mid_block = unet.mid_block
        self.up_blocks = unet.up_blocks
        self.device = unet.device
        self.dtype = unet.dtype

    def modules(self):
        return self.unet.modules()

    def parameters(self, recurse: bool = True):
        return self.unet.parameters(recurse=recurse)

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ):
        return self.unet.named_parameters(
            prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate
        )
        
    def set_attention_slice(
            self, 
            slice_size: Optional[Union[str, int]] = "auto"
        ):
        
        module_names, _ = self._get_signature_keys(self.unet)
        modules = [getattr(self, n, None) for n in module_names]
        modules = [m for m in modules if isinstance(m, torch.nn.Module) and hasattr(m, "set_attention_slice")]

        for module in modules:
            module.set_attention_slice(slice_size)
    
    def _get_signature_keys(self, obj):
        parameters = inspect.signature(obj.__init__).parameters
        required_parameters = {k: v for k, v in parameters.items() if v.default == inspect._empty}
        optional_parameters = set({k for k, v in parameters.items() if v.default != inspect._empty})
        expected_modules = set(required_parameters.keys()) - {"self"}
        return expected_modules, optional_parameters

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        if not self.oom:
            try:
                return self.unet(
                    sample=sample,
                    timestep=timestep,
                    encoder_hidden_states=encoder_hidden_states,
                    class_labels=class_labels,
                    timestep_cond=timestep_cond,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    down_block_additional_residuals=down_block_additional_residuals,
                    mid_block_additional_residual=mid_block_additional_residual,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=return_dict,
                )
            except Exception as err:
                if "MUSA out of memory. " in str(err) or "CUDA out of memory. " in str(err):
                    torch.musa.empty_cache() if MTGPU_DETECTION else torch.cuda.empty_cache() 
                    gc.collect()
                    self.oom = True
                    # print("We need split the batch into halfs against \"out of memory\"")
                    # self.set_attention_slice()
                    bs = sample.shape[0]
                    half_bs = bs // 2
                    down_block_additional_residuals_half1 = None
                    down_block_additional_residuals_half2 = None
                    if down_block_additional_residuals is not None:
                        down_block_additional_residuals_half1 = [item[:half_bs] for item in down_block_additional_residuals]
                        down_block_additional_residuals_half2 = [item[half_bs:] for item in down_block_additional_residuals]
                    half1 = self.unet(
                        sample=sample[:half_bs],
                        timestep=timestep,
                        encoder_hidden_states=encoder_hidden_states[:half_bs],
                        class_labels=class_labels,
                        timestep_cond=timestep_cond,
                        attention_mask=attention_mask,
                        cross_attention_kwargs=cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                        down_block_additional_residuals=down_block_additional_residuals_half1,
                        mid_block_additional_residual=mid_block_additional_residual[:half_bs] if mid_block_additional_residual is not None else None,
                        encoder_attention_mask=encoder_attention_mask,
                        return_dict=return_dict,
                    )

                    half2 = self.unet(
                        sample=sample[half_bs:],
                        timestep=timestep,
                        encoder_hidden_states=encoder_hidden_states[half_bs:],
                        class_labels=class_labels,
                        timestep_cond=timestep_cond,
                        attention_mask=attention_mask,
                        cross_attention_kwargs=cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                        down_block_additional_residuals=down_block_additional_residuals_half2,
                        mid_block_additional_residual=mid_block_additional_residual[half_bs:] if mid_block_additional_residual is not None else None,
                        encoder_attention_mask=encoder_attention_mask,
                        return_dict=return_dict,
                    )
                    return (torch.cat([half1[0], half2[0]]),)
        else:
            bs = sample.shape[0]
            half_bs = bs // 2
            
            down_block_additional_residuals_half1 = None
            down_block_additional_residuals_half2 = None
            if down_block_additional_residuals is not None:
                down_block_additional_residuals_half1 = [item[:half_bs] for item in down_block_additional_residuals]
                down_block_additional_residuals_half2 = [item[half_bs:] for item in down_block_additional_residuals]
            half1 = self.unet(
                sample=sample[:half_bs],
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states[:half_bs],
                class_labels=class_labels,
                timestep_cond=timestep_cond,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                down_block_additional_residuals=down_block_additional_residuals_half1,
                mid_block_additional_residual=mid_block_additional_residual[:half_bs] if mid_block_additional_residual is not None else None,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=return_dict,
            )

            half2 = self.unet(
                sample=sample[half_bs:],
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states[half_bs:],
                class_labels=class_labels,
                timestep_cond=timestep_cond,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                down_block_additional_residuals=down_block_additional_residuals_half2,
                mid_block_additional_residual=mid_block_additional_residual[half_bs:] if mid_block_additional_residual is not None else None,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=return_dict,
            )

            return (torch.cat([half1[0], half2[0]]),)

class AntiOOMAutoencoderKL(torch.nn.Module):
    def __init__(self, vae: AutoencoderKL):
        super(AntiOOMAutoencoderKL, self).__init__()
        self.vae = vae
        self.config = vae.config
        self.device = vae.device
        self.dtype = vae.dtype

    def modules(self):
        return self.vae.modules()

    def parameters(self, recurse: bool = True):
        return self.vae.parameters(recurse=recurse)

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ):
        return self.vae.named_parameters(
            prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate
        )
        
    def encode(
            self, 
            x: torch.FloatTensor, 
            return_dict: bool = True
        ):
        try:
            return self.vae.encode(
                x=x,
                return_dict=return_dict
            )
        except Exception as err:
            if "MUSA out of memory. " in str(err) or "CUDA out of memory. " in str(err):
                torch.musa.empty_cache() if MTGPU_DETECTION else torch.cuda.empty_cache() 
                gc.collect()
                # print("We need enable vae tiling against \"out of memory\"")
                self.vae.enable_tiling()
                result = self.vae.encode(
                    x=x,
                    return_dict=return_dict
                )
                
                return result
    
    def decode(
        self,
        z: torch.FloatTensor, 
        return_dict: bool = True
    ):
        try:
            return self.vae.decode(
                z=z,
                return_dict=return_dict
            )
        except Exception as err:
            if "MUSA out of memory. " in str(err) or "CUDA out of memory. " in str(err):
                torch.musa.empty_cache() if MTGPU_DETECTION else torch.cuda.empty_cache() 
                gc.collect()
                # print("We need enable vae tiling against \"out of memory\"")
                self.vae.enable_tiling()
                result = self.vae.decode(
                    z=z,
                    return_dict=return_dict
                )
                self.vae.disable_tiling()
                
                return result