import torch
import diffusers
from diffusers.utils import is_accelerate_available, is_accelerate_version

try:    
    import torch_musa
    MTGPU_DETECTION = True
except:
    print("We cannot import torch_musa")
    MTGPU_DETECTION = False
        

def replace_sequential_cpu_offload_from_cuda_to_musa():
    def enable_sequential_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, unet,
        text_encoder, vae and safety checker have their state dicts saved to CPU and then are moved to a
        `torch.device('meta') and loaded to GPU only when their specific submodule has its `forward` method called.
        Note that offloading happens on a submodule basis. Memory savings are higher than with
        `enable_model_cpu_offload`, but performance is lower.
        """
        if is_accelerate_available() and is_accelerate_version(">=", "0.14.0"):
            from accelerate import cpu_offload
        else:
            raise ImportError(
                "`enable_sequential_cpu_offload` requires `accelerate v0.14.0` or higher"
            )
        if MTGPU_DETECTION:
            device = torch.device(f"musa:{gpu_id}")
        else:
            device = torch.device(f"cuda:{gpu_id}") if torch.cuda.is_available() else torch.device("cpu")

        if self.device.type != "cpu":
            self.to("cpu", silence_dtype_warnings=True)
            torch.musa.empty_cache() if MTGPU_DETECTION else torch.cuda.empty_cache()  # otherwise we don't see the memory savings (but they probably exist)

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            cpu_offload(cpu_offloaded_model, device)

        if self.safety_checker is not None:
            cpu_offload(
                self.safety_checker, execution_device=device, offload_buffers=True
            )

    diffusers.StableDiffusionPipeline.enable_sequential_cpu_offload = (
        enable_sequential_cpu_offload
    )
    diffusers.StableDiffusionImg2ImgPipeline.enable_sequential_cpu_offload = (
        enable_sequential_cpu_offload
    )
    diffusers.StableDiffusionControlNetPipeline.enable_sequential_cpu_offload = (
        enable_sequential_cpu_offload
    )
    diffusers.StableDiffusionControlNetImg2ImgPipeline.enable_sequential_cpu_offload = (
        enable_sequential_cpu_offload
    )
    diffusers.StableDiffusionControlNetInpaintPipeline.enable_sequential_cpu_offload = (
        enable_sequential_cpu_offload
    )


def replace_model_cpu_offload_from_cuda_to_musa():
    def enable_model_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, reducing memory usage with a low impact on performance. Compared
        to `enable_sequential_cpu_offload`, this method moves one whole model at a time to the GPU when its `forward`
        method is called, and the model remains in GPU until the next model runs. Memory savings are lower than with
        `enable_sequential_cpu_offload`, but performance is much better due to the iterative execution of the `unet`.
        """
        if is_accelerate_available() and is_accelerate_version(">=", "0.17.0.dev0"):
            from accelerate import cpu_offload_with_hook
        else:
            raise ImportError(
                "`enable_model_cpu_offload` requires `accelerate v0.17.0` or higher."
            )

        device = torch.device(f"musa:{gpu_id}")

        if self.device.type != "cpu":
            self.to("cpu", silence_dtype_warnings=True)
            torch.musa.empty_cache()  # otherwise we don't see the memory savings (but they probably exist)

        hook = None
        for cpu_offloaded_model in [self.text_encoder, self.unet, self.vae]:
            _, hook = cpu_offload_with_hook(
                cpu_offloaded_model, device, prev_module_hook=hook
            )

        if self.safety_checker is not None:
            _, hook = cpu_offload_with_hook(
                self.safety_checker, device, prev_module_hook=hook
            )

        # We'll offload the last model manually.
        self.final_offload_hook = hook

    diffusers.StableDiffusionPipeline.enable_model_cpu_offload = (
        enable_model_cpu_offload
    )
    diffusers.StableDiffusionImg2ImgPipeline.enable_model_cpu_offload = (
        enable_model_cpu_offload
    )
    diffusers.StableDiffusionControlNetPipeline.enable_model_cpu_offload = (
        enable_model_cpu_offload
    )
    diffusers.StableDiffusionControlNetImg2ImgPipeline.enable_model_cpu_offload = (
        enable_model_cpu_offload
    )
    diffusers.StableDiffusionControlNetInpaintPipeline.enable_model_cpu_offload = (
        enable_model_cpu_offload
    )
