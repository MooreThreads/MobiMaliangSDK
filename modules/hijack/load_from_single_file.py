import diffusers
from pathlib import Path
from diffusers.utils import (
    DIFFUSERS_CACHE,
    HF_HUB_OFFLINE,
    is_safetensors_available,
)

from huggingface_hub import hf_hub_download
def replace_from_single_file():
    @classmethod
    def from_single_file(cls, pretrained_model_link_or_path, **kwargs):
        r"""
        Instantiate a [`DiffusionPipeline`] from pretrained pipeline weights saved in the `.ckpt` format. The pipeline
        is set in evaluation mode (`model.eval()`) by default.

        Parameters:
            pretrained_model_link_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:
                    - A link to the `.ckpt` file (for example
                      `"https://huggingface.co/<repo_id>/blob/main/<path_to_file>.ckpt"`) on the Hub.
                    - A path to a *file* containing all pipeline weights.
            torch_dtype (`str` or `torch.dtype`, *optional*):
                Override the default `torch.dtype` and load the model with another dtype. If `"auto"` is passed, the
                dtype is automatically derived from the model's weights.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to resume downloading the model weights and configuration files. If set to `False`, any
                incompletely downloaded files are deleted.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to True, the model
                won't be downloaded from the Hub.
            use_auth_token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                `diffusers-cli login` (stored in `~/.huggingface`) is used.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.
            use_safetensors (`bool`, *optional*, defaults to `None`):
                If set to `None`, the safetensors weights are downloaded if they're available **and** if the
                safetensors library is installed. If set to `True`, the model is forcibly loaded from safetensors
                weights. If set to `False`, safetensors weights are not loaded.
            extract_ema (`bool`, *optional*, defaults to `False`):
                Whether to extract the EMA weights or not. Pass `True` to extract the EMA weights which usually yield
                higher quality images for inference. Non-EMA weights are usually better to continue finetuning.
            upcast_attention (`bool`, *optional*, defaults to `None`):
                Whether the attention computation should always be upcasted.
            image_size (`int`, *optional*, defaults to 512):
                The image size the model was trained on. Use 512 for all Stable Diffusion v1 models and the Stable
                Diffusion v2 base model. Use 768 for Stable Diffusion v2.
            prediction_type (`str`, *optional*):
                The prediction type the model was trained on. Use `'epsilon'` for all Stable Diffusion v1 models and
                the Stable Diffusion v2 base model. Use `'v_prediction'` for Stable Diffusion v2.
            num_in_channels (`int`, *optional*, defaults to `None`):
                The number of input channels. If `None`, it will be automatically inferred.
            scheduler_type (`str`, *optional*, defaults to `"pndm"`):
                Type of scheduler to use. Should be one of `["pndm", "lms", "heun", "euler", "euler-ancestral", "dpm",
                "ddim"]`.
            load_safety_checker (`bool`, *optional*, defaults to `True`):
                Whether to load the safety checker or not.
            text_encoder (`CLIPTextModel`, *optional*, defaults to `None`):
                An instance of
                [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel) to use,
                specifically the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)
                variant. If this parameter is `None`, the function will load a new instance of [CLIP] by itself, if
                needed.
            tokenizer (`CLIPTokenizer`, *optional*, defaults to `None`):
                An instance of
                [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer)
                to use. If this parameter is `None`, the function will load a new instance of [CLIPTokenizer] by
                itself, if needed.
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to overwrite load and saveable variables (for example the pipeline components of the
                specific pipeline class). The overwritten components are directly passed to the pipelines `__init__`
                method. See example below for more information.

        Examples:

        ```py
        >>> from diffusers import StableDiffusionPipeline

        >>> # Download pipeline from huggingface.co and cache.
        >>> pipeline = StableDiffusionPipeline.from_single_file(
        ...     "https://huggingface.co/WarriorMama777/OrangeMixs/blob/main/Models/AbyssOrangeMix/AbyssOrangeMix.safetensors"
        ... )

        >>> # Download pipeline from local file
        >>> # file is downloaded under ./v1-5-pruned-emaonly.ckpt
        >>> pipeline = StableDiffusionPipeline.from_single_file("./v1-5-pruned-emaonly")

        >>> # Enable float16 and move to GPU
        >>> pipeline = StableDiffusionPipeline.from_single_file(
        ...     "https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.ckpt",
        ...     torch_dtype=torch.float16,
        ... )
        >>> pipeline.to("cuda")
        ```
        """
        # import here to avoid circular dependency
        from diffusers.pipelines.stable_diffusion.convert_from_ckpt import download_from_original_stable_diffusion_ckpt

        cache_dir = kwargs.pop("cache_dir", DIFFUSERS_CACHE)
        original_config_file = kwargs.pop("original_config_file", None)
        resume_download = kwargs.pop("resume_download", False)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", HF_HUB_OFFLINE)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        extract_ema = kwargs.pop("extract_ema", False)
        image_size = kwargs.pop("image_size", None)
        scheduler_type = kwargs.pop("scheduler_type", "pndm")
        num_in_channels = kwargs.pop("num_in_channels", None)
        upcast_attention = kwargs.pop("upcast_attention", None)
        load_safety_checker = kwargs.pop("load_safety_checker", True)
        prediction_type = kwargs.pop("prediction_type", None)
        text_encoder = kwargs.pop("text_encoder", None)
        tokenizer = kwargs.pop("tokenizer", None)

        torch_dtype = kwargs.pop("torch_dtype", None)

        use_safetensors = kwargs.pop("use_safetensors", None if is_safetensors_available() else False)

        pipeline_name = cls.__name__
        file_extension = pretrained_model_link_or_path.rsplit(".", 1)[-1]
        from_safetensors = file_extension == "safetensors"

        if from_safetensors and use_safetensors is False:
            raise ValueError("Make sure to install `safetensors` with `pip install safetensors`.")

        # TODO: For now we only support stable diffusion
        stable_unclip = None
        model_type = None
        controlnet = False

        if pipeline_name == "StableDiffusionControlNetPipeline":
            # Model type will be inferred from the checkpoint.
            controlnet = True
        elif "StableDiffusion" in pipeline_name:
            # Model type will be inferred from the checkpoint.
            pass
        elif pipeline_name == "StableUnCLIPPipeline":
            model_type = "FrozenOpenCLIPEmbedder"
            stable_unclip = "txt2img"
        elif pipeline_name == "StableUnCLIPImg2ImgPipeline":
            model_type = "FrozenOpenCLIPEmbedder"
            stable_unclip = "img2img"
        elif pipeline_name == "PaintByExamplePipeline":
            model_type = "PaintByExample"
        elif pipeline_name == "LDMTextToImagePipeline":
            model_type = "LDMTextToImage"
        else:
            raise ValueError(f"Unhandled pipeline class: {pipeline_name}")

        # remove huggingface url
        for prefix in ["https://huggingface.co/", "huggingface.co/", "hf.co/", "https://hf.co/"]:
            if pretrained_model_link_or_path.startswith(prefix):
                pretrained_model_link_or_path = pretrained_model_link_or_path[len(prefix) :]

        # Code based on diffusers.pipelines.pipeline_utils.DiffusionPipeline.from_pretrained
        ckpt_path = Path(pretrained_model_link_or_path)
        if not ckpt_path.is_file():
            # get repo_id and (potentially nested) file path of ckpt in repo
            repo_id = "/".join(ckpt_path.parts[:2])
            file_path = "/".join(ckpt_path.parts[2:])

            if file_path.startswith("blob/"):
                file_path = file_path[len("blob/") :]

            if file_path.startswith("main/"):
                file_path = file_path[len("main/") :]

            pretrained_model_link_or_path = hf_hub_download(
                repo_id,
                filename=file_path,
                cache_dir=cache_dir,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                revision=revision,
                force_download=force_download,
            )
        pipe = download_from_original_stable_diffusion_ckpt(
            pretrained_model_link_or_path,
            original_config_file=original_config_file,
            pipeline_class=cls,
            model_type=model_type,
            stable_unclip=stable_unclip,
            controlnet=controlnet,
            from_safetensors=from_safetensors,
            extract_ema=extract_ema,
            image_size=image_size,
            scheduler_type=scheduler_type,
            num_in_channels=num_in_channels,
            upcast_attention=upcast_attention,
            load_safety_checker=load_safety_checker,
            prediction_type=prediction_type,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )

        if torch_dtype is not None:
            pipe.to(torch_dtype=torch_dtype)

        return pipe
    diffusers.loaders.FromSingleFileMixin.from_single_file = from_single_file

    
