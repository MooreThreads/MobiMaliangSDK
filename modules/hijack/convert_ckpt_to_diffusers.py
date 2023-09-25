import diffusers
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
    create_unet_diffusers_config,
    convert_ldm_unet_checkpoint,
)
from diffusers import ControlNetModel


def replace_convert_controlnet_checkpoint():
    def convert_controlnet_checkpoint(
        checkpoint,
        original_config,
        checkpoint_path,
        image_size,
        upcast_attention,
        extract_ema,
        use_linear_projection=None,
        cross_attention_dim=None,
    ):
        ctrlnet_config = create_unet_diffusers_config(
            original_config, image_size=image_size, controlnet=True
        )
        ctrlnet_config["upcast_attention"] = upcast_attention

        ctrlnet_config.pop("sample_size")

        if use_linear_projection is not None:
            ctrlnet_config["use_linear_projection"] = use_linear_projection

        if cross_attention_dim is not None:
            ctrlnet_config["cross_attention_dim"] = cross_attention_dim

        # Some controlnet ckpt files are distributed independently from the rest of the
        # model components i.e. https://huggingface.co/thibaud/controlnet-sd21/
        if "time_embed.0.weight" in checkpoint:
            skip_extract_state_dict = True
        else:
            skip_extract_state_dict = False

        converted_ctrl_checkpoint = convert_ldm_unet_checkpoint(
            checkpoint,
            ctrlnet_config,
            path=checkpoint_path,
            extract_ema=extract_ema,
            controlnet=True,
            skip_extract_state_dict=skip_extract_state_dict,
        )
        ctrlnet_config.pop("addition_embed_type")
        ctrlnet_config.pop("addition_time_embed_dim")
        ctrlnet_config.pop("transformer_layers_per_block")
        controlnet_model = ControlNetModel(**ctrlnet_config)

        controlnet_model.load_state_dict(converted_ctrl_checkpoint)

        return controlnet_model

    diffusers.pipelines.stable_diffusion.convert_from_ckpt.convert_controlnet_checkpoint = (
        convert_controlnet_checkpoint
    )
