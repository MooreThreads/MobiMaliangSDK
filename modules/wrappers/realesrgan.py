import torch
import os
import sys
import gc
import re
from PIL import Image
from io import BytesIO
import base64
import numpy as np
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(__dir__))
from modules.shared import MTGPU_DETECTION, device


class RealESRGANWrapper:
    def __init__(self, device=device):
        self.device = device
        self.model = None

    def load_models(self, model_path):
        RealESRGAN_models = {
            "RealESRGAN_x4plus": RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=4,
            )
        }

        RealESRGAN_dir = os.path.dirname(model_path)
        sys.path.append(os.path.abspath(RealESRGAN_dir))
        model_name = os.path.basename(model_path).split(".")[0]
        self.model = RealESRGANer(
            scale=2,
            model_path=model_path,
            model=RealESRGAN_models[model_name],
            pre_pad=0,
            half=False,
            device="cpu",
        )
        self.model.model.name = model_name

    def start(self):
        
        torch.musa.empty_cache() if MTGPU_DETECTION else torch.cuda.empty_cache()
        gc.collect()
        assert self.model is not None, "please load deepbooru model first"
        self.model.model.to(self.device)
        self.model.device = self.device

    def stop(self):
        
        self.model.model.to("cpu")
        self.model.device = "cpu"
        torch.musa.empty_cache() if MTGPU_DETECTION else torch.cuda.empty_cache()
        gc.collect()

    def super_resolution(self, image):
        assert image is not None, "Input image should not be None"
        image = image.convert("RGB")
        np_image = np.array(image, dtype=np.uint8)
        output, _ = self.model.enhance(np_image)
        res = Image.fromarray(output)
        return res

    def __call__(self, image):

        self.start()
        res = self.super_resolution(image)
        self.stop()

        return res
