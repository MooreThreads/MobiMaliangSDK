import torch
import gc
import numpy as np
from PIL import Image
from controlnet_aux import (
    CannyDetector,
    ContentShuffleDetector,
    OpenposeDetector,
    LineartDetector,
    NormalBaeDetector,
    HEDdetector,
    PidiNetDetector,
    MLSDdetector,
)
from transformers import pipeline

import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(__dir__))
from shared import MTGPU_DETECTION, device
    
class CannyDetectorWrapper:
    def __init__(self):
        self.processor = CannyDetector()

    def __call__(self, input_image, low_threshold=100, high_threshold=200):
        return self.processor(input_image, low_threshold, high_threshold)


class ContentShuffleDetectorWrapper:
    def __init__(self):
        self.processor = ContentShuffleDetector()

    def __call__(
        self,
        input_image,
        detect_resolution=512,
        image_resolution=512,
        output_type="pil",
    ):
        return self.processor(
            input_image=input_image,
            detect_resolution=detect_resolution,
            image_resolution=image_resolution,
            output_type=output_type,
        )


class ResizeForConditionalImageWrapper:
    def __init__(self):
        pass

    def __call__(self, input_image, down_sampling_ratio=8.0):
        input_image = input_image.convert("RGB")
        W, H = input_image.size
        w, h = int(W // down_sampling_ratio), int(H // down_sampling_ratio)
        img = input_image.resize((w, h), resample=Image.LANCZOS)
        H = int(round(H / 8.0)) * 8
        W = int(round(W / 8.0)) * 8
        img = img.resize((W, H), resample=Image.LANCZOS)
        return img


class ZoomConditionalImageWrapper:
    def __init__(self):
        pass

    def __call__(self, input_image, up_sampling_ratio=2.0):
        input_image = input_image.convert("RGB")
        w, h = input_image.size
        W, H = int(w * up_sampling_ratio), int(h * up_sampling_ratio)
        H = int(round(H / 8.0)) * 8
        W = int(round(W / 8.0)) * 8
        img = input_image.resize((W, H), resample=Image.LANCZOS)
        return img


class InpaintingConditionalImageWrapper:
    def __init__(self):
        pass

    def __call__(self, input_image, mask_image):

        image = np.array(input_image.convert("RGB")).astype(np.float32) / 255.0
        mask_image = np.array(mask_image.convert("L")).astype(np.float32) / 255.0

        assert (
            image.shape[0:1] == mask_image.shape[0:1]
        ), "image and mask_image must have the same image size"
        image[mask_image > 0.5] = -1.0  # set as masked pixel
        image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        return image


class ContorlnetAnnotatorBasicWrapper:
    def __init__(self, device=device):
        self.device = device
        self.processor = None

    def start(self):
        torch.musa.empty_cache() if MTGPU_DETECTION else torch.cuda.empty_cache()
        gc.collect()
        assert self.processor is not None, "pleaase load processor first"
        self.processor.to(self.device)

    def stop(self):
        self.processor.to("cpu")
        torch.musa.empty_cache() if MTGPU_DETECTION else torch.cuda.empty_cache()
        gc.collect()


class OpenposeDetectorWrapper(ContorlnetAnnotatorBasicWrapper):
    def __init__(self, device=device):
        super(OpenposeDetectorWrapper, self).__init__(device=device)

    def load_models(self, models_dir):
        self.processor = OpenposeDetector.from_pretrained(models_dir).to("cpu")

    def __call__(self, input_image, include_hand=True, include_face=True):

        self.start()
        procesed_image = self.processor(
            input_image=input_image,
            include_hand=include_hand,
            include_face=include_face,
        )
        self.stop()

        return procesed_image


class LineartDetectorWrapper(ContorlnetAnnotatorBasicWrapper):
    def __init__(self, device=device):
        super(LineartDetectorWrapper, self).__init__(device=device)

    def load_models(self, models_dir):
        self.processor = LineartDetector.from_pretrained(models_dir).to("cpu")

    def __call__(
        self,
        input_image,
        coarse=False,
        detect_resolution=512,
        image_resolution=512,
        output_type="pil",
    ):

        self.start()
        procesed_image = self.processor(
            input_image=input_image,
            coarse=coarse,
            detect_resolution=detect_resolution,
            image_resolution=image_resolution,
            output_type=output_type,
        )
        self.stop()

        return procesed_image


class NormalBaeDetectorWrapper(ContorlnetAnnotatorBasicWrapper):
    def __init__(self, device=device):
        super(NormalBaeDetectorWrapper, self).__init__(device=device)

    def load_models(self, models_dir):
        self.processor = NormalBaeDetector.from_pretrained(models_dir).to("cpu")

    def __call__(
        self,
        input_image,
        detect_resolution=512,
        image_resolution=512,
        output_type="pil",
    ):

        self.start()
        procesed_image = self.processor(
            input_image=input_image,
            detect_resolution=detect_resolution,
            image_resolution=image_resolution,
            output_type=output_type,
        )
        self.stop()

        return procesed_image


class HEDdetectorWrapper(ContorlnetAnnotatorBasicWrapper):
    def __init__(self, device=device):
        super(HEDdetectorWrapper, self).__init__(device=device)

    def load_models(self, models_dir):
        self.processor = HEDdetector.from_pretrained(models_dir).to("cpu")

    def __call__(
        self,
        input_image,
        detect_resolution=512,
        image_resolution=512,
        safe=False,
        output_type="pil",
        scribble=False,
    ):

        self.start()
        procesed_image = self.processor(
            input_image=input_image,
            detect_resolution=detect_resolution,
            image_resolution=image_resolution,
            safe=safe,
            output_type=output_type,
            scribble=scribble,
        )
        self.stop()

        return procesed_image


class PidiNetDetectorWrapper(ContorlnetAnnotatorBasicWrapper):
    def __init__(self, device=device):
        super(PidiNetDetectorWrapper, self).__init__(device=device)

    def load_models(self, models_dir):
        self.processor = PidiNetDetector.from_pretrained(models_dir).to("cpu")

    def __call__(
        self,
        input_image,
        detect_resolution=512,
        image_resolution=512,
        safe=False,
        output_type="pil",
        scribble=False,
        apply_filter=False,
    ):

        self.start()
        procesed_image = self.processor(
            input_image=input_image,
            detect_resolution=detect_resolution,
            image_resolution=image_resolution,
            safe=safe,
            output_type=output_type,
            scribble=scribble,
            apply_filter=apply_filter,
        )
        self.stop()

        return procesed_image


class MLSDdetectorWrapper(ContorlnetAnnotatorBasicWrapper):
    def __init__(self, device=device):
        super(MLSDdetectorWrapper, self).__init__(device=device)

    def load_models(self, models_dir):
        self.processor = MLSDdetector.from_pretrained(models_dir).to("cpu")

    def __call__(
        self,
        input_image,
        thr_v=0.1,
        thr_d=0.1,
        detect_resolution=512,
        image_resolution=512,
        output_type="pil",
    ):

        self.start()
        procesed_image = self.processor(
            input_image=input_image,
            thr_v=thr_v,
            thr_d=thr_d,
            detect_resolution=detect_resolution,
            image_resolution=image_resolution,
            output_type=output_type,
        )
        self.stop()

        return procesed_image
