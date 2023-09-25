import numpy as np
from PIL import Image
from tqdm import tqdm

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(__dir__))
from wrappers import RealESRGANWrapper


class TileRealESRGAN(RealESRGANWrapper):
    def __init__(self, device):
        super(TileRealESRGAN, self).__init__(device=device)

    def get_views(self, width, height, window_size=720):
        stride = window_size - 2
        window_height_size = min(height, window_size)
        window_width_size = min(width, window_size)

        if height % stride == 0:
            num_blocks_height = int(max(1, height // stride))
        else:
            num_blocks_height = int(height // stride + 1)
        if width % stride == 0:
            num_blocks_width = int(max(1, width // stride))
        else:
            num_blocks_width = int(width // stride + 1)
        total_num_blocks = int(num_blocks_height * num_blocks_width)
        views = []

        for i in range(total_num_blocks):
            h_start = int((i // num_blocks_width) * stride)
            h_end = h_start + window_height_size
            if h_end > height:
                h_start = int(height - stride)
                h_end = int(height)
            w_start = int((i % num_blocks_width) * stride)
            w_end = w_start + window_width_size
            if w_end > width:
                w_start = int(width - stride)
                w_end = int(width)
            views.append((h_start, h_end, w_start, w_end))
        return views

    def __call__(self, image):

        self.start()

        sr_ratio = 4
        width, height = image.size
        width = int(width // 2 * 2)
        height = int(height // 2 * 2)
        image = image.resize((width, height))
        view_grids = self.get_views(width, height)
        np_value = np.zeros((height * sr_ratio, width * sr_ratio, 3))
        np_count = np.zeros_like(np_value)

        for h_start, h_end, w_start, w_end in tqdm(view_grids):
            image_crop = image.crop((w_start, h_start, w_end, h_end))
            sr_image_crop = self.super_resolution(image_crop)
            np_sr_image_crop = np.array(sr_image_crop)

            np_value[
                h_start * sr_ratio : h_end * sr_ratio,
                w_start * sr_ratio : w_end * sr_ratio,
            ] += np_sr_image_crop
            np_count[
                h_start * sr_ratio : h_end * sr_ratio,
                w_start * sr_ratio : w_end * sr_ratio,
            ] += 1

        self.stop()
        np_sr_image = np.where(np_count > 0, np_value / np_count, np_value).astype(
            np.uint8
        )
        sr_image = Image.fromarray(np_sr_image)

        return sr_image
