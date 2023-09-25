import torch
import uvicorn
import os
import sys
import time
from fastapi import File, UploadFile, Request, FastAPI
from fastapi import Request
import json

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(os.path.dirname(__dir__)))

from modules.shared import SDModel, device
from modules.basic.txt2img import txt2img

from modules.tools.image_utils import image2base64, base642image
from modules.tools.loggers import server_logger

MODEL_DIR = "models"
BASE_MODEL_DIR = os.path.join(MODEL_DIR, "Stable-diffusion/ml-person-1.2.safetensors")
LORA_PATH = os.path.join(MODEL_DIR, "models/lora_models/blindbox_v1_mix.safetensors")

sd_model = SDModel(device=device, requires_safety_checker=False)
sd_model.load_models(BASE_MODEL_DIR, diffusers_format=False)
app = FastAPI()

@app.post("/text2image")
async def text2image(request: Request):
    try:
        params = await request.json()
        prompt = params.pop("prompt", "masterpiece, best quaility")
        seed = params.pop("seed", 0)
        images, status = txt2img(
            sd_model,
            prompt=prompt,
            seed=seed,
            **params
        )
        decoded_images = [image2base64(image) for image in images]
        rsp = {"images": decoded_images, "message":"ok"}
        
    except Exception as e:
        server_logger.error(" ------ {}".format(e))
        rsp = {"images": [], "message": f"err: {e}"}
    
    return rsp

@app.post("/lora2image")
async def lora2image(request: Request):
    try:
        params = await request.json()
        lora_path = params.pop("lora_path", LORA_PATH)
        sd_model.load_lora(lora_path, alpha=1)
        assert lora_path is not None
        
        prompt = params.pop("prompt", "masterpiece, best quaility")
        seed = params.pop("seed", 0)
        images, status  = txt2img(
            sd_model,
            prompt,
            seed=seed,
            lora_model_name_or_path_or_dict=LORA_PATH,
            lora_alpha=0.75,
            **params
        )
        decoded_images = [image2base64(image) for image in images]
        
        rsp = {"images": decoded_images, "message":"ok"}
    except Exception as e:
        server_logger.error(" ------ {}".format(e))
        rsp = {"images": [], "message": f"err: {e}"}
    return rsp

if __name__ == '__main__':
    uvicorn.run(app, port=1001, host="0.0.0.0")