import torch
import gc
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(__dir__))
from modules.shared import MTGPU_DETECTION, device

class Translator:
    def __init__(self, device=device):
        self.device = device
        self.tokenizer = None
        self.model = None

    def load_models(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    def start(self):
        
        torch.musa.empty_cache() if MTGPU_DETECTION else torch.cuda.empty_cache()
        gc.collect()
        assert self.model is not None, "please load translator model first"
        self.model.to(self.device)

    def stop(self):
        
        self.model.to("cpu")
        torch.musa.empty_cache() if MTGPU_DETECTION else torch.cuda.empty_cache()
        gc.collect()

    def __call__(self, text):

        self.start()

        tokenized_text = self.tokenizer([text], return_tensors="pt")
        translation = self.model.generate(**tokenized_text)

        self.stop()

        return self.tokenizer.batch_decode(translation, skip_special_tokens=True)[0]
