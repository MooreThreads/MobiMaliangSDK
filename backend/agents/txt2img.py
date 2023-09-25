import requests
import json
import os
import sys
from PIL import Image

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(os.path.dirname(__dir__)))

from modules.tools.image_utils import image2base64, base642image

data = {
    "prompt": "a pretty puppy", 
    "negative_prompt": "ugly, low quaility, blur",
    "seed": 2023
}
# fill in your ip address and port like 0.0.0.0:1001
response = requests.post('http://x.x.x.x:x/text2image', json=data)
result = json.loads(response.content.decode("utf-8"))
base64_str = result["images"][0]
image = base642image(base64_str)
image.save("outputs/agent_txt2img.jpg")