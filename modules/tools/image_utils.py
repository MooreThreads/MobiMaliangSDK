import base64
from io import BytesIO
from PIL import Image

# given an image path, read it and return the base64 encoding
def read_img_as_base64(path):
    with open(path, "rb") as fr:
        encoded = base64.b64encode(fr.read()).decode()
        img = f"data:image/jpeg;base64,{encoded}"
    return img


# read the generation parameters from png info
def read_png_info(image):
    args = {}

    png_info = image.info
    params = png_info.pop("parameters")

    params = params.split("\nNegative prompt: ")
    args["prompt"] = params[0]

    params = params[1]
    pos = params.rfind("\n")
    args["negative_prompt"] = params[:pos]

    params = params[pos + 1 :].split(", ")

    args["steps"] = int(params[0].lstrip("Steps: "))
    args["cfg"] = float(params[2].lstrip("CFG scale: "))
    args["seed"] = int(params[3].lstrip("Seed: "))
    args["width"], args["height"] = params[4].lstrip("Size: ").split("x")
    args["width"] = int(args["width"])
    args["height"] = int(args["height"])

    return args

def image2base64(img):
    output_buffer = BytesIO()
    img.save(output_buffer, format='PNG')
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data)
    return base64_str

def base642image(base64_str):
    image = base64.b64decode(base64_str)
    image = BytesIO(image)
    image = Image.open(image)
    return image