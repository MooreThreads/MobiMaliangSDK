# -*- coding: UTF-8 -*-

# 摩笔马良WebUI前端代码
# 包括模型处理, 页面样式, 侧边栏, 创作tab, 画廊tab, 工具tab六部分内容

import numpy as np
import streamlit as st
from PIL import Image, ImageFilter, PngImagePlugin
from glob import glob
from tqdm import tqdm
import pickle
import os
import cv2
import random
import time
import sys
from io import BytesIO
from config import *
import warnings
warnings.filterwarnings("ignore")
os.environ["PYTORCH_MUSA_ALLOC_CONF"] = "max_split_size_mb:32"

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(__dir__))

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from diffusers import logging
logging.set_verbosity_error()

from modules.shared import SDModel, Controlnet, MTGPU_DETECTION, device
from modules.basic.txt2img import txt2img
from modules.basic.img2img import img2img
from modules.controlnet.canny2img import canny2img
from modules.controlnet.pose2img import pose2img
from modules.controlnet.tilehd2img import tilehd2img
from modules.controlnet.mlsd2img import mlsd2img
from modules.tools.image_utils import read_img_as_base64, read_png_info
from modules.tools.prompt_parser import get_lora_from_prompt, contain_zh
from modules.tools.prompt_engineering import get_enhancer, get_artists, get_neg_prompt
from modules.wrappers import *

from transformers import CLIPModel

##### maliang #####

args, btns = {}, {}

args["save_dir"] = os.path.join("frontend", "output")
os.makedirs(args["save_dir"], exist_ok=True)

args["sd_ckpts"] = sorted(glob(os.path.join("models", "Stable-diffusion", "*")))
args["sd_ckpts"] = [path.split(os.sep)[-1] for path in args["sd_ckpts"]]

args["text_embeddings"] = sorted(glob(os.path.join("models", "embeddings", "*.pt")))

args["lora_names"] = sorted(glob(os.path.join("models", "lora", "*.safetensors")))
args["lora_names"] = [
    "<lora:" + path.split(os.sep)[-1][:-len(".safetensors")] + ":0.8>"
    for path in args["lora_names"]
]
args["loras"] = {}

random_prompt_path = np.random.choice(
    glob(os.path.join("assets", "random_prompts", "*"))
)
with open(random_prompt_path, "r") as fr:
    lines = fr.readlines()
    args["random_prompts"] = [line.rstrip("\n") for line in lines]

args["sampler"] = SAMPLER

args["loaded_loras"] = os.path.join("models", "loaded_loras")
os.makedirs(args["loaded_loras"], exist_ok=True)
os.makedirs(os.path.join(args["loaded_loras"], "fixed"), exist_ok=True)
os.makedirs(os.path.join(args["loaded_loras"], "dynamic"), exist_ok=True)
loaded_ckpt_path = os.path.join(args["loaded_loras"], "loaded_ckpt.txt")


@st.cache_resource(max_entries=1)
def load_ckpt():
    ml = SDModel(device=device, requires_safety_checker=NEED_SAFETY)

    path = os.path.join("models", "Stable-diffusion", st.session_state.ckpt_name)
    ml.load_models(path, diffusers_format=os.path.isdir(path))

    _, _ = txt2img(
        sd_model=ml,
        prompt="",
        height=256,
        width=256,
        num_inference_steps=1,
        guidance_scale=7.5,
        negative_prompt=None,
        num_images_per_prompt=1,
        seed=1234,
        callback=None,
        text_inversion_model_name_or_path=args["text_embeddings"],
        fast_mode=False,
    )

    with open(loaded_ckpt_path, "w") as fw:
        fw.write(st.session_state.ckpt_name)

    return ml

def switch_ckpt():
    ml.offload_models()

    path = os.path.join("models", "Stable-diffusion", st.session_state.ckpt_name)
    ml.load_models(path, diffusers_format=os.path.isdir(path))

    _, _ = txt2img(
        sd_model=ml,
        prompt="",
        height=256,
        width=256,
        num_inference_steps=1,
        guidance_scale=7.5,
        negative_prompt=None,
        num_images_per_prompt=1,
        seed=1234,
        callback=None,
        text_inversion_model_name_or_path=args["text_embeddings"],
        fast_mode=False,
    )

    with open(loaded_ckpt_path, "w") as fw:
        fw.write(st.session_state.ckpt_name)

@st.cache_resource()
def load_controlnet(control_type):
    net = Controlnet(device=device)

    if control_type == "canny":
        processor = CannyDetectorWrapper()
        net.load_models(
            os.path.join("models", "controlnet", "canny_v11"), diffusers_format=True
        )

    elif control_type == "pose":
        processor = OpenposeDetectorWrapper(device=device)
        processor.load_models(os.path.join("models", "controlnet", "annotators"))
        net.load_models(
            os.path.join("models", "controlnet", "openpose_v11"), diffusers_format=True
        )

    elif control_type == "tile":
        processor = ZoomConditionalImageWrapper()
        net.load_models(
            os.path.join("models", "controlnet", "tile_v11"), diffusers_format=True
        )

    elif control_type == 'mlsd':
        processor = MLSDdetectorWrapper(device="cpu")
        processor.load_models(os.path.join("models", "controlnet", "annotators"))
        net.load_models(
            os.path.join("models", "controlnet", "mlsd_v11"), diffusers_format=True
        )

    return processor, net


mlcn_canny, mlcn_pose, mlcn_tile, mlcn_mlsd = None, None, None, None


@st.cache_resource()
def load_safetensors(safetensors_name):
    sf = load_file(safetensors_name)
    return sf


@st.cache_resource()
def load_translator():
    ml_translator = Translator(device="cpu")
    ml_translator.load_models(os.path.join("models", "tools", "zh2en"))
    return ml_translator


ml_translator = None


@st.cache_resource()
def load_sr():
    ml_sr = RealESRGANWrapper(device=device)
    ml_sr.load_models(os.path.join("models", "tools", "RealESRGAN_x4plus.pth"))
    return ml_sr


ml_sr = None


@st.cache_resource()
def load_tagger():
    ml_tagger = DeepDanbooruWrapper(device=device)
    ml_tagger.load_models(os.path.join("models", "tools", "model-resnet_custom_v3.pt"))
    return ml_tagger


ml_tagger = None


@st.cache_resource()
def load_enhancer():
    with open(os.path.join("assets", "enhancer.pkl"), "rb") as fr:
        data_enhancer = pickle.load(fr)

    with open(os.path.join("assets", "artist.txt"), "r") as fr:
        lines = fr.readlines()
        artists_list = [line.rstrip("\n").split("^")[0] for line in lines]

    return [data_enhancer, artists_list]


ml_enhancer = load_enhancer()


@st.cache_resource
def load_dynamic_loras():
    dynamic_loras_dir = os.path.join("models", "tools", "dynamic_loras")

    if os.path.exists(dynamic_loras_dir):
        ml_clip = CLIPModel.from_pretrained(os.path.join("models", "tools", "clip-vit-large-patch14"))
        ml_cc = np.load(os.path.join(dynamic_loras_dir, "centroids.npy"))
        ml_loras_paths = sorted(glob(os.path.join(dynamic_loras_dir, "lora", "*.safetensors")))
    else:
        ml_clip, ml_cc, ml_loras_paths = None, None, None

    return ml_clip, ml_cc, ml_loras_paths

ml_clip, ml_cc, ml_loras_paths = load_dynamic_loras()


##### style #####

st.markdown(
    """
    <style type='text/css'>
    .css-pxxe24 {
        visibility: hidden;
    }
    div[data-testid='stTickBar'] {
        display: none;
    }
    div.block-container {
        padding-bottom: 50px;
    }

    section[data-testid='stSidebar'] {
        width: 25% !important;
        min-width: 356px !important;
    }
    section[data-testid='stSidebar'] div.block-container {
        padding-bottom: 0px;
    }
    section[data-testid='stSidebar'] > div:first-child > div:nth-child(1) {
        display: none;
    }
    section[data-testid='stSidebar'] > div:first-child > div:nth-child(2) {
        padding: 1.8rem;
    }
    section[data-testid='stSidebar'] > div:first-child > div:nth-child(2) > div:first-child > div:first-child > div:first-child > div:nth-child(1) p {
        margin-top: -10px;
    }
    section[data-testid='stSidebar'] > div:first-child > div:nth-child(2) > div:first-child > div:first-child > div:first-child > div:nth-child(1) div[role='progressbar'] {
        margin-top: 10px;
        margin-bottom: 20px;
    }
    section[data-testid='stSidebar'] > div:first-child > div:nth-child(2) > div:first-child > div:first-child > div:first-child > div:nth-child(2),
    section[data-testid='stSidebar'] > div:first-child > div:nth-child(2) > div:first-child > div:first-child > div:first-child > div:nth-child(3) {

    }
    section[data-testid='stSidebar'] > div:first-child > div:nth-child(2) > div:first-child > div:first-child > div:first-child > div:nth-child(2) > div[data-testid='stVerticalBlock'] div[data-testid='stImage'],
    section[data-testid='stSidebar'] > div:first-child > div:nth-child(2) > div:first-child > div:first-child > div:first-child > div:nth-child(3) > div[data-testid='stVerticalBlock'] div[data-testid='stImage'] {
        width: 100%;
    }
    section[data-testid='stSidebar'] > div:first-child > div:nth-child(2) > div:first-child > div:first-child > div:first-child > div:nth-child(2) > div[data-testid='stVerticalBlock'] button[title='View fullscreen'],
    section[data-testid='stSidebar'] > div:first-child > div:nth-child(2) > div:first-child > div:first-child > div:first-child > div:nth-child(3) > div[data-testid='stVerticalBlock'] button[title='View fullscreen'] {
        display: none;
    }
    section[data-testid='stSidebar'] > div:first-child > div:nth-child(2) > div:first-child > div:first-child > div:first-child > div:nth-child(2) > div[data-testid='stVerticalBlock'] button[kind='secondary'],
    section[data-testid='stSidebar'] > div:first-child > div:nth-child(2) > div:first-child > div:first-child > div:first-child > div:nth-child(3) > div[data-testid='stVerticalBlock'] button[kind='secondary'],
    div.st-bc > div:nth-child(5) div.stDownloadButton button[kind='secondary'] {
        position: absolute;
        top: -60px;
        right: 7px;
        background-color: rgba(13, 17, 23, 0.4);
        width: 80px;
        border-radius: 4px;
    }
    section[data-testid='stSidebar'] > div:first-child > div:nth-child(2) > div:first-child > div:first-child > div:first-child > div:nth-child(2) > div[data-testid='stVerticalBlock'] button[kind='secondary']:hover,
    section[data-testid='stSidebar'] > div:first-child > div:nth-child(2) > div:first-child > div:first-child > div:first-child > div:nth-child(3) > div[data-testid='stVerticalBlock'] button[kind='secondary']:hover,
    div.st-bc > div:nth-child(5) div.stDownloadButton button[kind='secondary']:hover {
        color: rgb(250, 250, 250);
        background-color: #FF671D;
        background-opacity: 1;
    }


    section.main > div.block-container > div:first-child > div > div.element-container button[title='View fullscreen'] {
        display: none;
    }
    section.main > div.block-container > div:first-child > div > div.element-container img {
        width: 50%;
        margin-left: 30%;
        margin-bottom: 15px;
    }
    div.st-bc > div:nth-child(1) div[data-baseweb='tab-list'] button:first-child {
        margin-left: 33%;
        margin-right: 4%;
    }
    div.st-bc > div:nth-child(1) div[data-baseweb='tab-list'] button:nth-child(2) {
        margin-right: 4%;
    }


    div.st-bc > div:nth-child(3) > div > div > div:nth-child(2) button {
        width: 100%;
    }
    div.st-bc > div:nth-child(3) > div > div > div:nth-child(2) > div:nth-child(3) button {
        border-color: #FF671D;
    }
    div.st-bc > div:nth-child(3) > div > div > div:nth-child(2) > div:nth-child(3) button:hover {
        color: rgb(250, 250, 250);
        background-color: #FF671D;
    }
    div.st-bc > div:nth-child(3) button[title='View fullscreen'] {
        display: none;
    }


    .image_cell {
        width: 160px;
        margin: 10px 20px 0px 0px;
        border-radius: 6px;
        border: 2px solid #0E1117;
        opacity: 0.7;
        display: inline-block;
        position: relative;
    }
    div.st-bc > div:nth-child(4) div[data-testid='stHorizontalBlock'] div[data-testid='stVerticalBlock'] div[data-testid='stVerticalBlock']:hover .image_cell {
        opacity: 1;
        border-color: #FF671D;
        cursor: pointer;
    }
    .image_cell img {
        width:100%;
        border-radius:4px;
    }
    div.st-bc > div:nth-child(4) div[data-testid='stHorizontalBlock'] div[data-testid='stVerticalBlock'] div[data-testid='stVerticalBlock'] {
        margin-bottom: -80px;
    }
    div.st-bc > div:nth-child(4) div[data-testid='stHorizontalBlock'] div[data-testid='stVerticalBlock'] div[data-testid='stVerticalBlock'] div.element-container {
        margin-bottom: -16px;
    }
    div.st-bc > div:nth-child(4) div.stButton button[kind='secondary'] {
        position: relative;
        background-color: rgba(13, 17, 23, 0.4);
        opacity: 0;
        border-radius: 4px;
        padding: 0 10px 0 10px;
    }
    div.st-bc > div:nth-child(4) div[data-testid='stHorizontalBlock'] div[data-testid='stVerticalBlock'] div[data-testid='stVerticalBlock'] div.element-container:nth-child(1) {
        margin-bottom: -24px;
    }
    div.st-bc > div:nth-child(4) div[data-testid='stHorizontalBlock'] div[data-testid='stVerticalBlock'] div[data-testid='stVerticalBlock'] div.element-container:nth-child(2) div.stButton button[kind='secondary'] {
        right: -7px;
        top: -22px;
    }
    div.st-bc > div:nth-child(4) div[data-testid='stHorizontalBlock'] div[data-testid='stVerticalBlock'] div[data-testid='stVerticalBlock'] div.element-container:nth-child(3) div.stButton button[kind='secondary'] {
        right: -57px;
        top: -60px;
        visibility: hidden;
    }
    div.st-bc > div:nth-child(4) div[data-testid='stHorizontalBlock'] div[data-testid='stVerticalBlock'] div[data-testid='stVerticalBlock'] div.element-container:nth-child(4) div.stButton button[kind='secondary'] {
        right: -107px;
        top: -98px;
    }
    div.st-bc > div:nth-child(4) div[data-testid='stHorizontalBlock'] div[data-testid='stVerticalBlock'] div[data-testid='stVerticalBlock']:hover div.stButton button[kind='secondary'] {
        opacity: 1;
    }
    div.st-bc > div:nth-child(4) div.stButton button[kind='secondary']:hover {
        color: rgb(250, 250, 250);
        background-color: #FF671D;
        background-opacity: 1;
    }
    div.st-bc > div:nth-child(4) div.stButton button[kind='secondary'] p {
        font-size: 12px;
    }
    div.st-bc > div:nth-child(4) > div > div > div:nth-child(2) {
        margin-top: 20px;
    }
    div.st-bc > div:nth-child(4) > div > div > div:nth-child(2) > div:nth-child(3) p {
        margin-top: 7px;
    }
    

    div.st-bc > div:nth-child(5) div[data-testid='stExpander']:nth-child(1) div.element-container:nth-child(3),
    div.st-bc > div:nth-child(5) div[data-testid='stExpander']:nth-child(1) div.element-container:nth-child(5) {
        margin-bottom: -16px;
    }

    button[title="View fullscreen"] {
        right: 2px;
        top: 2px;
        color: #FFF;
        background-color: #FF671D;
    }

    footer {
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

##### header #####

st.image(Image.open(os.path.join("assets", "imgs", "banner.png")), caption="")

tab1, tab2, tab3 = st.tabs(["创作\t:pencil2:", "画廊\t:city_sunrise:", "工具\t:lollipop:"])


##### 创作页面 #####

with tab1:
    prompt_container = st.container()
    col1, col2, col3 = st.columns([4, 1, 1])

    with col1:
        args["negative_prompt"] = st.text_input(
            label="我不想要（可选）",
            value="",
            placeholder="我不想要（可选）",
            label_visibility="collapsed",
        )

    with col2:
        btns["choose_random_prompt"] = st.button("随机描述", on_click=None)
        if btns["choose_random_prompt"]:
            st.session_state.random_prompt = np.random.choice(args["random_prompts"])

    with col3:
        btns["start_generation"] = st.button("创 作", on_click=None)

    with prompt_container:
        args["prompt"] = st.text_area(
            label="描述画面内容",
            value=st.session_state.random_prompt
            if "random_prompt" in st.session_state
            else "",
            placeholder="描述画面内容",
            label_visibility="collapsed",
            height=160,
        )

    with st.container():
        col1, col2 = st.columns([1, 1])
        with col1:
            if os.path.exists(loaded_ckpt_path):
                with open(loaded_ckpt_path, "r") as fr:
                    lines = fr.readlines()

                    if len(lines) > 0:
                        line = lines[0]
                    else:
                        line = args["sd_ckpts"][0] 
            else:
                line = args["sd_ckpts"][0]

            args["model"] = st.selectbox(
                label="选择模型", options=args["sd_ckpts"], index=args["sd_ckpts"].index(line), on_change=switch_ckpt, key="ckpt_name"
            )
            ml = load_ckpt()

            if FREE_HW:
                args["height"] = st.slider(label="高度", min_value=384, max_value=MAX_SIZE, value=512, step=64)
                st.write('\n')

        with col2:
            args["num_images"] = st.number_input(
                label="生成数量", min_value=1, max_value=4, value=1, step=1, on_change=None
            )

            if FREE_HW:
                args["width"] = st.slider(label="宽度", min_value=384, max_value=MAX_SIZE, value=512, step=64)
                st.write('\n')

        if not FREE_HW:
            args["hw"] = st.radio(
                "分辨率 (高 x 宽)",
                ("512 x 512", "768 x 512", "512 x 768", "576 x 576", "640 x 640"),
                index=0,
                horizontal=True,
            )
            args["hw"] = args["hw"].split(" x ")
            args["height"] = int(args["hw"][0])
            args["width"] = int(args["hw"][1])

    col1, col2 = st.columns([1, 2])
    with col1:
        with st.expander("**高级配置**\t:rocket:（可选）", expanded=False):
            args["steps"] = st.slider(
                label="步数（推荐20）", min_value=20, max_value=30, value=20, step=2
            )
            st.write("\n")
            args["cfg"] = st.slider(
                label="引导强度（推荐7.5）", min_value=5.0, max_value=10.0, value=7.5, step=0.5
            )
            st.write("\n")

            if "last_seed" not in st.session_state:
                st.session_state.last_seed = "-1"

            seed_radio = st.radio("随机数种子", ("随机", "上次"), index=0, horizontal=True)
            args["seed"] = st.text_input(
                "种子值（-1表示随机选取）",
                value="-1" if seed_radio == "随机" else st.session_state.last_seed,
                on_change=None,
                placeholder="",
            )
            args["use_enhance"] = st.checkbox("正向增强", value=True)
            args["use_negative_enhance"] = st.checkbox("负向增强", value=True)
            args["use_fast_mode"] = st.checkbox("快速生成", value=True)

    with col2:
        with st.expander("**图像控制**\t:cityscape:（可选）", expanded=False):
            args["init_image"] = st.file_uploader(
                "上传参考图", type=["png", "jpg", "jpeg"], on_change=None
            )

            if args["init_image"] is not None:
                args["init_image"] = np.asarray(
                    bytearray(args["init_image"].read()), dtype=np.uint8
                )
                args["init_image"] = cv2.imdecode(args["init_image"], 1)[:, :, ::-1]
                args["init_image"] = Image.fromarray(
                    args["init_image"].astype("uint8")
                ).convert("RGB")

                w, h = args["init_image"].size
                if max(w, h) > MAX_SIZE:
                    if h > w:
                        args["init_image"] = args["init_image"].resize(
                            (int(MAX_SIZE * w / h), MAX_SIZE)
                        )
                    else:
                        args["init_image"] = args["init_image"].resize(
                            (MAX_SIZE, int(MAX_SIZE * h / w))
                        )

                args["use_init_image"] = True
                st.image(args["init_image"])

                args["denoising_strength"] = st.slider(
                    label="重绘幅度", min_value=0.1, max_value=1.0, value=0.8, step=0.1
                )
                st.write("\n")
            else:
                args["use_init_image"] = False

            args["controlnet_type"] = st.selectbox(
                label="可控生成", options=["不启用", "边缘", "骨骼", "直线"], index=0, on_change=None
            )

            if args["controlnet_type"] != "不启用":
                args["controlnet_image"] = st.file_uploader(
                    "上传控制图", type=["png", "jpg", "jpeg"], on_change=None
                )

                if args["controlnet_image"] is not None:
                    args["controlnet_image"] = np.asarray(
                        bytearray(args["controlnet_image"].read()), dtype=np.uint8
                    )
                    args["controlnet_image"] = cv2.imdecode(
                        args["controlnet_image"], 1
                    )[:, :, ::-1]
                    args["controlnet_image"] = Image.fromarray(
                        args["controlnet_image"].astype("uint8")
                    ).convert("RGB")
                    st.image(args["controlnet_image"])

    with st.expander("**LORA**\t:fire:（可选）", expanded=False):
        st.caption("可用lora，在描述框中使用，可修改权重")
        for lora in args["lora_names"]:
            st.code(lora)

        args['use_dynamic_lora'] = st.checkbox('动态增强', value=True)
        if args['use_dynamic_lora']:
            args['dynamic_mode'] = st.radio('动态增强强度', ('自动', '强', '中', '弱'), index=0, horizontal=True, label_visibility='collapsed')

skip_layers_none = []
skip_layers_some = [
    "lora_unet_mid_block_attentions_0",
    "lora_unet_up_blocks_1_attentions_0",
    "lora_unet_up_blocks_1_attentions_1",
    "lora_unet_up_blocks_1_attentions_2",
]
skip_layers_more = [
    "lora_unet_down_blocks_2_attentions_0",
    "lora_unet_down_blocks_2_attentions_1",
    "lora_unet_mid_block_attentions_0",
    "lora_unet_up_blocks_1_attentions_0",
    "lora_unet_up_blocks_1_attentions_1",
    "lora_unet_up_blocks_1_attentions_2",
    "lora_unet_up_blocks_2_attentions_0",
    "lora_unet_up_blocks_2_attentions_1",
    "lora_unet_up_blocks_2_attentions_2",
]

if args["use_dynamic_lora"]:
    if args["dynamic_mode"] == "自动":
        if args["model"].find("ml-person") >= 0:
            dynamic_skip_layers = skip_layers_more
        elif args["model"].find("ml-2.5D") >= 0:
            dynamic_skip_layers = skip_layers_some
        else:
            dynamic_skip_layers = skip_layers_none
    elif args["dynamic_mode"] == "强":
        dynamic_skip_layers = skip_layers_none
    elif args["dynamic_mode"] == "中":
        dynamic_skip_layers = skip_layers_some
    elif args["dynamic_mode"] == "弱":
        dynamic_skip_layers = skip_layers_more

##### sidebar #####


def progress_callback(step, timestep, image, total_timestep=1000):
    timestep = timestep.to("cpu")
    percentage = (total_timestep - timestep + 1) / total_timestep
    pr_bar.progress(
        int(percentage * 100), text=":smiley:\t生成中，请稍等：{:.2f}%".format(percentage * 100)
    )
    return step, timestep, image


def get_dynamic_lora_from_prompt(prompt, N=3):
    if ml_clip is None:
        return []

    inputs = ml.tokenizer([prompt], padding=True, return_tensors='pt')
    if inputs['input_ids'].shape[1] > 77:
        sos = torch.tensor(49406) # inputs['input_ids'][0][0]
        eos = torch.tensor(49407) # inputs['input_ids'][0][-1]
        inputs['input_ids'] = inputs['input_ids'][:, :77]
        inputs['input_ids'][:, -1] = eos
        inputs['attention_mask'] = inputs['attention_mask'][:, :77]
        inputs['attention_mask'][:, -1] = 0

    text_features = ml_clip.get_text_features(**inputs).data.numpy()

    similarity = F.cosine_similarity(torch.tensor(text_features), torch.tensor(ml_cc), dim=1).data.numpy()
    index = np.argsort(similarity)[::-1]

    idx = index[:N]
    alphas = [similarity[i] for i in idx]

    start, end = 0.2, 0.6
    alphas = [(a - alphas[-1]) / (alphas[0] - alphas[-1]) * (end - start) + start for a in alphas]

    s = np.sum(alphas)
    alphas = [round(r / s, 3) for r in alphas]

    return [[ml_loras_paths[idx[i]], alphas[i]] for i in range(len(idx))]


def load_loras(loras, dynamic=False, skip_layers=[]):
    for lora in loras:
        name, alpha = lora

        if dynamic:
            lora_name = name.split(os.sep)[-1][:-len(".safetensors")]
            lora_path = os.path.join(args["loaded_loras"], "dynamic", f"{lora_name}.pkl")

            if not os.path.exists(lora_path):
                ml.load_lora(
                    name, alpha=alpha, is_path=True, skip_layers=skip_layers
                )

                with open(lora_path, "wb") as fw:
                    pickle.dump([alpha, skip_layers], fw)
        else:
            path = os.path.join("models", "lora", name + ".safetensors")
            lora_path = os.path.join(args["loaded_loras"], "fixed", f"{name}.pkl")
            
            if os.path.exists(path):
                if name not in args["loras"]:
                    args["loras"][name] = load_safetensors(path)

                if not os.path.exists(lora_path):
                    ml.load_lora(
                        args["loras"][name],
                        alpha=alpha,
                        is_path=False,
                        skip_layers=skip_layers,
                    )

                    with open(lora_path, "wb") as fw:
                        pickle.dump(alpha, fw)


def offload_loras(loras, dynamic=False, skip_layers=[]):
    for lora in loras:
        name, alpha = lora

        if dynamic:
            lora_name = name.split(os.sep)[-1][:-len(".safetensors")]
            lora_path = os.path.join(args["loaded_loras"], "dynamic", f"{lora_name}.pkl")

            if os.path.exists(lora_path):
                ml.offload_lora(
                    name, alpha=alpha, is_path=True, skip_layers=skip_layers
                )
                os.system(f"rm {lora_path}")
        else:
            path = os.path.join("models", "lora", name + ".safetensors")
            lora_path = os.path.join(args["loaded_loras"], "fixed", f"{name}.pkl")
            
            if os.path.exists(path):
                if name not in args["loras"]:
                    args["loras"][name] = load_safetensors(path)
                
                if os.path.exists(lora_path):
                    ml.offload_lora(
                        args["loras"][name],
                        alpha=alpha,
                        is_path=False,
                        skip_layers=skip_layers,
                    )
                    os.system(f"rm {lora_path}")


def enhance_prompt(prompt, seed, use_artist=False):
    enhancer = get_enhancer(ml_enhancer[0], seed)
    prompt += ", " + enhancer

    if use_artist:
        artist = get_artists(ml_enhancer[1], seed)
        if artist:
            prompt += ", (" + artist + ":0.6)"

    return prompt


def enhance_negative_prompt(neg):
    safety_neg_prompt, neg_embeddings_prompt, quality_neg_prompt = get_neg_prompt()

    if args["model"].find("ml-general") >= 0:
        neg_embeddings_prompt = ""

    if neg == "":
        neg = safety_neg_prompt + neg_embeddings_prompt + quality_neg_prompt
    else:
        neg = (
            safety_neg_prompt + neg + ", " + neg_embeddings_prompt + quality_neg_prompt
        )
    return neg


with st.sidebar:
    msg = st.container()
    msg.write(" ")

    res = st.container()

    if btns["start_generation"]:
        if args["prompt"] == "":
            msg.write(":slightly_frowning_face:\t请输入画面描述")
        else:
            pr_bar = msg.progress(0, text=":smiley:\t生成中，请稍等")

            if contain_zh(args["prompt"]):
                if ml_translator is None:
                    ml_translator = load_translator()

                args["prompt"] = ml_translator(args["prompt"])

            if contain_zh(args["negative_prompt"]):
                args["negative_prompt"] = ml_translator(args["negative_prompt"])

            if args["seed"] in ["-1", "0"]:
                use_seed = random.randrange(4294967294)
            else:
                try:
                    use_seed = int(args["seed"])
                except:
                    use_seed = random.randrange(4294967294)

            st.session_state.last_seed = str(use_seed)
            st.session_state.last_images = []

            # 加载lora
            fixed_loras, prompt = get_lora_from_prompt(args["prompt"])
            load_loras(fixed_loras)

            if args['use_dynamic_lora']:
                dynamic_loras = get_dynamic_lora_from_prompt(prompt)
                load_loras(dynamic_loras, dynamic=True, skip_layers=dynamic_skip_layers)

            negative_prompt = enhance_negative_prompt(args["negative_prompt"]) if args["use_negative_enhance"] else args["negative_prompt"]

            ori_prompt = prompt
            for i in range(args["num_images"]):
                if args["use_enhance"]:
                    prompt = enhance_prompt(ori_prompt, use_seed + i)

                if args["controlnet_type"] == "不启用" or args["controlnet_image"] is None:
                    if not args["use_init_image"]:
                        images, status = txt2img(
                            sd_model=ml,
                            prompt=prompt,
                            height=args["height"],
                            width=args["width"],
                            num_inference_steps=args["steps"],
                            guidance_scale=args["cfg"],
                            negative_prompt=negative_prompt,
                            num_images_per_prompt=1,
                            seed=use_seed + i,
                            callback=progress_callback,
                            fast_mode=args["use_fast_mode"],
                        )
                    else:
                        images, status = img2img(
                            sd_model=ml,
                            prompt=prompt,
                            image=args["init_image"],
                            strength=args["denoising_strength"],
                            num_inference_steps=args["steps"],
                            guidance_scale=args["cfg"],
                            negative_prompt=negative_prompt,
                            num_images_per_prompt=1,
                            seed=use_seed + i,
                            callback=progress_callback,
                        )

                    anno_image = None

                else:
                    if args["use_init_image"]:
                        w, h = args["init_image"].size
                        if w == args["width"] and h == args["height"]:
                            also_use_init_image = True
                        else:
                            also_use_init_image = False
                    else:
                        also_use_init_image = False

                    if args["controlnet_type"] == "边缘":
                        if mlcn_canny is None:
                            mlcn_canny = load_controlnet("canny")

                        mlcn_canny[1].controlnet = mlcn_canny[1].controlnet.to(device)

                        images, status, anno_image = canny2img(
                            sd_model=ml,
                            controlnet_model=mlcn_canny[1],
                            processor=mlcn_canny[0],
                            ori_image=args["controlnet_image"],
                            aux_image=args["init_image"]
                            if also_use_init_image
                            else None,
                            is_canny_image=False,
                            prompt=prompt,
                            height=args["height"],
                            width=args["width"],
                            strength=1.0,
                            num_inference_steps=args["steps"],
                            guidance_scale=args["cfg"],
                            negative_prompt=negative_prompt,
                            num_images_per_prompt=1,
                            seed=use_seed + i,
                            callback=progress_callback,
                            fast_mode=args["use_fast_mode"],
                        )

                        mlcn_canny[1].controlnet = mlcn_canny[1].controlnet.to("cpu")

                    elif args["controlnet_type"] == "骨骼":
                        if mlcn_pose is None:
                            mlcn_pose = load_controlnet("pose")

                        mlcn_pose[1].controlnet = mlcn_pose[1].controlnet.to(device)

                        images, status, anno_image = pose2img(
                            sd_model=ml,
                            controlnet_model=mlcn_pose[1],
                            processor=mlcn_pose[0],
                            ori_image=args["controlnet_image"],
                            aux_image=args["init_image"]
                            if also_use_init_image
                            else None,
                            is_pose_image=False,
                            prompt=prompt,
                            height=args["height"],
                            width=args["width"],
                            strength=1.0,
                            num_inference_steps=args["steps"],
                            guidance_scale=args["cfg"],
                            negative_prompt=negative_prompt,
                            num_images_per_prompt=1,
                            seed=use_seed + i,
                            callback=progress_callback,
                            fast_mode=args["use_fast_mode"],
                        )

                        mlcn_pose[1].controlnet = mlcn_pose[1].controlnet.to("cpu")

                    elif args["controlnet_type"] == "直线":
                        if mlcn_mlsd is None:
                            mlcn_mlsd = load_controlnet("mlsd")

                        mlcn_mlsd[1].controlnet = mlcn_mlsd[1].controlnet.to(device)

                        images, status, anno_image = mlsd2img(
                            sd_model=ml,
                            controlnet_model=mlcn_mlsd[1],
                            processor=mlcn_mlsd[0],
                            ori_image=args["controlnet_image"],
                            aux_image=args["init_image"]
                            if also_use_init_image
                            else None,
                            is_mlsd_image=False,
                            prompt=prompt,
                            height=args["height"],
                            width=args["width"],
                            strength=1.0,
                            num_inference_steps=args["steps"],
                            guidance_scale=args["cfg"],
                            negative_prompt=negative_prompt,
                            num_images_per_prompt=1,
                            seed=use_seed + i,
                            callback=progress_callback,
                            fast_mode=args["use_fast_mode"],
                        )

                        mlcn_mlsd[1].controlnet = mlcn_mlsd[1].controlnet.to("cpu")

                image = images[0]

                # nsfw
                if status:
                    image = image.filter(
                        ImageFilter.GaussianBlur(
                            radius=max(args["height"], args["width"]) // 8
                        )
                    )

                res.image(image)

                save_name = (
                    str(int(time.time()))
                    + "-"
                    + str(np.random.randint(0, 10e4)).zfill(4)
                    + ".png"
                )

                pnginfo_data = PngImagePlugin.PngInfo()
                value = args["prompt"] + "\n"
                value += "Negative prompt: " + args["negative_prompt"] + "\n"
                value += (
                    "Steps: "
                    + str(args["steps"])
                    + ", Sampler: "
                    + args["sampler"]
                    + ", "
                )
                value += "CFG scale: " + str(args["cfg"]) + ", "
                value += "Seed: " + str(use_seed + i) + ", "
                value += (
                    "Size: " + str(args["width"]) + "x" + str(args["height"]) + ", "
                )
                value += "Model: " + args["model"]
                pnginfo = {"parameters": value}

                for k, v in pnginfo.items():
                    pnginfo_data.add_text(k, str(v))

                save_path = os.path.join(args["save_dir"], save_name)
                image.save(save_path, pnginfo=pnginfo_data)
                st.session_state.last_images.append([image.copy(), save_name])

                w, h = image.size
                image.thumbnail((w // 2, h // 2))
                image.save(
                    os.path.join(args["save_dir"], save_name.rstrip(".png") + ".jpg")
                )

                with open(save_path, "rb") as fr:
                    res.download_button(
                        label="下载", data=fr, file_name=save_name, mime="image/png"
                    )

            if anno_image is not None:
                res.image(anno_image)

            pr_bar.empty()

    elif "last_images" in st.session_state:
        pr_bar = msg.progress(0, text=":smiley:\t生成中，请稍等")
        pr_bar.empty()

        last_images = st.session_state.last_images

        for item in last_images:
            image, save_name = item

            res.image(image)

            save_path = os.path.join(args["save_dir"], save_name)

            if os.path.exists(save_path):
                with open(save_path, "rb") as fr:
                    res.download_button(
                        label="下载", data=fr, file_name=save_name, mime="image/png"
                    )
            else:
                buffered = BytesIO()
                image.save(buffered, format="png")
                res.download_button(
                    label="下载",
                    data=buffered.getvalue(),
                    file_name=save_name,
                    mime="image/png",
                )

##### 画廊页面 #####

with tab2:
    paths = sorted(glob(os.path.join(args["save_dir"], "*.jpg")))[::-1]

    num_image_per_page = 20
    num_of_columns = 4
    total_page = len(paths) // num_image_per_page
    if len(paths) % num_image_per_page > 0:
        total_page += 1

    gallery = st.container()
    col1, col2, col3, col4 = gallery.columns([1, 1, 1, 1])
    columns = [col1, col2, col3, col4]

    if total_page > 1:
        col1, col2, col3 = st.columns([3, 4, 4])
        with col2:
            args["current_page"] = st.number_input(
                label="页码选择",
                min_value=1,
                max_value=total_page,
                value=1,
                step=1,
                on_change=None,
                label_visibility="collapsed",
            )
        with col3:
            st.write(f"共 {total_page} 页")
    else:
        args["current_page"] = 1

    for i, path in enumerate(
        paths[
            (args["current_page"] - 1)
            * num_image_per_page : args["current_page"]
            * num_image_per_page
        ]
    ):
        img = read_img_as_base64(path)

        funcs = columns[i % num_of_columns].container()

        funcs.write(
            f"""
            <div class='image_cell'>
                <img src='{img}'>
            </div>
            """,
            unsafe_allow_html=True,
        )

        name = path.split(os.sep)[-1].rstrip(".jpg")

        tiled_btn = funcs.button("精绘", key=f"hd_{name}", on_click=None)

        if tiled_btn:
            pr_bar = msg.progress(0, text=":smiley:\t精绘中，请稍等")

            st.session_state.last_images = []

            save_name = name + "_tiled.png"
            save_path = os.path.join(args["save_dir"], save_name)

            image = Image.open(os.path.join(args["save_dir"], f"{name}.png"))
            params = read_png_info(image)

            # 加载lora
            fixed_loras, prompt = get_lora_from_prompt(params["prompt"])
            load_loras(fixed_loras)

            if args['use_dynamic_lora']:
                dynamic_loras = get_dynamic_lora_from_prompt(prompt)
                load_loras(dynamic_loras, dynamic=True, skip_layers=dynamic_skip_layers)

            negative_prompt = enhance_negative_prompt(params["negative_prompt"]) if args["use_negative_enhance"] else params["negative_prompt"]

            if args["use_enhance"]:
                prompt = enhance_prompt(prompt, params["seed"])

            if mlcn_tile is None:
                mlcn_tile = load_controlnet("tile")

            mlcn_tile[1].controlnet = mlcn_tile[1].controlnet.to(mlcn_tile[1].device)

            images, status, anno_image = tilehd2img(
                sd_model=ml,
                controlnet_model=mlcn_tile[1],
                processor=mlcn_tile[0],
                ori_image=image,
                is_zoom_image=False,
                up_sampling_ratio=2,
                prompt=prompt,
                # height=params["height"],
                # width=params["width"],
                strength=0.3,
                num_inference_steps=params["steps"],
                guidance_scale=params["cfg"],
                negative_prompt=negative_prompt,
                num_images_per_prompt=1,
                seed=params["seed"],
                callback=progress_callback,
            )

            mlcn_tile[1].controlnet = mlcn_tile[1].controlnet.to("cpu")

            image = images[0]

            # nsfw
            if status:
                image = image.filter(
                    ImageFilter.GaussianBlur(
                        radius=max(params["height"], params["width"]) // 8
                    )
                )

            image.save(save_path)
            st.session_state.last_images.append([image.copy(), save_name])

            res.image(image)

            with open(save_path, 'rb') as fr:
                res.download_button(label='下载', data=fr, key=save_name, file_name=save_name, mime='image/png')

            pr_bar.empty()
            st.experimental_rerun()

        face_btn = funcs.button("修脸", key=f"face_{name}", on_click=None)

        del_btn = funcs.button("删除", key=f"del_{name}", on_click=None)
        if del_btn:
            del_path = os.path.join(args["save_dir"], name)
            os.system(f"rm {del_path}.png")
            os.system(f"rm {del_path}.jpg")

            del_hd_path = f"{del_path}_tiled.png"
            if os.path.exists(del_hd_path):
                os.system(f"rm {del_hd_path}")

            st.experimental_rerun()

##### 工具页面 #####


def record_download_hd():
    st.session_state.hd_image_has_download = True


with tab3:
    with st.expander("**X4超分辨率**\t:rainbow:", expanded=True):
        args["hd_image"] = st.file_uploader(
            "上传图片",
            key="upload_hd_image",
            type=["png", "jpg", "jpeg"],
            on_change=None,
            label_visibility="collapsed",
        )
        if args["hd_image"] is not None:
            filename = args["hd_image"].name
            args["hd_image"] = np.asarray(
                bytearray(args["hd_image"].read()), dtype=np.uint8
            )
            args["hd_image"] = cv2.imdecode(args["hd_image"], 1)[:, :, ::-1]
            args["hd_image"] = Image.fromarray(
                args["hd_image"].astype("uint8")
            ).convert("RGB")

            if (
                "last_hd_image_name" in st.session_state
                and st.session_state.last_hd_image_name == filename
            ):
                hd_image = st.session_state.last_hd_image
            else:
                with st.spinner(text=":smiley:\t超分中，请稍等"):
                    if ml_sr is None:
                        ml_sr = load_sr()

                    hd_image = ml_sr(args["hd_image"])
                    st.session_state.last_hd_image = hd_image
                    st.session_state.last_hd_image_name = filename
                    st.session_state.hd_image_has_download = False

            if (
                "hd_image_has_download" in st.session_state
                and not st.session_state.hd_image_has_download
            ):
                args["hd_image"] = args["hd_image"].resize(hd_image.size)
                combine = np.concatenate(
                    [np.array(args["hd_image"]), np.array(hd_image)], 1
                )
                combine = Image.fromarray(combine.astype("uint8")).convert("RGB")
                st.image(combine)

                buf = BytesIO()
                hd_image.save(buf, format="PNG")
                save_name = "hd_" + filename[: filename.rfind(".")] + ".png"
                st.download_button(
                    label="下载",
                    data=buf.getvalue(),
                    file_name=save_name,
                    mime="image/png",
                    on_click=record_download_hd,
                )

        else:
            st.session_state.last_hd_image = None
            st.session_state.last_hd_image_name = ""
            st.session_state.hd_image_has_download = False

    with st.expander("**获取图片描述**\t:pencil2:", expanded=True):
        args["tag_image"] = st.file_uploader(
            "上传图片",
            key="upload_tag_image",
            type=["png", "jpg", "jpeg"],
            on_change=None,
            label_visibility="collapsed",
        )
        if args["tag_image"] is not None:
            args["tag_image"] = np.asarray(
                bytearray(args["tag_image"].read()), dtype=np.uint8
            )
            args["tag_image"] = cv2.imdecode(args["tag_image"], 1)[:, :, ::-1]
            args["tag_image"] = Image.fromarray(
                args["tag_image"].astype("uint8")
            ).convert("RGB")

            if ml_tagger is None:
                ml_tagger = load_tagger()

            caption = ml_tagger(args["tag_image"])

            st.code(caption)

# 卸载lora
paths = sorted(glob(os.path.join(args["loaded_loras"], "fixed", "*")))
fixed_loras = []
for path in paths:
    with open(path, "rb") as fr:
        alpha = pickle.load(fr)
        fixed_loras.append([path.split(os.sep)[-1][:-len(".pkl")], alpha])
offload_loras(fixed_loras)

paths = sorted(glob(os.path.join(args["loaded_loras"], "dynamic", "*")))
dynamic_loras = []
skip_layers = []
for path in paths:
    with open(path, 'rb') as fr:
        alpha, skip_layers = pickle.load(fr)
        dynamic_loras.append([os.path.join("models", "tools", "dynamic_loras", "lora",
            path.split(os.sep)[-1][:-len('.pkl')] + ".safetensors"), alpha])

offload_loras(dynamic_loras, dynamic=True, skip_layers=skip_layers)