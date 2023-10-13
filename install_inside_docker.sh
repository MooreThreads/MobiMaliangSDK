apt-get update && apt-get install ffmpeg libsm6 libxext6 -y
pip install -r requirements.txt --no-deps
mkdir ~/.cache/huggingface/hub/;rm -rf ~/.cache/huggingface/hub/models--openai--clip-vit-large-patch14/;cp -r models/tools/models--openai--clip-vit-large-patch14/ ~/.cache/huggingface/hub