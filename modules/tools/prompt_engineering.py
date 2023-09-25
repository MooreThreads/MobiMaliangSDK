import numpy as np


def get_enhancer(data, seed):
    np.random.seed(seed)
    quality = np.random.choice(
        data["quality"][0], size=3, replace=False, p=data["quality"][1]
    )

    composition = np.random.choice(
        data["composition"][0], size=1 if np.random.uniform(0, 1) > 0.5 else 0, replace=False, p=data["composition"][1]
    )

    color = np.random.choice(
        data["color"][0], size=1 if np.random.uniform(0, 1) > 0.5 else 0, replace=False, p=data["color"][1]
    )

    light = np.random.choice(
        data["light"][0], size=1, replace=False, p=data["light"][1]
    )

    content = np.random.choice(
        data["content"][0], size=3, replace=False, p=data["content"][1]
    )

    software = np.random.choice(
        data['software'][0], size=0, replace=False, p=data['software'][1]
    )

    photography = np.random.choice(
        data["photography"][0], size=0, replace=False, p=data["photography"][1]
    )

    result = np.concatenate(
        [quality, composition, color, light, content, software, photography]
    )
    np.random.shuffle(result)
    return ", ".join(list(result))


def get_artists(artists, seed):
    np.random.seed(seed)
    a = np.random.uniform(0, 1)

    # 0: 0.5; 1: 0.25; 2: 0.15; 3: 0.1
    if a < 0.5:
        num_artist = 0
        return ""
    elif a < 0.75:
        num_artist = 1
    elif a < 0.9:
        num_artist = 2
    else:
        num_artist = 3

    return ", ".join(np.random.choice(artists, num_artist, replace=False))


def get_neg_prompt():
    safety_neg_prompt = "nsfw, "
    neg_embeddings = "bad-picture-chill-75v, "
    quality_neg_prompt = "worst quality, low resolution, bad anatomy, malformed limbs, wrong fingers"
    return safety_neg_prompt, neg_embeddings, quality_neg_prompt
