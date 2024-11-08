import argparse
import asyncio
import json
import os
from collections import defaultdict
from glob import glob
from multiprocessing import cpu_count
from typing import List

from PIL import Image
from tqdm import tqdm

from sae_auto_interp.clients import OpenAIClient
from sae_auto_interp.prompt import GPT_CONSISTENCY_PROMPT
from sae_auto_interp.utils import load_explanation

API_ENDPOINT = os.getenv("OPENAI_API_ENDPOINT")
API_TYPE = os.getenv("OPENAI_API_TYPE")
API_KEY = os.getenv("OPENAI_API_KEY")
API_VERSION = os.getenv("OPENAI_API_VERSION")
DEPLOYMENT_NAME = os.getenv("OPENAI_DEPLOYMENT_NAME")


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--explanation-dir", type=str, help="Path to your explanation")
    parser.add_argument("--label-file", type=str, help="Path to your label file")
    parser.add_argument(
        "--save-path",
        type=str,
        help="Path to save the result. Will also use it as the cache",
    )
    return parser.parse_args()


def _process_image(feature):
    image_folder = os.path.join(
        args.explanation_dir, "images", "model_layers_24", feature, "activated_images"
    )
    image_files = glob(os.path.join(image_folder, "*.*"))
    images = [Image.open(image) for image in image_files]
    return images


def _prepare_messages(images: List[Image.Image], feature: str):
    messages = [
        {
            "role": "user",
            "content": [],
        }
    ]

    for image in images:
        messages[0]["content"].append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{client.encode_images(image)}"
                },
            },
        )

    messages[0]["content"].append(
        {
            "type": "text",
            "text": GPT_CONSISTENCY_PROMPT.format(explanation=explanation[feature]),
        },
    )
    return messages


if __name__ == "__main__":
    args = parse_argument()

    # load explanation
    explanation = load_explanation(args.explanation_dir)

    # load label file
    with open(args.label_file, "r") as f:
        label_file = json.load(f)

    label_dict = defaultdict(list)
    for feature, label in label_file.items():
        label_dict[label].append(feature)

    # Fixing the sampled features to be the first 100 features for each label
    sampled_features = []
    for label, features in label_dict.items():
        sampled_features.extend(features[:100])

    if os.path.exists(args.save_path):
        consistency_score = json.load(open(args.save_path, "r"))
    else:
        consistency_score = {}

    # load client
    client = OpenAIClient(
        model="gpt-4o",
        api_type=API_TYPE,
        api_endpoint=API_ENDPOINT,
        api_version=API_VERSION,
        api_key=API_KEY,
        deployment_name=DEPLOYMENT_NAME,
        timeout=600,
    )

    os.makedirs(args.save_path.rsplit("/", 1)[0], exist_ok=True)

    async def _process():
        sem = asyncio.Semaphore(1)

        async def _generate(feature):
            images = _process_image(feature)
            messages = _prepare_messages(images, feature)
            async with sem:
                try:
                    result = await client.generate(messages)
                except Exception as e:
                    result = -1
                return feature, result

        tasks = [
            asyncio.create_task(_generate(feature))
            for feature in sampled_features
            if feature not in consistency_score
        ]

        pbar = tqdm(total=len(tasks), desc="Collected...")
        for completed_task in asyncio.as_completed(tasks):
            feature, result = await completed_task
            consistency_score[feature] = result
            json.dump(consistency_score, open(args.save_path, "w"), indent=4)
            pbar.update(1)

    asyncio.run(_process())
