import argparse
import json
import os

import matplotlib.pyplot as plt
import torch
from PIL import Image
from safetensors.torch import load_file
from torch.nn.functional import avg_pool1d, interpolate, max_pool1d
from transformers import AutoTokenizer

from sae_auto_interp.features.features import upsample_mask
from sae_auto_interp.utils import get_llava_image_pos


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filters-path",
        type=str,
        default="./filters",
        help="The path to save your filters",
    )
    parser.add_argument(
        "--attribution-path",
        type=str,
        default="./attribution_cache",
        help="The path for you attribution",
    )
    parser.add_argument(
        "--top_k",
        "-k",
        default=50,
        type=int,
        help="The top k features you want to pick",
    )
    parser.add_argument(
        "--pool",
        default="avg",
        choices=["max", "avg"],
        help="The pooling method you want to choose",
    )
    parser.add_argument(
        "--image-only",
        action="store_true",
        default=False,
        help="only pick features on image activations or not",
    )
    parser.add_argument(
        "--probing-data",
        type=str,
        help="The path to your probing data used in feature attribution",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="llava-hf/llama3-llava-next-8b-hf",
        help="The tokenizer path",
    )

    return parser.parse_args()


def format_text_rgb(text: str, r: int, g: int, b: int) -> str:
    prefix = f"\033[38;2;{r};{g};{b}m"
    suffix = f"\033[0m"
    return prefix + text + suffix


if __name__ == "__main__":
    args = parse_args()
    attribution = load_file(args.attribution_path)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    probing_data = json.load(open(args.probing_data, "r"))[0]
    prompt = probing_data["prompt"]
    image = Image.open(probing_data["image"])
    tokens = tokenizer(prompt, return_tensors="pt")["input_ids"][:, 1:].tolist()[0]
    tokens_str = [
        tokenizer.convert_tokens_to_string([tokenizer.convert_ids_to_tokens(token)])
        for token in tokens
    ]
    image_token = tokenizer.convert_tokens_to_ids("<image>")
    # Find the place that insert image embedding to
    prev, after = get_llava_image_pos(tokens, image_token)
    completion_tokens = tokens_str[after:]
    # I'm too lazy so I just hardcode this
    if "<|eot_id|>" in completion_tokens:
        completion_tokens = completion_tokens[:-4]
    if args.pool == "avg":
        pool = avg_pool1d
    elif args.pool == "max":
        pool = max_pool1d

    filters_dict = {}
    save_dir = args.attribution_path.rsplit(".", 1)[0]
    os.makedirs(save_dir, exist_ok=True)
    for module_name, attribution_act in attribution.items():
        image_act = attribution_act[:, prev:after]
        text_act = attribution_act[:, after:]
        if "<|eot_id|>" in tokens_str:
            text_act = text_act[:, :-4]  # hardcode remove <|eot_id|>

        image_attribution = pool(
            image_act,
            kernel_size=image_act.shape[1],
            stride=image_act.shape[1],
        ).squeeze(1)
        text_attribution = pool(
            text_act,
            kernel_size=text_act.shape[1],
            stride=text_act.shape[1],
        ).squeeze(1)

        image_top_k_indices = (
            image_attribution.topk(k=args.top_k).indices.flatten().tolist()
        )
        text_top_k_indices = (
            text_attribution.topk(k=args.top_k).indices.flatten().tolist()
        )

        if args.image_only:
            filters_dict[module_name] = image_top_k_indices
        else:
            top_k_indices = image_top_k_indices + text_top_k_indices

        activated_text = ""
        for rank, i in enumerate(text_top_k_indices):
            activations = text_act[i, :]
            # Get the base image attribution
            activated_text += f"feature_{i}: \n"
            activations = activations.clamp(min=0)
            activation_max = max(activations)
            activation_min = min(activations)
            activations = (activations - activation_min) / (
                activation_max - activation_min + 1e-5
            )
            for idx, token in enumerate(completion_tokens):
                activated_text += format_text_rgb(
                    token, int(255 * activations[idx].item()), 0, 0
                )
            activated_text += "\n"
        print(f"{module_name}:\n{activated_text}")

        for rank, i in enumerate(image_top_k_indices):
            image_features = (
                attribution_act[i, prev : prev + 576].view(24, 24).clamp(min=0)
            )
            upsampled_image_mask = upsample_mask(image_features, (336, 336))

            background = Image.new("L", (336, 336), 0).convert("RGB")
            resized_image = image.resize((336, 336))
            activation_images = Image.composite(
                background, resized_image, upsampled_image_mask
            ).convert("RGB")
            activation_images.save(
                os.path.join(save_dir, f"top_{rank}_feature_{i}.png")
            )

    name = f"filters_top_{args.top_k}_{args.probing_data.split('/')[-1].split('.')[0]}.json"
    with open(os.path.join(args.filters_path, name), "w") as f:
        json.dump(filters_dict, f, indent=4)
