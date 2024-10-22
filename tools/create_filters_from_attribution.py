import argparse
import json
import os

from PIL import Image
from safetensors.torch import load_file
from torch.nn.functional import avg_pool1d, max_pool1d
from transformers import AutoTokenizer

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
        action="store_false",
        default=True,
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


if __name__ == "__main__":
    args = parse_args()
    attribution = load_file(args.attribution_path)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    probing_data = json.load(open(args.probing_data, "r"))[0]
    prompt = probing_data["prompt"]
    image = Image.open(probing_data["image"])
    tokens = tokenizer(prompt, return_tensors="pt")["input_ids"][:, 1:].tolist()[0]
    image_token = tokenizer.convert_tokens_to_ids("<image>")
    # Find the place that insert image embedding to
    prev, after = get_llava_image_pos(tokens, image_token)
    if args.pool == "avg":
        pool = avg_pool1d
    elif args.pool == "max":
        pool = max_pool1d

    filters_dict = {}
    for module_name, attribution_act in attribution.items():
        if args.image_only:
            attribution_act = attribution_act[:, prev:after]
        pooled_attribution = pool(
            attribution_act,
            kernel_size=attribution_act.shape[1],
            stride=attribution_act.shape[1],
        ).squeeze(1)
        top_k_indices = pooled_attribution.topk(k=args.top_k).indices.flatten().tolist()
        filters_dict[module_name] = top_k_indices

    name = f"filters_top_{args.top_k}_{args.probing_data.split('/')[-1].split('.')[0]}.json"
    with open(os.path.join(args.filters_path, name), "w") as f:
        json.dump(filters_dict, f, indent=4)
