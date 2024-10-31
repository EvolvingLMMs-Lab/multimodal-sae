import argparse
import json
import os

import torch
from PIL import Image
from transformers import AutoTokenizer

from sae_auto_interp.features.features import upsample_mask
from sae_auto_interp.utils import load_single_sae, maybe_load_llava_model


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="llava-hf/llama3-llava-next-8b-hf",
        help="The model name of your trained model",
    )
    parser.add_argument(
        "--sae-path", type=str, help="The path to your sae, can be hub or local"
    )
    parser.add_argument(
        "--module-name",
        type=str,
        default="model.layers.24",
        help="The module name of your sae",
    )
    parser.add_argument(
        "--image-path", "-i", type=str, help="The path to your image", default=None
    )
    parser.add_argument(
        "--text",
        "-t",
        type=str,
        help="The text you want to ask the model",
        default=None,
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        help="The top k features you want to probe",
        default=10,
    )
    parser.add_argument(
        "--interval",
        type=str,
        help="The interval of top k features. Split the interval in two parts,"
        "e.g. 1-10,11-20, will probe features from top 1 to 10 and 11 to 20",
        default=None,
    )
    parser.add_argument(
        "--save-to",
        "-s",
        type=str,
        default="./results/probe_activations",
        help="The path to store your stored_activations",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_argument()

    sae = load_single_sae(args.sae_path, args.module_name)
    model, processor = maybe_load_llava_model(
        args.model, rank=0, dtype=torch.float16, hf_token=None
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    device = model.device

    image = Image.open(args.image_path) if args.image_path is not None else None
    text: str = args.text if args.text is not None else None
    hooked_module = model.language_model.get_submodule(args.module_name)
    if args.interval is not None:
        interval = [int(i) for i in args.interval.split("-")]
    else:
        interval = [0, args.top_k]

    assert image is not None or text is not None, "Image and text can no both be None"

    # If text is not None, use chat template
    if text is not None:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                ],
            },
        ]
        # if image is also not None, append it into chat template
        if image is not None:
            conversation[0]["content"].append(
                {"type": "image"},
            )
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    # if only image is not None
    elif image is not None:
        prompt = "<image>"

    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
    topk_indices = None
    topk_acts = None

    def hook(module: torch.nn.Module, _, outputs):
        global topk_indices, topk_acts
        # Maybe unpack tuple outputs
        if isinstance(outputs, tuple):
            unpack_outputs = list(outputs)
        else:
            unpack_outputs = list(outputs)
        latents = sae.pre_acts(unpack_outputs[0])
        # When the tokenizer is llama and text is None (image only)
        # I skip the first bos tokens
        if "llama" in tokenizer.name_or_path and text is None:
            latents = latents[:, 1:, :]

        # avg and get the top-k activated features on the whole image
        topk_indices = (
            latents.squeeze(0).mean(dim=0).topk(k=interval[1]).indices.detach().cpu()
        )[interval[0] :]
        topk_acts = latents[:, :, topk_indices].squeeze(0).permute(1, 0).detach().cpu()

    handles = [hooked_module.register_forward_hook(hook)]
    try:
        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"].to(device),
                pixel_values=inputs["pixel_values"].to(device),
                image_sizes=inputs["image_sizes"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
            )
    finally:
        for handle in handles:
            handle.remove()

    os.makedirs(args.save_to, exist_ok=True)
    filters = {}
    filters[args.module_name] = topk_indices.tolist()
    if image is not None:
        base_img_tokens = 576
        patch_size = 24

        base_image_activations = [
            acts[:base_img_tokens].view(patch_size, patch_size) for acts in topk_acts
        ]

        upsampled_image_mask = [
            upsample_mask(acts, (336, 336)) for acts in base_image_activations
        ]

        background = Image.new("L", (336, 336), 0).convert("RGB")

        # Somehow as I looked closer into the llava-hf preprocessing code,
        # I found out that they don't use the padded image as the base image feat
        # but use the simple resized image. This is different from original llava but
        # we align to llava-hf for now as we use llava-hf
        resized_image = [image.resize((336, 336))] * len(upsampled_image_mask)
        activation_images = [
            Image.composite(background, im, upsampled_mask).convert("RGB")
            for im, upsampled_mask in zip(resized_image, upsampled_image_mask)
        ]

        image_dir = os.path.join(args.save_to, "images")
        os.makedirs(image_dir, exist_ok=True)
        for idx, im in zip(topk_indices, activation_images):
            im.save(os.path.join(image_dir, f"feat_{idx}.png"))

    filters_path = os.path.join(args.save_to, "filters.json")
    with open(filters_path, "w") as f:
        json.dump(filters, f)
