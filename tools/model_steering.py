import argparse

import torch
import transformers
from PIL import Image
from transformers import AutoTokenizer

from sae_auto_interp.sae import Sae
from sae_auto_interp.utils import (
    get_llava_image_pos,
    load_single_sae,
    maybe_load_llava_model,
)

transformers.logging.set_verbosity_error()


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
        "--image-path", "-i", type=str, help="The path to your image", default=None
    )
    parser.add_argument(
        "--text", "-t", type=str, help="The text you want to ask the model"
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
        "--clamp-value", "-k", type=float, default=10, help="The clamping value"
    )
    parser.add_argument(
        "--feature_idx",
        "-f",
        type=int,
        help="The idx of the feature that you want to clamp",
    )
    return parser.parse_args()


def clamp_features_max(
    sae: Sae, feature: int, hooked_module: torch.nn.Module, k: float = 10
):
    def hook(module: torch.nn.Module, _, outputs):
        # Maybe unpack tuple outputs
        if isinstance(outputs, tuple):
            unpack_outputs = list(outputs)
        else:
            unpack_outputs = list(outputs)
        latents = sae.pre_acts(unpack_outputs[0])
        # Only clamp the feature for the first forward
        if latents.shape[1] != 1:
            latents[:, :, feature] = k
        top_acts, top_indices = sae.select_topk(latents)
        sae_out = sae.decode(top_acts[0], top_indices[0]).unsqueeze(0).to(torch.float16)
        unpack_outputs[0] = sae_out
        if isinstance(outputs, tuple):
            outputs = tuple(unpack_outputs)
        else:
            outputs = unpack_outputs[0]
        return outputs

    handles = [hooked_module.register_forward_hook(hook)]

    return handles


if __name__ == "__main__":
    args = parse_argument()
    feature_idx: int = args.feature_idx
    sae = load_single_sae(args.sae_path, args.module_name)
    model, processor = maybe_load_llava_model(
        args.model, rank=0, dtype=torch.float16, hf_token=None
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    image = Image.open(args.image_path) if args.image_path is not None else None
    text: str = args.text
    hooked_module = model.language_model.get_submodule(args.module_name)

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
            ],
        },
    ]
    if image is not None:
        conversation[0]["content"].append(
            {"type": "image"},
        )

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
    # image_tokens = tokenizer.convert_tokens_to_ids("<image>")
    # prev, after = get_llava_image_pos(inputs["input_ids"][0].tolist(), image_tokens)
    # processor.patch_size = 14
    # image_embed_size = processor._get_number_of_features(image.size[0], image.size[1], 336, 336)
    # after = prev + image_embed_size
    print(" ===========  Original ===========")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=512)
    cont = output[:, inputs["input_ids"].shape[-1] :]
    print(processor.batch_decode(cont, skip_special_tokens=True)[0])

    print(" ===========  Steering ===========")

    handles = clamp_features_max(sae, feature_idx, hooked_module, k=args.clamp_value)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=512)
    cont = output[:, inputs["input_ids"].shape[-1] :]
    print(processor.batch_decode(cont, skip_special_tokens=True)[0])
