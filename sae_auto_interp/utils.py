import json
import os
from typing import Dict, List, Tuple, Union

import torch
from PIL import Image
from torchtyping import TensorType
from transformers import (
    AutoModel,
    AutoTokenizer,
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
    QuantoConfig,
)
from transformers.image_processing_utils import select_best_resolution

from .features import FeatureRecord
from .sae import Sae


def load_tokenized_data(
    ctx_len: int,
    tokenizer: AutoTokenizer,
    dataset_repo: str,
    dataset_split: str,
    dataset_name: str = "",
    seed: int = 22,
):
    """
    Load a huggingface dataset, tokenize it, and shuffle.
    """
    from datasets import load_dataset
    from transformer_lens import utils

    data = load_dataset(dataset_repo, name=dataset_name, split=dataset_split)

    tokens = utils.tokenize_and_concatenate(data, tokenizer, max_length=ctx_len)

    tokens = tokens.shuffle(seed)["tokens"]

    return tokens


def load_filter(path: str, device: str = "cuda:0"):
    with open(path) as f:
        filter = json.load(f)

    return {key: torch.tensor(value, device=device) for key, value in filter.items()}


def load_explanation(explanation_dir: str) -> Dict[str, str]:
    explanations = {}
    explanation_files = os.listdir(explanation_dir)
    explanation_files = [
        e for e in explanation_files if os.path.isfile(os.path.join(explanation_dir, e))
    ]
    for file in explanation_files:
        with open(os.path.join(explanation_dir, file), "r") as f:
            data = json.load(f)

        for da in data:
            for key_name, content in da.items():
                if key_name != "prompt":
                    explanations[key_name] = content
    return explanations


def maybe_load_llava_model(
    model_name, rank, dtype, hf_token
) -> Tuple[Union[AutoModel, LlavaNextForConditionalGeneration], LlavaNextProcessor]:
    if "llava" in model_name:
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name,
            device_map={"": f"cuda:{rank}"},
            torch_dtype=dtype,
            token=hf_token,
        )
        processor = LlavaNextProcessor.from_pretrained(model_name)
    else:
        model = AutoModel.from_pretrained(
            model_name,
            device_map={"": f"cuda:{rank}"},
            torch_dtype=dtype,
            token=hf_token,
        )
        processor = None

    return model, processor


def load_llava_quantized(
    model_name,
    rank,
) -> Tuple[Union[AutoModel, LlavaNextForConditionalGeneration], LlavaNextProcessor]:
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_name,
        device_map={"": f"cuda:{rank}"},
        quantization_config=(QuantoConfig(weights="float8")),
        torch_dtype=torch.float16,
    )
    processor = LlavaNextProcessor.from_pretrained(model_name)

    return model, processor


def load_saes(
    sae_path: str, filters: Dict[str, TensorType["indices"]] = None, device="cuda:0"
) -> Dict[str, Sae]:
    submodule_dict = {}
    if os.path.exists(sae_path):
        if filters is not None:
            for module_name, indices in filters.items():
                sae = Sae.load_from_disk(
                    os.path.join(sae_path, module_name), device=device
                )
                submodule_dict[module_name] = sae
        else:
            submodule_dict = Sae.load_many(sae_path, local=True, device=device)
    else:
        if filters is not None:
            for module_name, indices in filters.items():
                sae = Sae.load_from_hub(sae_path, module_name, device=device)
                submodule_dict[module_name] = sae
        else:
            submodule_dict = Sae.load_many(sae_path, local=False, device=device)

    return submodule_dict


def load_single_sae(sae_path: str, module_name: str, device="cuda:0") -> Sae:
    if os.path.exists(sae_path):
        sae = Sae.load_from_disk(os.path.join(sae_path, module_name), device=device)
    else:
        sae = Sae.load_from_hub(sae_path, module_name, device=device)
    return sae


def get_anyres_padded_images(
    image: Image.Image,
    image_grid_pinpoints: List[List[int]],
):
    height_best_resolution, width_best_resolution = select_best_resolution(
        [image.size[0], image.size[1]], image_grid_pinpoints
    )

    return image.resize((height_best_resolution, width_best_resolution))


def get_anyres_unpadded_size(
    orig_height: int,
    orig_width: int,
    height: int,
    width: int,
    image_grid_pinpoints: List[List[int]],
    patch_size: int,
):
    height_best_resolution, width_best_resolution = select_best_resolution(
        [orig_height, orig_width], image_grid_pinpoints
    )

    scale_height, scale_width = (
        height_best_resolution // height,
        width_best_resolution // width,
    )

    patches_height = height // patch_size
    patches_width = width // patch_size
    current_height = patches_height * scale_height
    current_width = patches_width * scale_width

    original_aspect_ratio = width / height
    current_aspect_ratio = current_width / current_height
    if original_aspect_ratio > current_aspect_ratio:
        new_height = (height * current_width) // width
        padding = (current_height - new_height) // 2
        current_height -= padding * 2
    else:
        new_width = (width * current_height) // height
        padding = (current_width - new_width) // 2
        current_width -= padding * 2

    # This will be the size of the unpadded anyres image tokens
    # There will be image newline at the end of each line
    return current_height, current_width


def get_llava_image_pos(input_ids: List[int], image_tok: int) -> Tuple[int, int]:
    """
    This is a simple split operation to find only image tokens from
    input ids. Only works for single image tok now :D
    """
    # Find the place of image token
    image_pos = input_ids.index(image_tok)
    # Image embed start from the image token location
    prev = image_pos
    # From the next token to the end will be text
    after = -(len(input_ids) - image_pos) + 1
    return prev, after


def display(
    record: FeatureRecord, tokenizer: AutoTokenizer, threshold: float = 0.0, n: int = 10
) -> str:
    from IPython.core.display import HTML, display

    def _to_string(tokens: TensorType["seq"], activations: TensorType["seq"]) -> str:
        result = []
        i = 0

        max_act = max(activations)
        _threshold = max_act * threshold

        while i < len(tokens):
            if activations[i] > _threshold:
                result.append("<mark>")
                while i < len(tokens) and activations[i] > _threshold:
                    result.append(tokens[i])
                    i += 1
                result.append("</mark>")
            else:
                result.append(tokens[i])
                i += 1
        return "".join(result)

    strings = [
        _to_string(tokenizer.batch_decode(example.tokens), example.activations)
        for example in record.examples[:n]
    ]

    display(HTML("<br><br>".join(strings)))


def load_tokenizer(model):
    """
    Loads tokenizer to the default NNsight configuration.
    """

    tokenizer = AutoTokenizer.from_pretrained(model, padding_side="left")
    tokenizer._pad_token = tokenizer._eos_token

    return tokenizer
