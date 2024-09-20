import json
import os
from typing import Dict

import torch
from torchtyping import TensorType
from transformers import AutoTokenizer

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


def load_explanation(explanation_dir: str):
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


def load_saes(
    sae_path: str, filters: Dict[str, TensorType["indices"]] = None, device="cuda:0"
) -> Dict[str, Sae]:
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
