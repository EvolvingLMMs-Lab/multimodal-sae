from typing import List

import torch
from transformers import PreTrainedTokenizer

from ..features import Example, FeatureRecord


def highlight(
    index: int,
    example: Example,
    tokenizer: PreTrainedTokenizer,
    threshold: float,
) -> str:
    result = f"Example {index}: "

    threshold = example.max_activation * threshold
    str_toks = tokenizer.batch_decode(example.tokens)
    example.str_toks = str_toks
    activations = example.activations

    def check(i):
        return activations[i] > threshold

    i = 0
    while i < len(str_toks):
        if check(i):
            result += "<<"

            while i < len(str_toks) and check(i):
                result += str_toks[i]
                i += 1
            result += ">>"
        else:
            result += str_toks[i]
            i += 1

    return "".join(result)


def join_activations(
    example: Example,
    threshold: float,
) -> str:
    activations = []

    threshold = example.max_activation * threshold
    for i, normalized in enumerate(example.normalized_activations):
        if example.activations[i] > threshold:
            activations.append((example.str_toks[i], int(normalized)))

    acts = ", ".join(f'("{item[0]}" : {item[1]})' for item in activations)

    return "Activations: " + acts


def normalize_examples(record: FeatureRecord, train: List[Example]):
    max_activation = record.examples[0].max_activation

    for example in train:
        example.normalized_activations = torch.floor(
            10 * example.activations / max_activation
        )
