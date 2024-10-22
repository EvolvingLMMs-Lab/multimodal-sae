from dataclasses import dataclass
from typing import List, Tuple, TypeVar, Union

import blobfile as bf
import numpy as np
import orjson
import torch
from PIL import Image
from torch.nn.functional import interpolate
from torchtyping import TensorType
from torchvision.transforms.functional import to_pil_image
from transformers import AutoProcessor


@dataclass
class Example:
    tokens: TensorType["seq"]
    activations: TensorType["seq"]

    def __hash__(self) -> int:
        return hash(tuple(self.tokens.tolist()))

    def __eq__(self, other: "Example") -> bool:
        return self.tokens.tolist() == other.tokens.tolist()

    @property
    def max_activation(self):
        return max(self.activations)


@dataclass
class ImageExample(Example):
    image: Image
    activation_image: Image
    mask: Image


ExampleType = TypeVar("ExampleType", bound=Union[Example, ImageExample])


def prepare_examples(tokens, activations):
    return [
        Example(
            tokens=toks,
            activations=acts,
        )
        for toks, acts in zip(tokens, activations)
    ]


def prepare_image_examples(tokens, activations, images, processor: AutoProcessor):
    # TODO : This is a hacky way to get the image tokens
    # I will change it to a better way after the new transformers release
    # After they release the new version with llava_ov, the processor class
    # has more utils
    # TODO: Currently only tries to get the activations for the base image feat
    # probably later try on how to get activations on unpadded image features
    # Possibly have to wait till the new transformers release
    base_img_tokens = 576
    patch_size = 24

    base_image_activations = [
        acts[:base_img_tokens].view(patch_size, patch_size) for acts in activations
    ]

    upsampled_image_mask = [
        upsample_mask(acts, (336, 336)) for acts in base_image_activations
    ]

    background = Image.new("L", (336, 336), 0).convert("RGB")

    # Somehow as I looked closer into the llava-hf preprocessing code,
    # I found out that they don't use the padded image as the base image feat
    # but use the simple resized image. This is different from original llava but
    # we align to llava-hf for now as we use llava-hf
    resized_image = [im.resize((336, 336)) for im in images]
    activation_images = [
        Image.composite(background, im, upsampled_mask).convert("RGB")
        for im, upsampled_mask in zip(resized_image, upsampled_image_mask)
    ]

    return [
        ImageExample(
            tokens=toks,
            activations=acts,
            image=image,
            activation_image=activation_image,
            mask=mask,
        )
        for toks, acts, image, activation_image, mask in zip(
            tokens, activations, images, activation_images, upsampled_image_mask
        )
    ]


@dataclass
class Feature:
    module_name: int
    feature_index: int

    def __repr__(self) -> str:
        return f"{self.module_name}_feature{self.feature_index}"


class FeatureRecord:
    def __init__(
        self,
        feature: Feature,
    ):
        self.feature = feature
        self.train: List[ExampleType] = None
        self.explanation: str = None
        self.examples: List[ExampleType] = None

    @property
    def max_activation(self):
        return self.examples[0].max_activation

    def save(self, directory: str, save_examples=False):
        path = f"{directory}/{self.feature}.json"
        serializable = self.__dict__

        if not save_examples:
            serializable.pop("examples")
            serializable.pop("train")
            serializable.pop("test")

        serializable.pop("feature")
        with bf.BlobFile(path, "wb") as f:
            f.write(orjson.dumps(serializable))


def upsample_mask(
    mask: torch.Tensor, image_size: Tuple[int, int], value=224, mode=Image.BILINEAR
) -> Image.Image:
    mask = (mask < 1e-5).int().numpy() * value
    mask_image = Image.fromarray(mask.astype(np.uint8), mode="L")
    upsampled_mask = mask_image.resize(image_size, mode)
    return upsampled_mask
