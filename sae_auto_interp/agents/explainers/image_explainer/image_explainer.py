import base64
import re
from io import BytesIO

import torch
from PIL import Image
from transformers import PreTrainedTokenizer

from ....clients import SRT
from ....features import FeatureRecord
from ...utils import highlight, join_activations, normalize_examples
from ..explainer import Explainer, ExplainerResult
from .prompts import build_prompt


class ImageExplainer(Explainer):
    name = "Simple"

    def __init__(
        self,
        client: SRT,
        verbose: bool = False,
        **generation_kwargs,
    ):
        self.client = client
        self.verbose = verbose
        self.generation_kwargs = generation_kwargs

    async def __call__(self, record: FeatureRecord):
        images = [train.image for train in record.train]
        encoded_images = [self.encode_images(image) for image in images]
        messages = build_prompt(encoded_images)
        response = await self.client.generate(messages, **self.generation_kwargs)

        explanation = self.parse_explanation(response)

        if self.verbose:
            return (
                messages[-1]["content"],
                response,
                ExplainerResult(record=record, explanation=explanation),
            )

        return ExplainerResult(record=record, explanation=explanation)

    def parse_explanation(self, text: str) -> str:
        try:
            match = re.search(r"\[EXPLANATION\]:\s*(.*)", text, re.DOTALL)

            return (
                match.group(1).strip()
                if match
                else f"Response {text}. Explanation could not be parsed."
            )
        except Exception:
            return f"Response {text}. Explanation could not be parsed."

    def encode_images(
        self,
        image: Image,
    ):
        output_buffer = BytesIO()
        image.save(output_buffer, format="PNG")
        byte_data = output_buffer.getvalue()
        base64_str = base64.b64encode(byte_data).decode("utf-8")
        return base64_str
