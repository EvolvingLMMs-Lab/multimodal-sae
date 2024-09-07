import re

import torch

from ...utils import highlight, join_activations, normalize_examples


class SimpleScorer:
    name = "Simple"

    def __init__(
        self,
        client,
        tokenizer,
        verbose: bool = False,
        threshold: float = 0.6,
        **generation_kwargs,
    ):
        self.client = client
        self.tokenizer = tokenizer
        self.verbose = verbose

        self.threshold = threshold
        self.generation_kwargs = generation_kwargs

    def _normalize_examples(self, record, train):
        normalize_examples(record, train)

    async def __call__(self, record):
        if self.activations:
            self._normalize_examples(record, record.train)

        if self.logits:
            messages = self._build_prompt(record.train, record.top_logits)
        else:
            messages = self._build_prompt(record.train, None)

        response = await self.client.generate(messages, **self.generation_kwargs)

        explanation = self.parse_explanation(response)

    def parse_explanation(self, text: str) -> str:
        try:
            match = re.search(r"\[EXPLANATION\]:\s*(.*)", text, re.DOTALL)

            return (
                match.group(1).strip() if match else "Explanation could not be parsed."
            )
        except Exception:
            return "Explanation could not be parsed."

    def _highlight(self, index, example):
        return highlight(index, example, self.tokenizer, self.threshold)

    def _join_activations(self, example):
        return join_activations(example, self.threshold)
