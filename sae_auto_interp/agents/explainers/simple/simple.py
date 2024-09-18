import re

import torch

from ...utils import highlight, join_activations, normalize_examples
from ..explainer import Explainer, ExplainerResult
from .prompt_builder import build_prompt


class SimpleExplainer(Explainer):
    name = "Simple"

    def __init__(
        self,
        client,
        tokenizer,
        verbose: bool = False,
        cot: bool = False,
        logits: bool = False,
        activations: bool = False,
        threshold: float = 0.6,
        **generation_kwargs,
    ):
        self.client = client
        self.tokenizer = tokenizer
        self.verbose = verbose

        self.cot = cot
        self.logits = logits
        self.activations = activations

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
                match.group(1).strip() if match else "Explanation could not be parsed."
            )
        except Exception:
            return "Explanation could not be parsed."

    def _highlight(self, index, example):
        return highlight(index, example, self.tokenizer, self.threshold)

    def _join_activations(self, example):
        return join_activations(example, self.threshold)

    def _build_prompt(self, examples, top_logits):
        highlighted_examples = []

        for i, example in enumerate(examples):
            highlighted_examples.append(self._highlight(i + 1, example))

            if self.activations:
                highlighted_examples.append(self._join_activations(example))

        highlighted_examples = "\n".join(highlighted_examples)

        return build_prompt(
            examples=highlighted_examples,
            cot=self.cot,
            activations=self.activations,
            top_logits=top_logits,
        )
