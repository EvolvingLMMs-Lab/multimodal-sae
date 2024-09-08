import re
from ast import literal_eval
from typing import Any, List, NamedTuple

import torch
from transformers import PreTrainedTokenizer

from sae_auto_interp.clients.client import Client
from sae_auto_interp.features.features import Example, FeatureRecord

from ...utils import highlight, join_activations, normalize_examples
from .prompt import prompt


class SimpleScorerResult(NamedTuple):
    record: FeatureRecord
    """Feature record passed through."""

    scores: Any
    """Generated score for feature."""


class SimpleScorer:
    name = "Simple"

    def __init__(
        self,
        client: Client,
        tokenizer: PreTrainedTokenizer,
        verbose: bool = False,
        threshold: float = 0.6,
        activations: bool = False,
        **generation_kwargs,
    ):
        self.client = client
        self.tokenizer = tokenizer
        self.verbose = verbose

        self.threshold = threshold
        self.activations = activations
        self.generation_kwargs = generation_kwargs

    def _normalize_examples(self, record, train):
        normalize_examples(record, train)

    async def __call__(self, record: FeatureRecord):
        if self.activations:
            self._normalize_examples(record, record.train)

        splited_examples = [
            record.train[i : i + 5] for i in range(0, len(record.train), 5)
        ]

        scores_list = []
        messages_list = []
        response_list = []
        for examples in splited_examples:
            messages = self._build_prompt(examples, record.explanation)
            response = await self.client.generate(messages, **self.generation_kwargs)
            scores = self.parse_scores(response)
            try:
                scores = literal_eval(scores)
                scores_list.extend(scores)
                messages_list.append(messages[-1]["content"])
                response_list.append(response)
            except Exception as e:
                # Probably some format does not match
                # Let's just keep continue, try different prompt
                # but eventually all don't work
                continue

        result = SimpleScorerResult(record=record, scores=scores_list)

        if self.verbose:
            return (
                messages_list,
                response_list,
                result,
            )

        return result

    def parse_scores(self, text: str) -> str:
        try:
            # Return the first list find
            match = re.search(r"\[.*\]", text, re.DOTALL)

            return match.group(0).strip() if match else "Scores could not be parsed."
        except Exception:
            return "Scores could not be parsed."

    def _highlight(self, index, example):
        return highlight(index, example, self.tokenizer, self.threshold)

    def _join_activations(self, example):
        return join_activations(example, self.threshold)

    def _build_prompt(self, train_examples: List[Example], explanation: str):
        higlighted_examples = []
        for idx, example in enumerate(train_examples):
            higlighted_examples.append(self._highlight(idx, example))

            if self.activations:
                higlighted_examples.append(self._join_activations(example))

        higlighted_examples = "\n".join(higlighted_examples)

        return prompt(examples=higlighted_examples, explanation=explanation)
