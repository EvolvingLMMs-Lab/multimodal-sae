import random
from collections import deque
from typing import List, Literal

from ..config import ExperimentConfig
from .features import Example, FeatureRecord


def split_activation_quantiles(
    examples: List[Example], n_quantiles: int, n_samples: int, seed: int = 22
):
    random.seed(seed)

    max_activation = examples[0].max_activation
    thresholds = [max_activation * i / n_quantiles for i in range(1, n_quantiles)]

    samples = []
    examples = deque(examples)

    for threshold in thresholds:
        quantile = []
        while examples and examples[0].max_activation < threshold:
            quantile.append(examples.popleft())

        sample = random.sample(quantile, n_samples)
        samples.append(sample)

    sample = random.sample(examples, n_samples)
    samples.append(sample)

    return samples


def split_quantiles(
    examples: List[Example], n_quantiles: int, n_samples: int, seed: int = 22
):
    random.seed(seed)

    quantile_size = len(examples) // n_quantiles

    samples = []

    for i in range(n_quantiles):
        quantile = examples[i * quantile_size : (i + 1) * quantile_size]

        sample = random.sample(quantile, n_samples)
        samples.append(sample)

    return samples


def train(
    examples: List[Example],
    n_train: int,
    train_type: Literal["top", "random", "quantile"],
    seed: int = 22,
    n_quantiles: int = 10,
    chosen_quantile: int = 0,
):
    if train_type == "top":
        return examples[:n_train]
    elif train_type == "random":
        random.seed(seed)
        return random.sample(examples, n_train)
    elif train_type == "quantile":
        return split_quantiles(examples, n_quantiles, n_train)[chosen_quantile]
    else:
        raise ValueError(f"Invalid train_type: {train_type}")


def test(
    examples: List[Example],
    n_test: int,
    n_quantiles: int,
    test_type: Literal["even", "activation"],
):
    if test_type == "even":
        return split_quantiles(examples, n_quantiles, n_test)
    elif test_type == "activation":
        return split_activation_quantiles(examples, n_quantiles, n_test)
    else:
        raise ValueError(f"Invalid test_type: {test_type}")


def sample(
    record: FeatureRecord,
    cfg: ExperimentConfig,
):
    examples = record.examples

    _train = train(
        examples,
        cfg.n_examples_train,
        cfg.train_type,
        cfg.n_quantiles,
        getattr(cfg, "chosen_quantile", 0),
    )

    _test = test(
        examples,
        cfg.n_examples_test,
        cfg.n_quantiles,
        cfg.test_type,
    )

    record.train = _train
    record.test = _test
