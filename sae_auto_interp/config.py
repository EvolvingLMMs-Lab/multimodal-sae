from dataclasses import dataclass
from typing import Literal, Union

from simple_parsing import Serializable, field


@dataclass
class ExperimentConfig(Serializable):
    train_type: str = "top"
    """Type of sampler to use"""

    n_examples_train: int = 10
    """Number of examples to sample for training"""

    n_examples_test: int = 7
    """Number of examples to sample for testing"""

    n_quantiles: int = 10
    """Number of quantiles to sample"""

    n_random: int = 5
    """Number of random examples to sample"""

    train_type: Literal["top", "random"] = "top"
    """Type of sampler to use for training"""

    test_type: Literal["even", "activation"] = "even"
    """Type of sampler to use for testing"""


@dataclass
class FeatureConfig(Serializable):
    width: int
    """Number of features in the autoencoder"""

    example_ctx_len: int
    """Length of each example."""

    min_examples: int = 200
    """Minimum number of examples for a feature to be included"""

    max_examples: int = 10000
    """Maximum number of examples for a feature to included"""

    n_splits: int = 2
    """Number of splits that features were devided into"""


@dataclass
class CacheConfig(Serializable):
    model: str = field(
        default="EleutherAI/pythia-160m",
        positional=True,
    )
    """Name of the model to use."""

    dataset: str = field(
        default="togethercomputer/RedPajama-Data-1T-Sample",
        positional=True,
    )
    """Path to the dataset."""

    sae_path: Union[str, None] = None
    """Path to your trained sae, can be either local or on the hub"""

    batch_size: int = 32
    """Number of sequences to process in a batch"""

    load_in_8bit: bool = False
    """Load the model in 8-bit mode."""

    split: str = "train"
    """Dataset split to use."""

    n_splits: int = 2
    """Number of splits to divide .safetensors into"""

    ctx_len: int = 2048
    """Context length of the autoencoder. Each batch is shape (batch_size, ctx_len)"""

    hf_token: Union[str, None] = None
    """Huggingface API token for downloading models."""

    save_dir: str = "./features_cache"
    """Save dir for your feature"""

    verbosity: str = "INFO"
    """Verbosity level"""
