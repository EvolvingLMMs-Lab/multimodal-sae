from dataclasses import dataclass
from typing import Literal, Union

from simple_parsing import Serializable, field, list_field


@dataclass
class ExperimentConfig(Serializable):
    model: str = "EleutherAI/pythia-160m"
    """Name of the model to use when training sae."""

    dataset: str = ("togethercomputer/RedPajama-Data-1T-Sample",)
    """Path to the dataset."""

    sae_path: Union[str, None] = None
    """Path to your trained sae, Should be local"""

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

    train_type: Literal["top", "random", "quantile"] = "top"
    """Type of sampler to use for training"""

    explainer: str = "meta-llama/Meta-Llama-3.1-405B-Instruct-FP8"
    """The name of the explainer model"""

    explanation_dir: str = "./explanation_dir"
    """Dir to save your explanation result"""

    scores_dir: str = "./scores_dir"
    """Dir to save your scores result"""

    selected_layers: list[int] = list_field()

    split: str = "train"
    """Dataset split to use."""

    save_dir: str = "./features_cache"
    """Save dir for your previous cached feature"""

    filters_path: str = None
    """The json file for filtering the features and sae should be in a json file"""


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

    filters_path: str = None
    """The json file for filtering the features and sae should be in a json file"""


@dataclass
class AttributionConfig(Serializable):
    model: str = field(
        default="EleutherAI/pythia-160m",
        positional=True,
    )
    """Name of the model to use."""

    data_path: str = "./data/digit.json"
    """Path to the dataset. Should be a formated json file"""

    sae_path: Union[str, None] = None
    """Path to your trained sae, can be either local or on the hub"""

    selected_sae: str = "layers.24"
    """Name of the selected sae"""

    save_dir: str = "./attribution_cache"
    """Save dir for your feature attribution result"""
