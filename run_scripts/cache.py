import datetime
import os
from dataclasses import dataclass
from typing import Union

import torch
import torch.distributed as dist
from datasets import load_dataset
from loguru import logger
from simple_parsing import Serializable, field, parse
from transformers import AutoModel, AutoTokenizer, LlavaNextForConditionalGeneration

from sae_auto_interp.features import FeatureCache
from sae_auto_interp.sae import Sae
from sae_auto_interp.sae.data import chunk_and_tokenize


@dataclass
class RunConfig(Serializable):
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
    """The batch size for the chunked tokens"""

    load_in_8bit: bool = False
    """Load the model in 8-bit mode."""

    split: str = "train"
    """Dataset split to use."""

    n_splits: int = 2
    """Number of splits to divide .safetensors into"""

    ctx_len: int = 2048
    """Context length to use."""

    hf_token: Union[str, None] = None
    """Huggingface API token for downloading models."""

    save_dir: str = "./features_cache"
    """Save dir for your feature"""

    verbosity: str = "INFO"
    """Verbosity level"""


def main(cfg: RunConfig):
    # Probably in the future we should do multi-processing
    # But now I think it's still only one process
    local_rank = os.environ.get("LOCAL_RANK")
    ddp = local_rank is not None
    rank = int(local_rank) if ddp else 0

    if ddp:
        torch.cuda.set_device(int(local_rank))
        dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=18000))

        if rank == 0:
            print(f"Using DDP across {dist.get_world_size()} GPUs.")

    if cfg.load_in_8bit:
        dtype = torch.float16
    elif torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = "auto"

    logger.info(f"Load Model : {cfg.model}")

    if "llava" in cfg.model:
        model = LlavaNextForConditionalGeneration.from_pretrained(
            cfg.model,
            device_map={"": f"cuda:{rank}"},
            torch_dtype=dtype,
            token=cfg.hf_token,
        )
    else:
        model = AutoModel.from_pretrained(
            cfg.model,
            device_map={"": f"cuda:{rank}"},
            torch_dtype=dtype,
            token=cfg.hf_token,
        )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model, token=cfg.hf_token)

    logger.info(f"Load Dataset : {cfg.dataset}")

    dataset = load_dataset(
        cfg.dataset,
        split=cfg.split,
        # TODO: Maybe set this to False by default? But RPJ requires it.
        trust_remote_code=True,
    )

    # Awkward hack to prevent other ranks from duplicating data preprocessing
    if not ddp or rank == 0:
        dataset = chunk_and_tokenize(dataset, tokenizer, max_seq_len=cfg.ctx_len)
    if ddp:
        dist.barrier()
        if rank != 0:
            dataset = chunk_and_tokenize(dataset, tokenizer, max_seq_len=cfg.ctx_len)
        # Make sure the dataset is splitted into contiguous chunk
        dataset = dataset.shard(dist.get_world_size(), rank, contiguous=True)

    logger.info(f"Load many sae from : {cfg.sae_path}")
    if os.path.exists(cfg.sae_path):
        submodule_dict = Sae.load_many(cfg.sae_path, local=True, device=model.device)
    else:
        submodule_dict = Sae.load_many(cfg.sae_path, local=False, device=model.device)

    cache = FeatureCache(
        model,
        tokenizer,
        submodule_dict,
        batch_size=cfg.batch_size,
        shard_size=len(dataset),
    )
    if ddp:
        dist.barrier()
    logger.info("Start caching activations")
    cache.run(cfg.ctx_len, dataset)

    cache.save_splits(n_splits=cfg.n_splits, save_dir=cfg.save_dir, rank=rank)
    if rank == 0:
        cache.concate_safetensors(n_splits=cfg.n_splits, save_dir=cfg.save_dir)

    if ddp:
        dist.barrier()


if __name__ == "__main__":
    cfg = parse(RunConfig)

    main(cfg)
