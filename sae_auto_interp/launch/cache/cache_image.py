import datetime
import json
import os

import torch
import torch.distributed as dist
from datasets import load_dataset
from loguru import logger
from simple_parsing import parse
from transformers import (
    AutoModel,
    AutoTokenizer,
    InstructBlipForConditionalGeneration,
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
)

from sae_auto_interp.config import CacheConfig
from sae_auto_interp.features import FeatureImageCache
from sae_auto_interp.sae import Sae
from sae_auto_interp.sae.data import chunk_and_tokenize
from sae_auto_interp.utils import load_filter, load_saes, maybe_load_llava_model


def main(cfg: CacheConfig):
    local_rank = os.environ.get("LOCAL_RANK")
    ddp = local_rank is not None
    rank = int(local_rank) if ddp else 0

    if ddp:
        torch.cuda.set_device(int(local_rank))
        dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=18000))

        if rank == 0:
            print(f"Using DDP across {dist.get_world_size()} GPUs.")

    if torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = "auto"

    logger.info(f"Load Model : {cfg.model}")

    model, processor = maybe_load_llava_model(cfg.model, rank, dtype, cfg.hf_token)

    if isinstance(model, InstructBlipForConditionalGeneration):
        tokenizer = processor.tokenizer
        tokenizer.chat_template = "{% for message in messages %}{% if message['role'] != 'system' %}{{ message['role'].upper() + ': '}}{% endif %}{# Render all images first #}{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}{{ '<image>\n' }}{% endfor %}{# Render all text next #}{% if message['role'] != 'assistant' %}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{{ content['text'] + ' '}}{% endfor %}{% else %}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{% generation %}{{ content['text'] + ' '}}{% endgeneration %}{% endfor %}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}"
    else:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model, token=cfg.hf_token)

    logger.info(f"Load Dataset : {cfg.dataset}")

    dataset = load_dataset(
        cfg.dataset,
        split=cfg.split,
        # TODO: Maybe set this to False by default? But RPJ requires it.
        trust_remote_code=True,
    )
    if cfg.filters_path is not None:
        filters = load_filter(cfg.filters_path, device=model.device)
    else:
        filters = None

    if ddp:
        dist.barrier()
        # Make sure the dataset is splitted into contiguous chunk
        dataset = dataset.shard(dist.get_world_size(), rank, contiguous=True)
        all_shards_len = torch.zeros(
            dist.get_world_size(), dtype=torch.int, device=model.device
        )
        cur_shards_len = torch.tensor(
            len(dataset), dtype=torch.int, device=model.device
        )
        dist.all_gather_into_tensor(all_shards_len, cur_shards_len)
        all_shards_len = all_shards_len.detach().cpu().tolist()

    logger.info(f"Load many sae from : {cfg.sae_path}")
    submodule_dict = load_saes(cfg.sae_path, filters=filters, device=model.device)

    logger.info(f"Select {submodule_dict.keys()}")

    cache = FeatureImageCache(
        model,
        tokenizer,
        submodule_dict,
        batch_size=cfg.batch_size,
        # When it is not ddp, it is just 0
        shard_size=sum(all_shards_len[:rank]) if ddp else 0,
        processor=processor,
        filters=filters,
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
    cfg = parse(CacheConfig)

    main(cfg)
