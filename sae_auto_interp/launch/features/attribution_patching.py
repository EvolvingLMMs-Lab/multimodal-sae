import os

import torch
import torch.distributed as dist
from loguru import logger
from safetensors.torch import save_file
from simple_parsing import parse
from transformers import AutoModelForCausalLM, AutoTokenizer

from sae_auto_interp.config import AttributionConfig
from sae_auto_interp.features import Attribution
from sae_auto_interp.utils import maybe_load_llava_model


def main(cfg: AttributionConfig):
    local_rank = os.environ.get("LOCAL_RANK")
    ddp = local_rank is not None
    rank = int(local_rank) if ddp else 0
    if ddp:
        torch.cuda.set_device(int(local_rank))
        dist.init_process_group("nccl")

        if rank == 0:
            print(f"Using DDP across {dist.get_world_size()} GPUs.")
    model_id = cfg.model

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    logger.info(f"Using model {model_id}")
    logger.info(f"Using sae {cfg.sae_path}")
    logger.info(f"Selecting sae layer {cfg.selected_sae}")
    model, processor = maybe_load_llava_model(
        model_id, rank, "torch.float16", hf_token=None
    )
    attribution = Attribution(
        model,
        tokenizer,
        sae_path=cfg.sae_path,
        data_path=cfg.data_path,
        selected_sae=cfg.selected_sae,
        image_processor=processor.image_processor,
    )
    if ddp:
        saes = [v for v in attribution.sae_dict.values()]
        num_latents = getattr(saes[0].cfg, "num_latens", None)
        k = (
            num_latents
            if num_latents is not None
            else saes[0].d_in * saes[0].cfg.expansion_factor
        )
        indices = torch.arange(k).chunk(dist.get_world_size())[rank]
        dist.barrier()
        attribution_dict = attribution.get_attribution(indices)
        all_attribution_dict = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(all_attribution_dict, attribution_dict)
        if rank == 0:
            for idx, attribution in enumerate(all_attribution_dict):
                # Skip the rank0 one
                if idx == 0:
                    continue
                for module_name, attribution in attribution.items():
                    attribution_dict[module_name].extend(attribution)
        dist.barrier()
    else:
        attribution_dict = attribution.get_attribution()

    if rank == 0:
        logger.info("Save results")
        attribution_dict = {
            k: torch.concatenate(v, dim=0) for k, v in attribution_dict.items()
        }
        os.makedirs(cfg.save_dir, exist_ok=True)
        output_file = os.path.join(
            cfg.save_dir,
            f"{model_id.split('/')[-1]}_{cfg.selected_sae.replace('.','_')}.safetensors",
        )
        logger.info(output_file)
        save_file(attribution_dict, output_file)


if __name__ == "__main__":
    args = parse(AttributionConfig)
    main(args)
