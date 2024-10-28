import argparse
import json
import os

import torch
import torch.distributed as dist
import transformers
from PIL import Image
from transformers import AutoTokenizer

from sae_auto_interp.features.steering import SteeringController
from sae_auto_interp.sae import Sae
from sae_auto_interp.utils import load_filter, load_saes, maybe_load_llava_model

transformers.logging.set_verbosity_error()


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="llava-hf/llama3-llava-next-8b-hf",
        help="The model name of your trained model",
    )
    parser.add_argument(
        "--image-path", "-i", type=str, help="The path to your image", default=None
    )
    parser.add_argument(
        "--text", "-t", type=str, help="The text you want to ask the model"
    )
    parser.add_argument(
        "--sae-path", type=str, help="The path to your sae, can be hub or local"
    )
    parser.add_argument(
        "--filters",
        type=str,
        help="The filters path",
    )
    parser.add_argument(
        "--clamp-value", "-k", type=float, default=50, help="The clamping value"
    )
    parser.add_argument(
        "--save-dir",
        "-s",
        default="./results/steering",
        help="The path to save your steering result",
    )
    return parser.parse_args()


def main():
    args = parse_argument()
    local_rank = os.environ.get("LOCAL_RANK")
    ddp = local_rank is not None
    rank = int(local_rank) if ddp else 0

    if ddp:
        torch.cuda.set_device(int(local_rank))
        dist.init_process_group("nccl")

    model, processor = maybe_load_llava_model(
        args.model, rank=rank, dtype=torch.float16, hf_token=None
    )

    filters = load_filter(args.filters, device="cpu")
    sae_dict = load_saes(args.sae_path, filters, device=f"cuda:{rank}")
    for module_name, module in sae_dict.items():
        if ddp:
            feature_idx = (
                filters[module_name]
                .tensor_split(dist.get_world_size())[rank]
                .cpu()
                .tolist()
            )
        else:
            feature_idx = filters[module_name].tolist()
        steering_controller = SteeringController(
            sae=module,
            module_name=module_name,
            feature_idx=feature_idx,
            prompt=args.text,
            model=model,
            processor=processor,
            image_path=args.image_path,
            k=args.clamp_value,
        )

        result_dict = steering_controller.run()

        if ddp:
            all_dict = [None for _ in range(dist.get_world_size())]
            dist.gather_object(result_dict, all_dict if rank == 0 else None, dst=0)
            if rank == 0:
                for result in all_dict:
                    result_dict.update(result)

        if rank == 0:
            os.makedirs(args.save_dir, exist_ok=True)
            with open(
                os.path.join(args.save_dir, f"{module_name}.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(result_dict, f, indent=4, ensure_ascii=False)

        if ddp:
            dist.barrier()


if __name__ == "__main__":
    main()
