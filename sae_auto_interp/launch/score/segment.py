import argparse
import json
import os

import torch
import torch.distributed as dist

from sae_auto_interp.agents.scorers import LabelRefiner, SegmentScorer
from sae_auto_interp.clients import SRT
from sae_auto_interp.utils import load_filter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--detector",
        type=str,
        default="IDEA-Research/grounding-dino-base",
        help="The detector you use",
    )
    parser.add_argument(
        "--segmentor",
        type=str,
        default="facebook/sam-vit-huge",
        help="The segmentor you use",
    )
    parser.add_argument(
        "--explanation_dir", type=str, help="The place where you store you explanation"
    )
    parser.add_argument(
        "--filters", type=str, help="Path to your filters", default=None
    )
    parser.add_argument(
        "--refine-cache",
        type=str,
        default=None,
        help="The path to your previous refined explanation",
    )
    parser.add_argument(
        "--save-path", type=str, help="The path to save your refine explanations"
    )
    parser.add_argument(
        "--selected-layer",
        type=str,
        help="The layer of the model to be evaluated on, such as `model.layers.24`",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    local_rank = os.environ.get("LOCAL_RANK")
    ddp = local_rank is not None
    rank = int(local_rank) if ddp else 0

    if args.filters is not None:
        filters = load_filter(args.filters)
        filters = filters[args.selected_layer]
    else:
        filters = None

    if ddp:
        torch.cuda.set_device(int(local_rank))
        dist.init_process_group("nccl")
    scorer = SegmentScorer(
        explanation_dir=args.explanation_dir,
        detector=args.detector,
        segmentor=args.segmentor,
        device=f"cuda:{rank}",
        filters=filters,
    )

    if args.refine_cache is None:
        client = SRT(model="meta-llama/Llama-3.1-8B-Instruct", tp=2)
        refiner = LabelRefiner(client, scorer.filtered_explanation)
        scorer.refine(refiner, save_path=args.save_path)
        client.clean()
    else:
        with open(args.refine_cache):
            scorer.explanation = json.load(open(args.refine_cache, "r"))
    scorer.load_model()
    scorer()
