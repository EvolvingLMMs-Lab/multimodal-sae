import argparse
import os

import torch
import torch.distributed as dist

from sae_auto_interp.agents.scorers import SegmentScorer


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

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    local_rank = os.environ.get("LOCAL_RANK")
    ddp = local_rank is not None
    rank = int(local_rank) if ddp else 0

    if ddp:
        torch.cuda.set_device(int(local_rank))
        dist.init_process_group("nccl")
    scorer = SegmentScorer(
        explanation_dir=args.explanation_dir,
        detector=args.detector,
        segmentor=args.segmentor,
        device=f"cuda:{rank}",
    )

    scorer()
