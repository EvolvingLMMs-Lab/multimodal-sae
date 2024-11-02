import argparse
import json
import os

import torch
import torch.distributed as dist

from sae_auto_interp.agents.scorers import ClipScorer, LabelRefiner
from sae_auto_interp.clients import SRT
from sae_auto_interp.utils import load_filter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path",
        "-d",
        type=str,
        help="The dataset you use",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="train",
        help="The dataset split you use",
    )
    parser.add_argument(
        "--clip_name_or_path",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="The clip you use",
    )
    parser.add_argument(
        "--refine-cache",
        type=str,
        default=None,
        help="The path to your previous refined explanation",
    )
    parser.add_argument(
        "--save-refine-path", type=str, help="The path to save your refine explanations"
    )
    parser.add_argument(
        "--explanation_dir", type=str, help="The place where you store you explanation"
    )
    parser.add_argument(
        "--save-score-path", type=str, help="The path to save your score"
    )
    parser.add_argument(
        "--evaluation_type",
        "-e",
        type=str,
        default="default",
        choices=["default", "random"],
        help="The evaluation type, default is the top k images, random is random select k images",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    scorer = ClipScorer(
        explanation_dir=args.explanation_dir,
        dataset_path=args.dataset_path,
        dataset_split=args.dataset_split,
        k=5,
        evaluation_type=args.evaluation_type,
        clip_model_name_or_path=args.clip_name_or_path,
        device="cuda",
    )

    if args.refine_cache is None:
        client = SRT(model="meta-llama/Llama-3.1-8B-Instruct", tp=2)
        refiner = LabelRefiner(client, scorer.explanations)
        scorer.refine(refiner, save_path=args.save_refine_path)
        client.clean()
    else:
        with open(args.refine_cache):
            scorer.explanations = json.load(open(args.refine_cache, "r"))

    scores = scorer.run()
    save_dir = "/".join(args.save_score_path.split("/")[:-1])
    os.makedirs(save_dir, exist_ok=True)
    with open(args.save_score_path, "w") as f:
        json.dump(scores, f, indent=4)
