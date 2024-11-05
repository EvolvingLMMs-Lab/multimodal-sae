import argparse
import json
import os

import torch
import torch.distributed as dist
from datasets import load_dataset
from transformers import AutoProcessor

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
        "--eval-type",
        type=str,
        choices=["default", "random"],
        default="default",
        help="Whether topk or randomly sample the images",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        help="The path to the dataset",
        default=None,
    )
    parser.add_argument(
        "--dataset-split", type=str, help="The split of the dataset", default=None
    )
    parser.add_argument(
        "--activation_dir",
        type=str,
        help="The path to your activation cache dir",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="llava-hf/llama3-llava-next-8b-hf",
        help="The path of the base llava model",
    )
    parser.add_argument(
        "--width", type=int, default=131072, help="The width of your sae"
    )
    parser.add_argument(
        "--n-splits",
        "-n",
        type=int,
        help="The n split you set when cache the activations",
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
        "--save-refine-path", type=str, help="The path to save your refine explanations"
    )
    parser.add_argument(
        "--save-score-path", type=str, help="The path to save your score"
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
        filters = filters[args.selected_layer].cpu()
    else:
        filters = None

    if ddp:
        torch.cuda.set_device(int(local_rank))
        dist.init_process_group("nccl")

    tokens = load_dataset(args.dataset_path, split=args.dataset_split)
    processor = AutoProcessor.from_pretrained(args.model_name)
    scorer = SegmentScorer(
        activation_dir=args.activation_dir,
        tokens=tokens,
        processor=processor,
        width=args.width,
        n_splits=args.n_splits,
        explanation_dir=args.explanation_dir,
        detector=args.detector,
        segmentor=args.segmentor,
        device=f"cuda:{rank}",
        filters=filters,
    )

    if args.refine_cache is None:
        if ddp:
            raise RuntimeError(
                "Please refine your description first and use the cache result to do the scoring"
            )
        client = SRT(model="meta-llama/Llama-3.1-8B-Instruct", tp=2)
        refiner = LabelRefiner(client, scorer.filtered_explanation)
        scorer.refine(refiner, save_path=args.save_refine_path)
        client.clean()
    else:
        with open(args.refine_cache):
            scorer.explanation = json.load(open(args.refine_cache, "r"))
    scorer.load_model()
    scores = scorer()

    if ddp:
        gathered_scores = [None for _ in range(dist.get_world_size())]
        all_rank_scores = dist.all_gather_object(gathered_scores, scores)
        final_scores = []
        for score in gathered_scores:
            final_scores.extend(score)
    else:
        final_scores = scores

    if rank == 0:
        save_dir = "/".join(args.save_score_path.split("/")[:-1])
        os.makedirs(save_dir, exist_ok=True)
        with open(args.save_score_path, "w") as f:
            json.dump(final_scores, f, indent=4)
    if ddp:
        dist.barrier()
