import asyncio
import os
from glob import glob
from typing import Literal, Union

import torch
from datasets import load_dataset
from natsort import natsorted
from PIL import Image
from torchmetrics.multimodal.clip_score import CLIPScore
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm

from sae_auto_interp.agents.scorers.segment.label_refiner import LabelRefiner
from sae_auto_interp.utils import load_explanation


class ClipScorer:
    def __init__(
        self,
        explanation_dir: str,
        dataset_path: str,
        dataset_split: str = "train",
        k: int = 5,
        evaluation_type: Literal["random", "default"] = "default",
        clip_model_name_or_path: str = "openai/clip-vit-base-patch16",
        device: Union[str, torch.device] = "cuda",
        random_runs: int = 30,
    ) -> None:
        self.clip_model_name_or_path = clip_model_name_or_path
        self.device = device
        self.metric = CLIPScore(clip_model_name_or_path).to(device)
        self.explanation_dir = explanation_dir
        self.explanations = load_explanation(explanation_dir)
        self.dataset_path = dataset_path
        self.dataset = load_dataset(dataset_path, split=dataset_split)
        self.features = [k for k in self.explanations.keys()]
        # Make sure they ordered layers by layers
        self.features = natsorted(self.features)
        self.eval_type = evaluation_type
        self.k = k
        self.random_runs = random_runs

    def refine(self, refiner: LabelRefiner, save_path):
        asyncio.run(refiner.refine())
        self.explanation = refiner.refine_features
        refiner.save_result(save_path)

    def run(
        self,
    ):
        self.scores = []
        pbar = tqdm(total=len(self.features), desc="Perform scoring")
        for feature in self.features:
            if "Unable to produce descriptions" in self.explanations[feature]:
                self.scores.append(
                    {
                        "feature": feature,
                        "clip_scores": [],
                        "avg_score": -1,
                        "k": -1,
                        "label": self.explanations[feature],
                    }
                )
                pbar.update(1)
                continue
            if self.eval_type == "default":
                model_layer = feature.split("_")[0].replace(".", "_")
                image_folder = os.path.join(
                    self.explanation_dir, "images", model_layer, feature, "images"
                )
                image_files = glob(os.path.join(image_folder, "*.*"))
                # Sort from top 0 to top k
                image_files = natsorted(image_files)
                images = [Image.open(im).convert("RGB") for im in image_files]
            elif self.eval_type == "random":
                images = []
                final_idx = []
                for _ in range(self.random_runs):
                    select_range = len(self.dataset)
                    select_idx = torch.arange(select_range)
                    select_idx = torch.randperm(select_range)[: self.k].tolist()
                    final_idx.extend(select_idx)
                images += [
                    im.convert("RGB") for im in self.dataset.select(final_idx)["image"]
                ]

            scores = []
            for idx, image in enumerate(images):
                image_tensor = pil_to_tensor(image)
                clip_score = (
                    self.metric(
                        image_tensor.to(self.device), self.explanations[feature]
                    )
                    .detach()
                    .cpu()
                    .item()
                )
                scores.append(clip_score)
            pbar.update(1)
            self.scores.append(
                {
                    "feature": feature,
                    "clip_scores": scores,
                    "avg_score": (sum(scores)) / len(scores),
                    "k": len(scores),
                    "label": self.explanations[feature],
                }
            )
        pbar.close()
        return self.scores
