import asyncio
import os
import random
from dataclasses import dataclass
from functools import partial
from glob import glob
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import torch
import torch.distributed as dist
from datasets import Dataset
from loguru import logger
from natsort import natsorted
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline

from sae_auto_interp.config import FeatureConfig
from sae_auto_interp.features import (
    FeatureDataset,
    pool_max_activations_windows_image,
    random_activations_image,
)
from sae_auto_interp.utils import load_explanation

from .label_refiner import LabelRefiner
from .utils import DetectionResult, get_boxes, refine_masks


class SegmentScorer:
    def __init__(
        self,
        explanation_dir: str,
        activation_dir: str,
        tokens: Dataset,
        processor: AutoProcessor,
        selected_layer: str = "model.layers.24",
        width: int = 131072,
        n_splits: int = 1024,
        detector: str = "IDEA-Research/grounding-dino-base",
        segmentor: str = "facebook/sam-vit-huge",
        device: str = "cuda",
        threshold: float = 0.3,
        filters: torch.Tensor = None,
    ) -> None:
        self.detector_id = detector
        self.segmentor_id = segmentor
        self.device = device
        self.threshold = threshold
        self.processor = AutoProcessor.from_pretrained(segmentor)
        self.explanation_dir = explanation_dir
        self.explanation = load_explanation(explanation_dir)
        self._maybe_init_ddp(filters=filters)
        self._build_dataset(activation_dir, width, n_splits, selected_layer)
        self._init_loader(tokens, processor)

    def _build_dataset(
        self,
        activation_dir: str,
        width: int,
        n_splits: int,
        selected_layer: str = "model.layers.24",
    ):
        self.modules = os.listdir(activation_dir)
        self.width = width
        self.n_splits = n_splits
        self.activation_dir = activation_dir
        self.filters = {selected_layer: self.filters}
        self.feature_cfg = FeatureConfig(
            width=self.width, max_examples=5, n_splits=n_splits
        )
        self.dataset = FeatureDataset(
            activation_dir,
            cfg=self.feature_cfg,
            modules=self.modules,
            features=self.filters,
        )

    def _maybe_init_ddp(self, filters: torch.Tensor = None):
        local_rank = os.environ.get("LOCAL_RANK")
        self.ddp = local_rank is not None
        self.rank = int(local_rank) if local_rank is not None else 0
        self.features = [k for k in self.explanation.keys()]
        # Make sure they ordered layers by layers
        self.features = natsorted(self.features)
        chunk_size = len(self.features) if filters is None else len(filters)
        if self.ddp:
            self.feature_idx = torch.arange(chunk_size).tensor_split(
                dist.get_world_size()
            )[self.rank]
        else:
            self.feature_idx = torch.arange(chunk_size)
        self.filters = self.feature_idx
        self.features = [
            self.features[idx]
            for idx in range(len(self.features))
            if idx in self.feature_idx
        ]
        # print(f"Rank {self.rank}, Features : {self.features}, idx : {self.feature_idx}")
        self.filtered_explanation = {
            k: v for k, v in self.explanation.items() if k in self.features
        }

    def _init_loader(self, tokens: Dataset, processor: AutoProcessor):
        self.loader = partial(
            self.dataset.load,
            constructor=partial(
                pool_max_activations_windows_image,
                tokens=tokens,
                cfg=self.feature_cfg,
                processor=processor,
            ),
        )

    def refine(self, refiner: LabelRefiner, save_path):
        asyncio.run(refiner.refine())
        self.explanation = refiner.refine_features
        refiner.save_result(save_path)

    def load_model(self):
        logger.info(f"Loading object detector : {self.detector_id}")
        self.object_detector = pipeline(
            model=self.detector_id,
            task="zero-shot-object-detection",
            device=self.device,
        )
        logger.info(f"Loading object detector : {self.segmentor_id}")
        self.segmentator = AutoModelForMaskGeneration.from_pretrained(
            self.segmentor_id
        ).to(self.device)

    def __call__(self) -> Any:
        self.scores = []
        pbar = tqdm(total=len(self.features), desc="Perform scoring", disable=self.rank)
        for records in self.loader():
            for record in records:
                explanation = self.explanation[f"{record.feature}"]
                if "Unable to produce descriptions" in explanation:
                    self.scores.append(
                        {
                            "feature": f"{record.feature}",
                            "iou_scores": [],
                            "avg_iou": -1,
                            "k": -1,
                            "activated_pct": -1,
                            "label": explanation,
                        }
                    )
                    pbar.update(1)
                    continue
                iou_scores = []
                activated_pct = []
                bad_cases = 0
                for idx, example in enumerate(record.examples):
                    image: Image.Image = example.image
                    mask: Image.Image = example.mask
                    image = image.resize(mask.size).convert("RGB")
                    try:
                        image_np, detections = self.grounded_segmentation(
                            image, [explanation]
                        )
                    except:
                        logger.info(
                            f"Unable to grounded for feature : {record.feature} - Top {idx + 1}"
                        )
                        iou_scores.append(-1)
                        bad_cases += 1
                        continue
                    # When I create mask, I set the activated region as 0 and inactivated region as
                    # sth larger than 0. So shift here
                    mask_np = np.array(mask)
                    zero_area = mask_np >= 224
                    nonzero_area = mask_np < 224
                    mask_np[zero_area] = 0
                    mask_np[nonzero_area] = 1
                    target = np.zeros_like(mask_np)
                    # Include every detection areas
                    for detection in detections:
                        target = np.logical_or(detection.mask, target)

                    iou_score = self._calculate_iou(mask_np, target)
                    iou_scores.append(iou_score)
                    activated_pct.append(mask_np.sum() / (mask.size[0] * mask.size[1]))
                    # annotated_image = Image.fromarray(self.annotate(image_np, detections))

                self.scores.append(
                    {
                        "feature": f"{record.feature}",
                        "iou_scores": iou_scores,
                        "avg_iou": (sum(iou_scores) + bad_cases) / len(iou_scores),
                        "k": len(iou_scores),
                        "activated_pct": sum(activated_pct) / len(activated_pct)
                        if len(activated_pct) != 0
                        else 0,
                        "label": explanation,
                    }
                )
                pbar.update(1)
        return self.scores

    def _calculate_iou(self, mask: np.array, target: np.array):
        intersection = np.logical_and(target, mask)
        union = np.logical_or(target, mask)
        iou_score = np.sum(intersection) / (np.sum(union))
        return iou_score

    def grounded_segmentation(
        self,
        image: Image.Image,
        labels: List[str],
        polygon_refinement: bool = False,
    ) -> Tuple[np.ndarray, List[DetectionResult]]:
        detections = self.detect(image, labels)
        detections = self.segment(image, detections, polygon_refinement)

        return np.array(image), detections

    def detect(
        self,
        image: Image.Image,
        labels: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Use Grounding DINO to detect a set of labels in an image in a zero-shot fashion.
        """
        labels = [label if label.endswith(".") else label + "." for label in labels]

        results = self.object_detector(
            image, candidate_labels=labels, threshold=self.threshold
        )
        results = [DetectionResult.from_dict(result) for result in results]

        return results

    def segment(
        self,
        image: Image.Image,
        detection_results: List[Dict[str, Any]],
        polygon_refinement: bool = False,
    ) -> List[DetectionResult]:
        """
        Use Segment Anything (SAM) to generate masks given an image + a set of bounding boxes.
        """

        boxes = get_boxes(detection_results)
        inputs = self.processor(
            images=image, input_boxes=boxes, return_tensors="pt"
        ).to(self.device)

        outputs = self.segmentator(**inputs)
        masks = self.processor.post_process_masks(
            masks=outputs.pred_masks,
            original_sizes=inputs.original_sizes,
            reshaped_input_sizes=inputs.reshaped_input_sizes,
        )[0]

        masks = refine_masks(masks, polygon_refinement)

        for detection_result, mask in zip(detection_results, masks):
            detection_result.mask = mask

        return detection_results

    def annotate(
        self,
        image: Union[Image.Image, np.ndarray],
        detection_results: List[DetectionResult],
    ) -> np.ndarray:
        # Convert PIL Image to OpenCV format
        image_cv2 = np.array(image) if isinstance(image, Image.Image) else image
        image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)

        # Iterate over detections and add bounding boxes and masks
        for detection in detection_results:
            label = detection.label
            score = detection.score
            box = detection.box
            mask = detection.mask

            # Sample a random color for each detection
            color = np.random.randint(0, 256, size=3)

            # Draw bounding box
            cv2.rectangle(
                image_cv2, (box.xmin, box.ymin), (box.xmax, box.ymax), color.tolist(), 2
            )
            cv2.putText(
                image_cv2,
                f"{label}: {score:.2f}",
                (box.xmin, box.ymin - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color.tolist(),
                2,
            )

            # If mask is available, apply it
            if mask is not None:
                # Convert mask to uint8
                mask_uint8 = (mask * 255).astype(np.uint8)
                contours, _ = cv2.findContours(
                    mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(image_cv2, contours, -1, color.tolist(), 2)

        return cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)


class RandomSegmentScorer(SegmentScorer):
    """
    A class that sample random image and do the scoring for baseline
    """

    def __init__(
        self,
        explanation_dir: str,
        activation_dir: str,
        tokens: Dataset,
        processor: AutoProcessor,
        selected_layer: str = "model.layers.24",
        width: int = 131072,
        n_splits: int = 1024,
        detector: str = "IDEA-Research/grounding-dino-base",
        segmentor: str = "facebook/sam-vit-huge",
        device: str = "cuda",
        threshold: float = 0.3,
        filters: torch.Tensor = None,
    ) -> None:
        super().__init__(
            explanation_dir,
            activation_dir,
            tokens,
            processor,
            selected_layer,
            width,
            n_splits,
            detector,
            segmentor,
            device,
            threshold,
            filters,
        )

    def _init_loader(self, tokens: Dataset, processor: AutoProcessor):
        self.loader = partial(
            self.dataset.load,
            constructor=partial(
                random_activations_image,
                tokens=tokens,
                cfg=self.feature_cfg,
                processor=processor,
            ),
        )
