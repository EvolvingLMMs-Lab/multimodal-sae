import asyncio
import os
import random
from dataclasses import dataclass
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
from loguru import logger
from natsort import natsorted
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline

from sae_auto_interp.utils import load_explanation

from .label_refiner import LabelRefiner
from .utils import DetectionResult, get_boxes, refine_masks


class SegmentScorer:
    def __init__(
        self,
        explanation_dir: str,
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
        self.features = [
            self.features[idx]
            for idx in range(len(self.features))
            if idx in self.feature_idx
        ]
        print(f"Rank {self.rank}, Features : {self.features}, idx : {self.feature_idx}")
        self.filtered_explanation = {
            k: v for k, v in self.explanation.items() if k in self.features
        }

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
        for feature in self.features:
            model_layer = feature.split("_")[0].replace(".", "_")
            mask_image_folder = os.path.join(
                self.explanation_dir, "images", model_layer, feature, "masks"
            )
            image_folder = os.path.join(
                self.explanation_dir, "images", model_layer, feature, "images"
            )
            image_files = glob(os.path.join(image_folder, "*.jpg"))
            # Sort from top 0 to top k
            image_files = natsorted(image_files)
            iou_scores = []
            bad_cases = 0
            for idx, image_file in enumerate(image_files):
                image = Image.open(image_file)
                mask = Image.open(os.path.join(mask_image_folder, f"{idx}_mask.jpg"))
                image = image.resize(mask.size)
                try:
                    image_np, detections = self.grounded_segmentation(
                        image, [self.explanation[feature]]
                    )
                except:
                    logger.info(
                        f"Unable to grounded for feature : {feature} - Top {idx + 1}"
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

                intersection = np.logical_and(target, mask_np)
                union = np.logical_or(target, mask_np)
                iou_score = np.sum(intersection) / (np.sum(union))
                iou_scores.append(iou_score)
                annotated_image = Image.fromarray(self.annotate(image_np, detections))

            self.scores.append(
                {
                    "feature": feature,
                    "iou_scores": iou_scores,
                    "avg_iou": (sum(iou_scores) + bad_cases) / len(iou_scores),
                    "k": len(iou_scores),
                    "activated_pct": mask_np.sum() / (mask.size[0] * mask.size[1]),
                }
            )
            pbar.update(1)
        return self.scores

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
