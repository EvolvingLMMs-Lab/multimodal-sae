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
from PIL import Image
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
        filters: Dict[str, torch.Tensor] = None,
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
        if self.ddp:
            self.feature_idx = torch.arange(len(self.features)).chunk(
                dist.get_world_size()
            )
        else:
            self.feature_idx = torch.arange(len(self.features))
        self.filters = self.feature_idx if filters is None else filters
        self.features = [
            self.features[idx]
            for idx in range(len(self.features))
            if idx in self.feature_idx and idx in self.filters
        ]
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
        for feature in self.features:
            model_layer = feature.split("_")[0].replace(".", "_")
            mask_image_folder = os.path.join(
                self.explanation_dir, "images", model_layer, feature, "masks"
            )
            image_folder = os.path.join(
                self.explanation_dir, "images", model_layer, feature, "images"
            )
            image_files = glob(os.path.join(image_folder, "*.jpg"))
            for idx, image_file in enumerate(image_files):
                image = Image.open(image_file)
                mask = Image.open(os.path.join(mask_image_folder, f"{idx}_mask.jpg"))
                image_np, detections = self.grounded_segmentation(
                    image, [self.explanation[feature]]
                )
                mask_np = np.array(mask)
        pass

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
