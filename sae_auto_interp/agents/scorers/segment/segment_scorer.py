import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import torch
from PIL import Image
from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline

from sae_auto_interp.utils import load_explanation

from .utils import DetectionResult, get_boxes, refine_masks


class SegmentScorer:
    def __init__(
        self,
        explanation_dir: str,
        detector: str = "IDEA-Research/grounding-dino-base",
        segmentor: str = "facebook/sam-vit-huge",
        device: str = "cuda",
        threshold: float = 0.3,
    ) -> None:
        self.detector = detector
        self.segmentor = segmentor
        self.device = device
        self.threshold = threshold
        self.object_detector = pipeline(
            model=detector, task="zero-shot-object-detection", device=device
        )
        self.segmentator = AutoModelForMaskGeneration.from_pretrained(segmentor).to(
            device
        )
        self.processor = AutoProcessor.from_pretrained(segmentor)
        self.explanation_dir = explanation_dir
        self.explanation = load_explanation(explanation_dir)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
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
