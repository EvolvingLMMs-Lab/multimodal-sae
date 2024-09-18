import asyncio
import json
import os
from functools import partial
from multiprocessing import cpu_count
from typing import Union

import torch
from datasets import load_dataset
from loguru import logger
from simple_parsing import ArgumentParser
from transformers import AutoProcessor

from sae_auto_interp.config import ExperimentConfig, FeatureConfig
from sae_auto_interp.features import (
    FeatureDataset,
    pool_max_activations_windows_image,
    sample_with_explanation,
)
from sae_auto_interp.features.features import FeatureRecord
from sae_auto_interp.pipeline import Pipeline


async def image_saver(record: FeatureRecord, save_dir: str):
    feature_name = f"{record.feature}"
    module_name = record.feature.module_name.replace(".", "_")
    save_dir = os.path.join(save_dir, module_name, feature_name)
    os.makedirs(save_dir, exist_ok=True)
    for idx, example in enumerate(record.examples):
        example.image.save(os.path.join(save_dir, f"examples_{idx}.jpg"))
        example.activation_image.save(
            os.path.join(save_dir, f"activated_examples_{idx}.jpg")
        )


def main(args: Union[FeatureConfig, ExperimentConfig]):
    ### Load tokens ###
    logger.info("Load dataset")
    tokens = load_dataset(args.experiment.dataset, split=args.experiment.split)
    processor = AutoProcessor.from_pretrained(args.experiment.model)

    modules = os.listdir(args.experiment.save_dir)
    if args.experiment.selected_layers:
        modules = [
            mod
            for idx, mod in enumerate(modules)
            if idx in args.experiment.selected_layers
        ]
    logger.info(f"Module list : {modules}")
    features = {mod: torch.arange(150) for mod in modules}

    dataset = FeatureDataset(
        raw_dir=args.experiment.save_dir,
        cfg=args.feature,
        modules=modules,
        features=features,
    )

    # Put every explanations in to a single dict with
    # key = the module layer + the feature name
    # value = the explanation
    explanations = {}
    explanation_files = os.listdir(args.experiment.explanation_dir)
    explanation_files = [
        e
        for e in explanation_files
        if os.path.isfile(os.path.join(args.experiment.explanation_dir, e))
    ]
    for file in explanation_files:
        with open(os.path.join(args.experiment.explanation_dir, file), "r") as f:
            data = json.load(f)

        for da in data:
            for key_name, content in da.items():
                if key_name != "prompt":
                    explanations[key_name] = content

    loader = partial(
        dataset.load,
        constructor=partial(
            pool_max_activations_windows_image,
            tokens=tokens,
            cfg=args.feature,
            processor=processor,
        ),
        sampler=partial(
            sample_with_explanation, cfg=args.experiment, explanations=explanations
        ),
    )

    save_dir = os.path.join(args.experiment.explanation_dir, "images")
    os.makedirs(save_dir, exist_ok=True)

    saver = partial(
        image_saver,
        save_dir=save_dir,
    )

    pipeline = Pipeline(loader, saver)

    asyncio.run(pipeline.run(max_processes=cpu_count() // 2))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(FeatureConfig, dest="feature")
    parser.add_arguments(ExperimentConfig, dest="experiment")

    args = parser.parse_args()
    main(args)
