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

from sae_auto_interp.agents.explainers import ExplainerResult, ImageExplainer
from sae_auto_interp.clients import SRT
from sae_auto_interp.config import ExperimentConfig, FeatureConfig
from sae_auto_interp.features import (
    FeatureDataset,
    pool_max_activations_windows_image,
    sample_with_explanation,
)
from sae_auto_interp.features.features import FeatureRecord
from sae_auto_interp.pipeline import Pipeline, process_wrapper
from sae_auto_interp.utils import load_explanation, load_filter


def main(args: Union[FeatureConfig, ExperimentConfig]):
    ### Load tokens ###
    logger.info("Load dataset")
    tokens = load_dataset(args.experiment.dataset, split=args.experiment.split)
    processor = AutoProcessor.from_pretrained(args.experiment.model)

    modules = os.listdir(args.experiment.save_dir)
    if args.experiment.filters_path is not None:
        filters = load_filter(args.experiment.filters_path, device="cpu")
    else:
        filters = None

    if filters is not None:
        modules = [mod for mod in modules if mod in filters]
    elif args.experiment.selected_layers:
        modules = [
            mod
            for idx, mod in enumerate(modules)
            if idx in args.experiment.selected_layers
        ]
    logger.info(f"Module list : {modules}")

    dataset = FeatureDataset(
        raw_dir=args.experiment.save_dir,
        cfg=args.feature,
        modules=modules,
        features=filters,
    )

    # Put every explanations in to a single dict with
    # key = the module layer + the feature name
    # value = the explanation
    explanations = load_explanation(args.experiment.explanation_dir)

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

    ### Load client ###
    logger.info("Setup server")

    client = SRT(model="lmms-lab/llava-onevision-qwen2-7b-ov", tp=2)

    ### Build Explainer pipe ###

    def explainer_postprocess(result: ExplainerResult):
        content, reps, result = result
        record = result.record
        images = [train.image for train in record.train]
        activated_images = [train.activation_image for train in record.train]
        module_name = result.record.feature.module_name.replace(".", "_")
        image_output_dir = f"{args.experiment.explanation_dir}/images/{module_name}"
        os.makedirs(image_output_dir, exist_ok=True)
        output_path = f"{args.experiment.explanation_dir}/{module_name}.json"
        if os.path.exists(output_path):
            output_file = json.load(open(output_path, "r"))
        else:
            output_file = []

        output_file.append(
            {f"{result.record.feature}": f"{result.explanation}", "prompt": content}
        )

        with open(output_path, "w") as f:
            json.dump(output_file, f, indent=4, ensure_ascii=False)

        idx = 0
        for image, activated_image in zip(images, activated_images):
            image.save(f"{image_output_dir}/top_{idx}.jpg")
            activated_image.save(f"{image_output_dir}/top{idx}_activated.jpg")

        return result

    os.makedirs(os.path.expanduser(args.experiment.explanation_dir), exist_ok=True)

    explainer_pipe = process_wrapper(
        ImageExplainer(
            client=client,
            verbose=True,
        ),
        postprocess=explainer_postprocess,
    )

    ### Build the pipeline ###

    pipeline = Pipeline(
        loader,
        explainer_pipe,
    )

    asyncio.run(pipeline.run(max_processes=cpu_count() // 2))
    client.clean()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(FeatureConfig, dest="feature")
    parser.add_arguments(ExperimentConfig, dest="experiment")

    args = parser.parse_args()
    main(args)
