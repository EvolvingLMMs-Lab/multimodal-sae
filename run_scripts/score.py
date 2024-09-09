import asyncio
import json
import os
from functools import partial
from multiprocessing import cpu_count
from typing import Union

from datasets import load_dataset
from loguru import logger
from simple_parsing import ArgumentParser
from transformers import AutoTokenizer

from sae_auto_interp.agents.scorers import SimpleScorer
from sae_auto_interp.clients import SRT
from sae_auto_interp.config import ExperimentConfig, FeatureConfig
from sae_auto_interp.features import (
    FeatureDataset,
    pool_max_activation_windows,
    sample_with_explanation,
)
from sae_auto_interp.pipeline import Pipeline, process_wrapper
from sae_auto_interp.sae.data import chunk_and_tokenize


def main(args: Union[FeatureConfig, ExperimentConfig]):
    ### Load tokens ###
    logger.info("Load tokenizer and dataset")
    tokenizer = AutoTokenizer.from_pretrained(args.experiment.model)
    tokens = load_dataset(args.experiment.dataset, split=args.experiment.split)

    logger.info(
        f"Chunking dataset into {args.feature.example_ctx_len} tokens per sample..."
    )

    tokens = chunk_and_tokenize(
        tokens,
        tokenizer,
        max_seq_len=args.feature.example_ctx_len,
    )
    tokens = tokens["input_ids"]

    modules = os.listdir(args.experiment.save_dir)
    if args.experiment.selected_layers:
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
    )

    # Put every explanations in to a single dict with
    # key = the module layer + the feature name
    # value = the explanation
    explanations = {}
    explanation_files = os.listdir(args.experiment.explanation_dir)
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
            pool_max_activation_windows, tokens=tokens, cfg=args.feature
        ),
        sampler=partial(
            sample_with_explanation, cfg=args.experiment, explanations=explanations
        ),
    )

    ### Load client ###
    logger.info("Setup server")

    client = SRT(model="meta-llama/Meta-Llama-3.1-70B-Instruct", tp=8)
    # client = SRT(model="meta-llama/Meta-Llama-3.1-8B-Instruct", tp=2)

    ### Build Explainer pipe ###

    def scorer_postprocess(result):
        messages_list, resps_list, result = result
        # Each result contains the scores from one feature
        module_name = result.record.feature.module_name.replace(".", "_")
        output_path = (
            f"{args.experiment.scores_dir}/{module_name}/{result.record.feature}.json"
        )
        os.makedirs(
            os.path.expanduser(f"{args.experiment.scores_dir}/{module_name}"),
            exist_ok=True,
        )
        result_data = []
        for idx, messages in enumerate(messages_list):
            result_data.append(
                {
                    "examples": messages,
                    "scores": result.scores[idx],
                    "max_activations": result.max_activations[idx],
                }
            )

        with open(output_path, "w") as f:
            json.dump(result_data, f, indent=4)

        return result

    os.makedirs(os.path.expanduser(args.experiment.scores_dir), exist_ok=True)

    scorer_pipe = process_wrapper(
        SimpleScorer(
            client=client,
            tokenizer=tokenizer,
            verbose=True,
            threshold=0.5,
            activations=True,
        ),
        postprocess=scorer_postprocess,
    )

    ### Build the pipeline ###

    pipeline = Pipeline(
        loader,
        scorer_pipe,
    )

    asyncio.run(pipeline.run(max_processes=cpu_count() // 2))
    client.clean()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(FeatureConfig, dest="feature")
    parser.add_arguments(ExperimentConfig, dest="experiment")

    args = parser.parse_args()
    main(args)
