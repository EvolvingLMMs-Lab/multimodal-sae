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

from sae_auto_interp.clients import SRT
from sae_auto_interp.config import ExperimentConfig, FeatureConfig
from sae_auto_interp.explainers import SimpleExplainer
from sae_auto_interp.features import FeatureDataset, pool_max_activation_windows, sample
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

    loader = partial(
        dataset.load,
        constructor=partial(
            pool_max_activation_windows, tokens=tokens, cfg=args.feature
        ),
        sampler=partial(sample, cfg=args.experiment),
    )

    ### Load client ###
    logger.info("Setup server")

    client = SRT(model="meta-llama/Meta-Llama-3.1-405B-Instruct-FP8", tp=8)

    ### Build Explainer pipe ###

    def explainer_postprocess(result):
        content, reps, result = result
        module_name = result.record.feature.module_name.replace(".", "_")
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

        return result

    os.makedirs(os.path.expanduser(args.experiment.explanation_dir), exist_ok=True)

    explainer_pipe = process_wrapper(
        SimpleExplainer(
            client,
            tokenizer=tokenizer,
            activations=True,
            max_tokens=500,
            temperature=0.0,
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
