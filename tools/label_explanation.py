import argparse
import asyncio
import json
from multiprocessing import cpu_count

from tqdm import tqdm

from sae_auto_interp.clients import SRT
from sae_auto_interp.prompt import CONCEPT_LABEL_PROMPT


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refined-explanation", type=str, help="The path of the refined explanation."
    )
    parser.add_argument("--save-path", type=str, help="The path to save your labels")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_argument()

    client = SRT(
        "meta-llama/Llama-3.1-70B-Instruct",
        tp=8,
    )
    explanations = json.load(open(args.refined_explanation, "r"))
    labels = {}
    kwargs = {"max_tokens": 16}

    async def _process():
        sem = asyncio.Semaphore(cpu_count() // 2)

        async def _worker(feature_name, prompt):
            async with sem:
                return feature_name, await client.generate(
                    CONCEPT_LABEL_PROMPT.format(description=prompt), **kwargs
                )

        tasks = [
            asyncio.create_task(_worker(feature_name, prompt))
            for feature_name, prompt in explanations.items()
            if "Unable to produce descriptions" not in prompt
        ]

        pbar = tqdm(total=len(tasks), desc="Collected")
        for completed_task in asyncio.as_completed(tasks):
            feature_name, result = await completed_task
            labels[feature_name] = result
            pbar.update(1)

    asyncio.run(_process())

    client.clean()

    with open(args.save_path, "w") as f:
        json.dump(labels, f, indent=4)
