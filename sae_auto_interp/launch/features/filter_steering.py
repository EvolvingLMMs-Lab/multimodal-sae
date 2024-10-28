import argparse
import asyncio
import json

from tqdm import tqdm

from sae_auto_interp.clients import SRT
from sae_auto_interp.prompt import STEERING_FILTER_PROMPT
from sae_auto_interp.utils import load_explanation


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--explanation-dir",
        "-e",
        type=str,
        help="The path to your explanation for features",
    )
    parser.add_argument(
        "--steering-path",
        "-s",
        type=str,
        help="The path where you stored your steering results",
    )
    return parser.parse_args()


def main():
    args = parse_argument()

    explanations = load_explanation(args.explanation_dir)
    steering_result = json.load(open(args.steering_path, "r"))
    explanations = {k: v for k, v in explanations.items() if k in steering_result}
    for k, v in explanations.items():
        steering_result[k]["explanation"] = v
    client = SRT(model="meta-llama/Llama-3.1-70B-Instruct", tp=8)

    async def run():
        sem = asyncio.Semaphore()

        async def _process(prompt, key):
            async with sem:
                return key, await client.generate(prompt)

        tasks = []
        for k in steering_result.keys():
            clamped_resps = steering_result[k]["clamped_resps"]
            origin_resps = steering_result[k]["original_resps"]
            prompt = STEERING_FILTER_PROMPT.format(
                clamped_resps=clamped_resps, origin_resps=origin_resps
            )
            tasks.append(asyncio.create_task(_process(prompt, k)))

        pbar = tqdm(total=len(tasks), desc="Collected")
        for completed_task in asyncio.as_completed(tasks):
            feature_name, result = await completed_task
            steering_result[feature_name]["category"] = result
            pbar.update(1)

    asyncio.run(run())
    with open(args.steering_path, "w", encoding="utf-8") as f:
        json.dump(steering_result, f, indent=4, ensure_ascii=False)
    client.clean()


if __name__ == "__main__":
    main()
