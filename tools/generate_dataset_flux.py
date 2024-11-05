import argparse
import json

import torch
from datasets import Dataset, Features, Image, Value
from diffusers import FluxPipeline
from tqdm import tqdm


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refined-explanation", type=str, help="The path of the refined explanation."
    )
    parser.add_argument("--hf-repo-id", type=str, help="The repo id of the hf dataset")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_argument()
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
    )
    pipe = pipe.to("cuda")

    explanations = json.load(open(args.refined_explanation, "r"))

    df = {"feature": [], "image": []}

    feature = Features({"feature": Value("string"), "image": Image()})

    pbar = tqdm(total=len(explanations))
    for feature, explanation in explanations.items():
        if "Unable to produce descriptions" in explanation:
            pbar.update(1)
            continue
        prompt = explanation
        image = pipe(
            prompt,
            height=512,
            width=512,
            guidance_scale=3.5,
            num_inference_steps=30,
            max_sequence_length=512,
            generator=torch.Generator("cpu").manual_seed(0),
        ).images[0]
        df["feature"].append(feature)
        df["image"].append(image)
        pbar.update(1)

    pbar.close()

    dataset = Dataset.from_dict(df)
    dataset.push_to_hub(args.hf_repo_id)
