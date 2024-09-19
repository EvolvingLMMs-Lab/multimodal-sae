import argparse
import json
import os

import torch
from datasets import load_dataset
from IPython.core.display import HTML
from IPython.display import display
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlavaNextForConditionalGeneration,
)

from sae_auto_interp.sae import Sae
from sae_auto_interp.sae.data import chunk_and_tokenize

HTML_START = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Activation Visualization</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            padding: 20px;
            line-height: 1.5;
        }}
        h2 {{
            color: #333;
            margin-bottom: 10px;
        }}
        p {{
            font-size: 16px;
            margin: 10px 0;
        }}
        hr {{
            border: none;
            border-top: 1px solid #ccc;
            margin: 20px 0;
        }}
        span {{
            padding: 3px 5px;
            border-radius: 3px;
        }}
    </style>
</head>
<body>
"""

HTML_END = """
</body>
</html>
"""


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sae-path",
        type=str,
        help="The path to your trained sae. Can be local or on the hub",
    )
    parser.add_argument("--model-path", type=str, help="The path to your hooked model.")
    parser.add_argument("--save-path", type=str, help="The path to save the html file")
    parser.add_argument(
        "--filters-path",
        type=str,
        help="The path to the filters.json file. Recommend to use so that you won't render to much feature",
    )
    parser.add_argument(
        "--ctx-len",
        type=int,
        default=64,
        help="The context length of the dataset. Will chunk each sample with this length",
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Manually pass in the text you want to render",
        default=None,
    )
    parser.add_argument(
        "--explanation-path",
        type=str,
        default=None,
        help="Path to the explanation of the features",
    )

    return parser.parse_args()


# Function to map activation levels to a color intensity
def activation_to_color(activation):
    return f"background-color: rgba(255, 0, 0, {activation});"  # Red background with varying transparency


def tokens_to_html(tokens, activations):
    colors = [activation_to_color(activation) for activation in activations]
    html_tokens = []
    for token, color in zip(tokens, colors):
        html_tokens.append(f'<span style="{color}">{token}</span>')
    return " ".join(html_tokens)


# Function to generate HTML for a single feature
def generate_feature_html(feature_name, tokens, activations, explanation):
    activation_max = max(activations)
    activation_min = min(activations)
    activations = (activations - activation_min) / (activation_max - activation_min)
    html_tokens = [
        f'<span style="{activation_to_color(act)}">{token}</span>'
        for token, act in zip(tokens, activations)
    ]
    styled_text = " ".join(html_tokens)
    feature_html = f"""
    <h2>{feature_name}</h2>
    <p>{explanation}</p>
    <p>{styled_text}</p>
    <hr>
    """
    return feature_html


def save_html(html_content, path):
    with open(path, "w") as file:
        file.write(html_content)


if __name__ == "__main__":
    device = "cuda"
    args = parse_argument()
    assert args.text is not None, "Please pass in a sentence by using --text"

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if "llava" in args.model_path:
        model = LlavaNextForConditionalGeneration.from_pretrained(
            args.model_path,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    if args.text is not None:
        text = args.text
        dataset = [text]
        dataset = tokenizer(text, return_tensors="pt")["input_ids"]

    if args.filters_path is not None:
        with open(args.filters_path, "r") as f:
            filters = json.load(f)
        filters = {k: torch.tensor(v) for k, v in filters.items()}
    else:
        filters = None

    with open(args.explanation_path, "r") as f:
        explanations = json.load(f)

    new_explanations = {}
    for feat in explanations:
        for k, v in feat.items():
            if "layers" in k:
                new_explanations[k] = v
    explanations = new_explanations
    del new_explanations

    submodule_dict = {}
    if os.path.exists(args.sae_path):
        if filters is not None:
            for module_name, indices in filters.items():
                logger.info(f"Load sae : {module_name}")
                sae = Sae.load_from_disk(
                    os.path.join(args.sae_path, module_name), device=device
                )
                submodule_dict[module_name] = sae
        else:
            submodule_dict = Sae.load_many(args.sae_path, local=True, device=device)
    else:
        if filters is not None:
            for module_name, indices in filters.items():
                logger.info(f"Load sae : {module_name}")
                sae = Sae.load_from_hub(args.sae_path, module_name, device=device)
                submodule_dict[module_name] = sae
        else:
            submodule_dict = Sae.load_many(args.sae_path, local=False, device=device)

    name_to_module = {
        name: getattr(model, "language_model", model.model).get_submodule(name)
        for name in submodule_dict.keys()
    }
    module_to_name = {v: k for k, v in name_to_module.items()}

    token_batches = DataLoader(dataset, batch_size=1, drop_last=False, shuffle=False)

    pbar = tqdm(total=len(token_batches), desc="Get Activations")

    full_html_content = HTML_START

    for token_batch in token_batches:
        tokens = tokenizer.convert_ids_to_tokens(token_batch[0])
        tokens = [
            tokenizer.convert_tokens_to_string(tokens[i : i + 1])
            for i in range(len(tokens))
        ]
        with torch.no_grad():
            buffer: dict[str, torch.Tensor] = {}

            def hook(module: torch.nn.Module, _, outputs):
                # Maybe unpack tuple outputs
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                name = module_to_name[module]
                buffer[name] = outputs

            # Forward pass on the model to get the next batch of activations
            handles = [
                mod.register_forward_hook(hook) for mod in name_to_module.values()
            ]
            device = model.device
            try:
                with torch.no_grad():
                    model(
                        token_batch.to(device),
                    )
            finally:
                for handle in handles:
                    handle.remove()

            for module_path, latents in buffer.items():
                latents = submodule_dict[module_path].pre_acts(latents)
                topk = torch.topk(latents, k=submodule_dict[module_path].cfg.k, dim=-1)
                # make all other values 0
                result = torch.zeros_like(latents)
                # results (bs, seq, num_latents)
                result.scatter_(-1, topk.indices, topk.values)
                latents = result.cpu()
                if filters is not None:
                    for indice in filters[module_path]:
                        activations = latents[0, :, indice]
                        full_html_content += generate_feature_html(
                            f"{module_path}_feature{indice}",
                            tokens,
                            activations,
                            explanations[f"{module_path}_feature{indice}"],
                        )
                else:
                    for indice in range(latents.shape[2]):
                        activations = latents[0, :, indice]
                        full_html_content += generate_feature_html(
                            f"{module_path}_feature{indice}",
                            tokens,
                            activations,
                            explanations[f"{module_path}_feature{indice}"],
                        )

        pbar.update(1)

    full_html_content += HTML_END
    save_html(full_html_content, args.save_path)
