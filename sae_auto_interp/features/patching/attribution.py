import collections
import json
import os
from functools import partial
from typing import List, Union

import torch
import torch.distributed as dist
from PIL import Image
from torchtyping import TensorType
from tqdm import tqdm
from transformers import LlavaNextImageProcessor, PreTrainedModel, PreTrainedTokenizer

from sae_auto_interp.sae import Sae

from .utils import (
    get_logit_diff,
    get_model_backward_cache_with_sae,
    get_model_forward_cache_with_sae,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Attribution:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        sae_path: str,
        data_path: str,
        selected_sae: str = None,
        image_processor: LlavaNextImageProcessor = None,
    ) -> None:
        self.model = model
        self.image_processor = image_processor

        if selected_sae is not None:
            if not os.path.exists(sae_path):
                sae = Sae.load_from_hub(
                    sae_path, hookpoint=selected_sae, device=model.device
                )
            else:
                sae = Sae.load_from_disk(
                    os.path.join(sae_path, selected_sae), device=model.device
                )
            self.sae_dict = {selected_sae: sae}
        else:
            self.sae_dict = Sae.load_many(
                sae_path,
                local=True if os.path.exists(sae_path) else False,
                device=model.device,
            )
        for module_name, sae in self.sae_dict.items():
            sae.eval()

        self.data_path = data_path
        with open(self.data_path, "r") as f:
            self.data = json.load(f)

        # Format your data in list of dict where
        # each dict should in format
        # {
        # "prompt": "xxx",
        # "answer": "xxx",
        # "baseline": "xxx"
        # }
        self.prompt = []
        # this should be a list with shape (batch_size, 2)
        # the first one should be the correct token
        # and the second one is the baseline token
        self.answer = []
        self.image = []
        self.image_sizes = []
        for item in self.data:
            self.prompt.append(item["prompt"])
            self.answer.append([str(item["answer"]), str(item["baseline"])])
            image = Image.open(item["image"])
            self.image.append(image)
            self.image_sizes.append([image.size[0], image.size[1]])

        self.pixel_values = self.image_processor(
            [im for im in self.image],
            do_pad=True,
            return_tensors="pt",
        )["pixel_values"]

        self.prompt_ids = tokenizer(self.prompt, return_tensors="pt")["input_ids"].to(
            model.device
        )
        self.answer_ids = []
        for answer in self.answer:
            self.answer_ids.append(
                [
                    tokenizer.convert_tokens_to_ids(answer[0]),
                    tokenizer.convert_tokens_to_ids(answer[1]),
                ]
            )
        self.answer_ids = torch.tensor(self.answer_ids).to(model.device)

        # If it is pure llama model, then have no language model
        # but model.model instead
        self.name_to_module = {
            name: getattr(model, "language_model", model.model).get_submodule(name)
            for name in self.sae_dict.keys()
        }
        self.module_to_name = {v: k for k, v in self.name_to_module.items()}

        self.metric = partial(get_logit_diff, answer_token_indices=self.answer_ids)

    def get_attribution(self, indices: Union[List[int], TensorType["indices"]] = None):
        local_rank = os.environ.get("LOCAL_RANK")
        ddp = local_rank is not None
        saes = [v for v in self.sae_dict.values()]
        if indices is None:
            num_latents = getattr(saes[0].cfg, "num_latens", None)
            k = (
                num_latents
                if num_latents is not None
                else saes[0].d_in * saes[0].cfg.expansion_factor
            )
            indices = torch.arange(k)
        rank = dist.get_rank()
        pbar = tqdm(
            total=len(indices), desc="Calculating attribution", disable=not rank == 0
        )
        attribution_dict = collections.defaultdict(list)
        for idx in indices:
            # Get the logits
            clean_logits, clean_cache = get_model_forward_cache_with_sae(
                self.model,
                {
                    "input_ids": self.prompt_ids,
                    "pixel_values": self.pixel_values,
                    "image_sizes": self.image_sizes,
                },
                self.sae_dict,
                self.module_to_name,
            )

            corrupted_logits, corrupted_cache = get_model_forward_cache_with_sae(
                self.model,
                {
                    "input_ids": self.prompt_ids,
                    "pixel_values": self.pixel_values,
                    "image_sizes": self.image_sizes,
                },
                self.sae_dict,
                self.module_to_name,
                off_features=idx,
            )

            # Set the retain grad of the corrupted residual stream to True
            # So that we get the gradient on the activations
            # Formula:
            # (clean_act - corrupted_act) * corrupted_act.grad
            # Then sum and reduce according to the dim
            for module_name, tensor in corrupted_cache.items():
                tensor.retain_grad()

            # Get the gradients
            values = get_model_backward_cache_with_sae(
                logits=corrupted_logits,
                metrics=self.metric,
            )

            for module_name in self.sae_dict.keys():
                attribution = (
                    clean_cache[module_name] - corrupted_cache[module_name]
                ) * corrupted_cache[module_name].grad
                attribution = attribution.detach().cpu()
                # Sum at the hidden_dim
                # Result in (batch_size, seq_len)
                attribution = attribution.sum(dim=-1)
                attribution_dict[module_name].append(attribution)
            pbar.update(1)

        pbar.close()
        if ddp:
            dist.barrier()

        return attribution_dict
