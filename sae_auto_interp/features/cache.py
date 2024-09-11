import os
import re
from collections import defaultdict
from typing import Dict, Sequence, Tuple, Union

import torch
import torch.distributed as dist
from datasets import Dataset
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader
from torchtyping import TensorType
from tqdm import tqdm
from transformers import (
    AutoModel,
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
    PreTrainedTokenizer,
)

from ..sae import Sae


class Cache:
    """
    The Buffer class stores feature locations and activations for modules.
    """

    def __init__(
        self,
        shard_size: int,
        filters: Dict[str, TensorType["indices"]] = None,
        batch_size: int = 64,
    ):
        self.feature_locations = defaultdict(list)
        self.feature_activations = defaultdict(list)
        self.filters = filters
        self.batch_size = batch_size
        # The size of the dataset splited onto one shard
        self.shard_size = shard_size
        self.rank = dist.get_rank() if dist.is_initialized() else 0

    def add(
        self,
        latents: TensorType["batch", "sequence", "feature"],
        batch_number: int,
        module_path: str,
    ):
        """
        Add the latents from a module to the buffer
        """
        feature_locations, feature_activations = self.get_nonzeros(latents, module_path)
        feature_locations = feature_locations.cpu()
        feature_activations = feature_activations.cpu()

        feature_locations[:, 0] += batch_number * self.batch_size + self.shard_size
        self.feature_locations[module_path].append(feature_locations)
        self.feature_activations[module_path].append(feature_activations)

    def save(self):
        """
        Concatenate the feature locations and activations
        """

        for module_path in self.feature_locations.keys():
            self.feature_locations[module_path] = torch.cat(
                self.feature_locations[module_path], dim=0
            )

            self.feature_activations[module_path] = torch.cat(
                self.feature_activations[module_path], dim=0
            )

    def get_nonzeros(
        self, latents: TensorType["batch", "seq", "feature"], module_path: str
    ):
        """
        Get the nonzero feature locations and activations
        """

        nonzero_feature_locations = torch.nonzero(latents.abs() > 1e-5)
        nonzero_feature_activations = latents[latents.abs() > 1e-5]

        # Return all nonzero features if no filter is provided
        if self.filters is None:
            return nonzero_feature_locations, nonzero_feature_activations

        # Return only the selected features if a filter is provided
        else:
            selected_features = self.filters[module_path]
            mask = torch.isin(nonzero_feature_locations[:, 2], selected_features)

            return nonzero_feature_locations[mask], nonzero_feature_activations[mask]


class FeatureCache:
    def __init__(
        self,
        model: Union[AutoModel, LlavaNextForConditionalGeneration],
        tokenizer: PreTrainedTokenizer,
        submodule_dict: Dict[str, Sae],
        batch_size: int,
        shard_size: int,
        filters: Dict[str, TensorType["indices"]] = None,
    ):
        if isinstance(model, LlavaNextForConditionalGeneration):
            self.llava_model = model
            self.model = self.llava_model.language_model
        else:
            self.llava_model = None
            self.model = model
        self.tokenizer = tokenizer

        self.name_to_module = {
            name: self.model.get_submodule(name) for name in submodule_dict.keys()
        }
        self.module_to_name = {v: k for k, v in self.name_to_module.items()}

        # Submodule dict is a dictionary of
        # Hook_layer_name : sae
        self.submodule_dict = submodule_dict

        self.batch_size = batch_size
        first_sae = list(submodule_dict.values())[0]
        self.width = (
            first_sae.cfg.num_latents
            if first_sae.cfg.num_latents
            else first_sae.cfg.d_in * first_sae.cfg.expansion_factor
        )

        self.cache = Cache(shard_size, filters, batch_size=batch_size)
        if filters is not None:
            self.filter_submodules(filters)

        print(submodule_dict.keys())

    def load_token_batches(
        self, n_tokens: int, tokens: TensorType["batch", "sequence"]
    ):
        max_batches = n_tokens // tokens.shape[1]
        tokens = tokens[:max_batches]

        n_mini_batches = len(tokens) // self.batch_size

        token_batches = [
            tokens[self.batch_size * i : self.batch_size * (i + 1), :]
            for i in range(n_mini_batches)
        ]

        return token_batches

    def filter_submodules(self, filters: Dict[str, TensorType["indices"]]):
        filtered_submodules = {}
        for module_path in self.submodule_dict.keys():
            if module_path in filters:
                filtered_submodules[module_path] = self.submodule_dict[module_path]
        self.submodule_dict = filtered_submodules

    def run(self, n_tokens: int, tokens: Dataset):
        token_batches = DataLoader(
            tokens, batch_size=self.batch_size, drop_last=True, shuffle=False
        )

        total_tokens = 0
        total_batches = len(token_batches)
        tokens_per_batch = n_tokens

        rank_zero = not dist.is_initialized() or dist.get_rank() == 0

        with tqdm(
            total=total_batches, desc="Caching features", disable=not rank_zero
        ) as pbar:
            for batch_number, batch in enumerate(token_batches):
                total_tokens += tokens_per_batch

                with torch.no_grad():
                    buffer: dict[str, torch.Tensor] = {}

                    def hook(module: torch.nn.Module, _, outputs):
                        # Maybe unpack tuple outputs
                        if isinstance(outputs, tuple):
                            outputs = outputs[0]

                        name = self.module_to_name[module]
                        buffer[name] = outputs

                    # Forward pass on the model to get the next batch of activations
                    handles = [
                        mod.register_forward_hook(hook)
                        for mod in self.name_to_module.values()
                    ]
                    device = self.model.device
                    try:
                        with torch.no_grad():
                            if self.llava_model is not None:
                                # Potential multi-model features
                                # here so I separate
                                self.llava_model(
                                    batch["input_ids"].to(device),
                                )
                            else:
                                self.model(batch["input_ids"].to(device))
                    finally:
                        for handle in handles:
                            handle.remove()

                    # Add the sae latents into the cache
                    # latents should be (bs, seq, num_latents)
                    # Only select the top_k features and masked out the others
                    for module_path, latents in buffer.items():
                        latents = self.submodule_dict[module_path].pre_acts(latents)
                        topk = torch.topk(
                            latents, k=self.submodule_dict[module_path].cfg.k, dim=-1
                        )
                        # make all other values 0
                        result = torch.zeros_like(latents)
                        # results (bs, seq, num_latents)
                        result.scatter_(-1, topk.indices, topk.values)
                        self.cache.add(result, batch_number, module_path)

                    del buffer
                    torch.cuda.empty_cache()

                # Update the progress bar
                pbar.update(1)
                pbar.set_postfix({"Total Tokens": f"{total_tokens:,}"})

        print(f"Total tokens processed: {total_tokens:,}")
        self.cache.save()
        if dist.is_initialized():
            dist.barrier()

    def save(self, save_dir):
        for module_path in self.cache.feature_locations.keys():
            output_file = f"{save_dir}/{module_path}.safetensors"

            data = {
                "locations": self.cache.feature_locations[module_path],
                "activations": self.cache.feature_activations[module_path],
            }

            save_file(data, output_file)

    def _generate_split_indices(self, n_splits):
        boundaries = torch.linspace(0, self.width, steps=n_splits + 1).long()

        # Adjust end by one
        return list(zip(boundaries[:-1], boundaries[1:] - 1))

    def concate_safetensors(self, n_splits: int, save_dir):
        split_indices = self._generate_split_indices(n_splits)

        for module_path in tqdm(
            self.cache.feature_locations.keys(), desc="Concating safetensors"
        ):
            module_dir = f"{save_dir}/{module_path}"
            for start, end in split_indices:
                activations = []
                locations = []
                res = [
                    f
                    for f in os.listdir(f"{module_dir}")
                    if re.search(r"Rank[0-9]_{}_{}\.safetensors".format(start, end), f)
                ]

                for r in res:
                    split_data = load_file(os.path.join(module_dir, r))
                    activations.append(split_data["activations"])
                    locations.append(split_data["locations"])
                    os.remove(os.path.join(module_dir, r))

                activations = torch.cat(activations, dim=0)
                locations = torch.cat(locations, dim=0)

                split_data = {
                    "locations": locations,
                    "activations": activations,
                }
                output_file = f"{module_dir}/{start}_{end}.safetensors"

                save_file(split_data, output_file)

    def save_splits(self, n_splits: int, save_dir, rank: int):
        split_indices = self._generate_split_indices(n_splits)

        for module_path in tqdm(
            self.cache.feature_locations.keys(), desc="Saving safetensors on each rank"
        ):
            feature_locations = self.cache.feature_locations[module_path]
            feature_activations = self.cache.feature_activations[module_path]

            features = feature_locations[:, 2]

            for start, end in split_indices:
                mask = (features >= start) & (features < end)

                masked_locations = feature_locations[mask]
                masked_activations = feature_activations[mask]

                module_dir = f"{save_dir}/{module_path}"
                os.makedirs(module_dir, exist_ok=True)

                output_file = f"{module_dir}/Rank{rank}_{start}_{end}.safetensors"

                split_data = {
                    "locations": masked_locations,
                    "activations": masked_activations,
                }

                save_file(split_data, output_file)


class FeatureImageCache(FeatureCache):
    def __init__(
        self,
        model: LlavaNextForConditionalGeneration,
        tokenizer: PreTrainedTokenizer,
        submodule_dict: Dict[str, Sae],
        batch_size: int,
        shard_size: int,
        filters: Dict[str, TensorType["indices"]] = None,
        processor: LlavaNextProcessor = LlavaNextProcessor.from_pretrained(
            "llava-hf/llama3-llava-next-8b-hf"
        ),
    ):
        super().__init__(
            model, tokenizer, submodule_dict, batch_size, shard_size, filters
        )
        self.processor = processor
        self.prompt = "<image>"

    def run(self, n_tokens: int, tokens: Dataset):
        def collate_fn(instances: Sequence[Tuple]) -> Dict[str, torch.Tensor]:
            images = [instance["image"].convert("RGB") for instance in instances]
            image_sizes = [image.size for image in images]
            image_sizes = torch.tensor(image_sizes).to(torch.long)
            batch = dict(
                images=images,
                image_sizes=image_sizes,
            )

            return batch

        token_batches = DataLoader(
            tokens,
            batch_size=self.batch_size,
            drop_last=True,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )

        total_tokens = 0
        total_batches = len(token_batches)

        rank_zero = not dist.is_initialized() or dist.get_rank() == 0

        with tqdm(
            total=total_batches, desc="Caching features", disable=not rank_zero
        ) as pbar:
            for batch_number, batch in enumerate(token_batches):
                images = batch["images"]
                inputs = self.processor(
                    text=[self.prompt] * self.batch_size,
                    images=images,
                    return_tensors="pt",
                )

                total_tokens += self.batch_size

                with torch.no_grad():
                    buffer: dict[str, torch.Tensor] = {}

                    def hook(module: torch.nn.Module, _, outputs):
                        # Maybe unpack tuple outputs
                        if isinstance(outputs, tuple):
                            outputs = outputs[0]

                        name = self.module_to_name[module]
                        buffer[name] = outputs

                    # Forward pass on the model to get the next batch of activations
                    handles = [
                        mod.register_forward_hook(hook)
                        for mod in self.name_to_module.values()
                    ]
                    device = self.model.device
                    try:
                        with torch.no_grad():
                            outputs = self.llava_model(
                                input_ids=inputs["input_ids"].to(device),
                                pixel_values=inputs["pixel_values"].to(device),
                                image_sizes=inputs["image_sizes"].to(device),
                                attention_mask=inputs["attention_mask"].to(device),
                            )
                    finally:
                        for handle in handles:
                            handle.remove()

                    # Add the sae latents into the cache
                    # latents should be (bs, seq, num_latents)
                    # Only select the top_k features and masked out the others
                    for module_path, latents in buffer.items():
                        # We don't include the text
                        # So the llama tokenizer will add a bos token,
                        # I hardcode to remove it here,
                        # possibly later should some better way to remove this text
                        latents = self.submodule_dict[module_path].pre_acts(
                            latents[:, 1:, :]
                        )
                        topk = torch.topk(
                            latents, k=self.submodule_dict[module_path].cfg.k, dim=-1
                        )
                        # make all other values 0
                        result = torch.zeros_like(latents)
                        # results (bs, seq, num_latents)
                        result.scatter_(-1, topk.indices, topk.values)
                        self.cache.add(result, batch_number, module_path)

                    del buffer
                    torch.cuda.empty_cache()

                # Update the progress bar
                pbar.update(1)
                pbar.set_postfix({"Total Images": f"{total_tokens:,}"})

        print(f"Total Images processed: {total_tokens:,}")
        self.cache.save()
        if dist.is_initialized():
            dist.barrier()
