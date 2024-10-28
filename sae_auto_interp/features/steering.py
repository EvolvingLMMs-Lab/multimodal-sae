import os
from typing import List

import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

from sae_auto_interp.sae import Sae


class SteeringController:
    def __init__(
        self,
        sae: Sae,
        module_name: str,
        feature_idx: List[int],
        model: nn.Module,
        processor: AutoProcessor,
        prompt: str,
        image_path: str = None,
        k: float = 50,
    ):
        """
        Args:
            sae (Sae): The Sae module
            module_name (str): The module layer name, say model.layers.24
            feature_idx (List[int]): A list of feature idx
            model (nn.Module): The model to be hooked on
            processor (AutoProcessor): The processor for the model
            prompt (str): A text sentence
            image_path (str, optional): A path to your input image. Defaults to None.
            k (float, optional): The clamping value. Defaults to 50.
        """
        self.sae = sae
        self.feature_idx = feature_idx
        self.model = model
        self.prompt = prompt
        self.module_name = module_name
        self.hooked_module = model.language_model.get_submodule(module_name)
        self.processor = processor
        self.k = k
        local_rank = os.environ.get("LOCAL_RANK")
        self.ddp = local_rank is not None
        self.rank = int(local_rank) if local_rank is not None else 0
        self.conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        self.image = Image.open(image_path) if image_path is not None else None
        if self.image is not None:
            self.conversation[0]["content"].append(
                {"type": "image"},
            )

        self.prompt = processor.apply_chat_template(
            self.conversation, add_generation_prompt=True
        )

        self.inputs = processor(
            images=self.image, text=self.prompt, return_tensors="pt"
        ).to(model.device)

    def run(self):
        result_dict = {}
        with torch.no_grad():
            output = self.model.generate(**self.inputs, max_new_tokens=512)
        cont = output[:, self.inputs["input_ids"].shape[-1] :]
        original_resps = self.processor.batch_decode(cont, skip_special_tokens=True)[0]

        pbar = tqdm(total=len(self.feature_idx), desc="Clamping...", disable=self.rank)
        for idx in self.feature_idx:
            feature_name = f"{self.module_name}_feature{idx}"

            handles = self.clamp_features_max(
                self.sae, idx, self.hooked_module, k=self.k
            )
            try:
                with torch.no_grad():
                    output = self.model.generate(**self.inputs, max_new_tokens=512)
            finally:
                for handle in handles:
                    handle.remove()
            cont = output[:, self.inputs["input_ids"].shape[-1] :]
            clamped_resps = self.processor.batch_decode(cont, skip_special_tokens=True)[
                0
            ]
            result_dict[feature_name] = {
                "original_resps": original_resps,
                "clamped_resps": clamped_resps,
                "idx": idx,
            }
            pbar.update(1)
        return result_dict

    def clamp_features_max(
        self, sae: Sae, feature: int, hooked_module: torch.nn.Module, k: float = 10
    ):
        def hook(module: torch.nn.Module, _, outputs):
            # Maybe unpack tuple outputs
            if isinstance(outputs, tuple):
                unpack_outputs = list(outputs)
            else:
                unpack_outputs = list(outputs)
            latents = sae.pre_acts(unpack_outputs[0])
            # Only clamp the feature for the first forward
            if latents.shape[1] != 1:
                latents[:, :, feature] = k
            top_acts, top_indices = sae.select_topk(latents)
            sae_out = (
                sae.decode(top_acts[0], top_indices[0]).unsqueeze(0).to(torch.float16)
            )
            unpack_outputs[0] = sae_out
            if isinstance(outputs, tuple):
                outputs = tuple(unpack_outputs)
            else:
                outputs = unpack_outputs[0]
            return outputs

        handles = [hooked_module.register_forward_hook(hook)]

        return handles
