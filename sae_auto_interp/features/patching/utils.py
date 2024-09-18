from typing import Any, Dict, Tuple, Union

import torch
from torchtyping import TensorType

from sae_auto_interp.sae import Sae


def get_logit_diff(
    logits: TensorType["batch", "seq", "vocab_size"],
    answer_token_indices: TensorType["batch", 2],
):
    if len(logits.shape) == 3:
        # Get final logits only
        logits = logits[:, -1, :]
    correct_logits = logits.gather(1, answer_token_indices[:, 0].unsqueeze(1))
    incorrect_logits = logits.gather(1, answer_token_indices[:, 1].unsqueeze(1))
    return (correct_logits - incorrect_logits).mean()


def get_model_forward_cache_with_sae(
    model: torch.nn.Module,
    inputs: Dict[str, Any],
    sae_dict: Dict[str, Sae],
    module_to_name: Dict[torch.nn.Module, str],
    off_features: int = None,
) -> Tuple[
    TensorType["batch", "seq_len", "dim"],
    Dict[str, TensorType["batch", "seq_len", "dim"]],
]:
    cache = {}

    def forward_cache_hook(module: torch.nn.Module, inputs, outputs):
        # Maybe unpack tuple outputs
        if isinstance(outputs, tuple):
            unpack_outputs = list(outputs)
        else:
            unpack_outputs = list(outputs)
        name = module_to_name[module]
        sae = sae_dict[name]
        bs, seq_len, dim = unpack_outputs[0].shape
        latents = sae.pre_acts(unpack_outputs[0].flatten(0, 1))
        if off_features is not None:
            mask = torch.ones_like(latents)
            mask[:, off_features] = 0
            # Avoid inplace operation of 0 here, use a mask
            latents = latents * mask
        top_acts, top_indices = sae.select_topk(latents)
        sae_out = sae.decode(top_acts, top_indices).to(torch.float16)
        # Since it is reconstruction, all dim should be the same
        sae_out = sae_out.view(bs, seq_len, dim)
        new_outputs = [sae_out] + unpack_outputs[1:]
        cache[name] = sae_out
        if isinstance(outputs, tuple):
            outputs = tuple(new_outputs)
        else:
            outputs = new_outputs[0]
        return outputs

    handles = [
        mod.register_forward_hook(forward_cache_hook) for mod in module_to_name.keys()
    ]
    try:
        outputs = model(**inputs)
        logits = outputs["logits"]
    finally:
        for handle in handles:
            handle.remove()

    return logits, cache


def get_model_backward_cache_with_sae(
    logits: TensorType["batch", "seq_len", "dim"], metrics
):
    values = metrics(logits)
    values.backward()

    return values
