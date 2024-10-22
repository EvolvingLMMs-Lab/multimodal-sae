import torch
from datasets import Dataset
from torchtyping import TensorType
from transformers import AutoProcessor

from ..config import FeatureConfig
from .features import FeatureRecord, prepare_examples, prepare_image_examples
from .loader import BufferOutput


def _to_dense(tokens, activations, locations):
    # Reconstruct dense tensor
    batch_len, seq_len = tokens.shape
    sparse_activations = torch.sparse_coo_tensor(
        locations.t(), activations, (batch_len, seq_len)
    )
    dense_activations = sparse_activations.to_dense()

    # Get unique location rows along the tokens tensor
    unique_batch_pos = torch.unique(locations[:, 0])
    token_batches = tokens[unique_batch_pos]
    dense_activations = dense_activations[unique_batch_pos]

    return token_batches, dense_activations


# TODO: We should add an option to change stride size
def _reconstruct_examples(dense_activations, token_batches, ctx_len):
    # Max pool activations
    # (bs, ctx_len) -> (bs, 1)
    avg_pools = torch.nn.functional.max_pool1d(
        dense_activations, kernel_size=ctx_len, stride=ctx_len
    )

    # Unfold tokens and activations to match
    # Kc : I really don't know why this is
    # needed. Tested
    # (activation_windows != dense_activations).sum()
    # and the results is 0
    activation_windows = dense_activations.unfold(1, ctx_len, ctx_len).reshape(
        -1, ctx_len
    )
    # Also confirmed
    # (token_batches != token_windows).sum() is 0
    token_windows = token_batches.unfold(1, ctx_len, ctx_len).reshape(-1, ctx_len)

    return token_windows, activation_windows, avg_pools


def _top_k_pools(dense_activations, token_batches, ctx_len, max_examples):
    token_windows, activation_windows, avg_pools = _reconstruct_examples(
        dense_activations, token_batches, ctx_len
    )

    # Filter out zero pools
    non_zero_mask = avg_pools != 0
    non_zero_pools = avg_pools[non_zero_mask]

    # Get top k activation pools
    k = min(max_examples, len(non_zero_pools))
    top_indices = torch.topk(avg_pools.flatten(), k).indices

    # Get the top indices
    activation_windows = activation_windows[top_indices]
    token_windows = token_windows[top_indices]

    return token_windows, activation_windows


def pool_max_activation_windows(
    record: FeatureRecord,
    buffer_output: BufferOutput,
    tokens: TensorType["batch", "seq"],
    cfg: FeatureConfig,
):
    token_batches, dense_activations = _to_dense(
        tokens, buffer_output.activations, buffer_output.locations
    )

    token_windows, activation_windows = _top_k_pools(
        dense_activations, token_batches, cfg.example_ctx_len, cfg.max_examples
    )

    # Set as examples
    record.examples = prepare_examples(token_windows, activation_windows)


def pool_max_activations_windows_image(
    record: FeatureRecord,
    buffer_output: BufferOutput,
    tokens: Dataset,
    cfg: FeatureConfig,
    processor: AutoProcessor,
):
    activations = buffer_output.activations
    locations = buffer_output.locations

    # Num of Images
    batch_size = len(tokens)
    # Create a fake seq len here,
    # even llava-ov have less than 8000 image tokens so this should be enough for now
    seq_len = 8000
    fake_tokens = torch.zeros(batch_size, seq_len)
    sparse_activations = torch.sparse_coo_tensor(
        locations.t(), activations, (batch_size, seq_len)
    )
    dense_activations = sparse_activations.to_dense()

    avg_pools = torch.nn.functional.avg_pool1d(
        dense_activations, kernel_size=seq_len, stride=seq_len
    )

    # An ugly hardcode here, because there are duplicated images in llava-next data
    # get top k + 50 indices and remove duplicate
    top_indices = torch.topk(
        avg_pools.flatten(), cfg.max_examples + 50
    ).indices.tolist()
    image_ids = tokens.select(indices=top_indices)["id"]
    presence_image_id = set()
    new_top_indices = []
    for idx, image_id in enumerate(image_ids):
        if image_id not in presence_image_id:
            new_top_indices.append(top_indices[idx])
            presence_image_id.add(image_id)
    if len(new_top_indices) < cfg.max_examples:
        new_top_indices.append(
            [new_top_indices[0]] * (len(cfg.max_examples) - len(new_top_indices))
        )
    elif len(new_top_indices) > cfg.max_examples:
        new_top_indices = new_top_indices[: cfg.max_examples]
    top_indices = new_top_indices

    # will construct fake tokens eventually
    top_images = tokens.select(indices=top_indices)["image"]
    # top_images = tokens.select(indices=[0, 1, 2])["image"]

    record.examples = prepare_image_examples(
        fake_tokens[top_indices], dense_activations[top_indices], top_images, processor
    )
    # record.examples = prepare_image_examples(fake_tokens[:3], dense_activations[:3], top_images)


def random_activation_windows(
    record,
    tokens: TensorType["batch", "seq"],
    buffer_output: BufferOutput,
    ctx_len: int,
    n_random: int,
):
    torch.manual_seed(22)
    batch_size = tokens.shape[0]
    unique_batch_pos = buffer_output.locations[:, 0].unique()

    mask = torch.ones(batch_size, dtype=torch.bool)
    mask[unique_batch_pos] = False

    available_indices = mask.nonzero().squeeze()

    selected_indices = available_indices[
        torch.randperm(len(available_indices))[:n_random]
    ]

    toks = tokens[selected_indices, 10 : 10 + ctx_len]

    record.random_examples = prepare_examples(
        toks,
        torch.zeros_like(toks),
    )


def default_constructor(
    record: FeatureRecord,
    tokens: TensorType["batch", "seq"],
    buffer_output: BufferOutput,
    n_random: int,
    ctx_len: int,
    max_examples: int,
):
    pool_max_activation_windows(
        record,
        tokens=tokens,
        buffer_output=buffer_output,
        ctx_len=ctx_len,
        max_examples=max_examples,
    )

    random_activation_windows(
        record,
        tokens=tokens,
        buffer_output=buffer_output,
        n_random=n_random,
        ctx_len=ctx_len,
    )
