from typing import List

import torch
from torchtyping import TensorType
from umap import UMAP

from sae_auto_interp.sae import Sae

from .dim_reducer import DimReducer


class UmapReducer(DimReducer):
    def __init__(self, name: str, n_components: int, **kwargs) -> None:
        super().__init__(name, n_components, **kwargs)
        self.umap = UMAP(n_components=n_components, **kwargs)

    def fit(self, X: TensorType["n_samples", "n_features"], **kwargs):
        return self.umap.fit(X, **kwargs)

    def transform(
        self, X: TensorType["n_samples", "n_features"], **kwargs
    ) -> TensorType["n_samples", "n_components"]:
        return self.umap.transform(X, **kwargs)

    def fit_sae_list(
        self,
        sae_list: List[Sae],
    ):
        weights = []
        for sae in sae_list:
            weights.append(sae.W_dec.detach())
        weights = torch.concat(weights, dim=0)
        return self.fit(weights)
