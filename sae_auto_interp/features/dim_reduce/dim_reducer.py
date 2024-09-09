from abc import ABC, abstractmethod

from torchtyping import TensorType


class DimReducer(ABC):
    def __init__(
        self,
        name: str,  # Name of the reducer
        n_components: int,  # The dim you want to reduce to
        **kwargs,
    ) -> None:
        super().__init__()
        self.name = name
        self.n_components = n_components

    @abstractmethod
    def fit(self, X: TensorType["n_samples", "n_features"], **kwargs):
        raise NotImplementedError

    @abstractmethod
    def transform(
        self, X: TensorType["n_samples", "n_features"], **kwargs
    ) -> TensorType["n_samples", "n_components"]:
        raise NotImplementedError

    def fit_transform(
        self,
        X: TensorType["n_samples", "n_features"],
        **kwargs,
    ) -> TensorType["n_samples", "n_components"]:
        self.fit(X, **kwargs)
        return self.transform(X, **kwargs)
