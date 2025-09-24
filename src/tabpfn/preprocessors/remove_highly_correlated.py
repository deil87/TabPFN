"""Remove Constant Features Step."""

from __future__ import annotations

from typing_extensions import override

import numpy as np
import torch
import faiss
import torch.nn.functional as F

from tabpfn.preprocessors.preprocessing_helpers import (
    FeaturePreprocessingTransformerStep,
)


class RemoveHighlyCorrelatedFeaturesStep(FeaturePreprocessingTransformerStep):
    """Remove features that are highly correlated in the training data.
    Further improvements:
    - faiss-gpu can be used if installed dynamically based on environment
    - use approximate nearest neighbor search instead of exact ( see https://github.com/facebookresearch/faiss/wiki/Faiss-indexes )
    - introduce support for categorical features ( eg. Chi-square test based removal )
    - add parameter to choose which feature to keep among correlated ones (currently first one is kept) 
    """

    def __init__(
        self,
        *,
        threshold: float = 0.98,
        **kwargs: Any,
        ) -> None:
        super().__init__()
        self.threshold = threshold
        self.sel_: list[bool] | None = None

    @override
    def _fit(  # type: ignore
        self, X: np.ndarray | torch.Tensor, categorical_features: list[int]
    ) -> list[int]:
        x_norm = None
        
        if isinstance(X, torch.Tensor):
            x_norm = F.normalize(X, p=2, dim=0).numpy() 
        else:
            x_norm = np.linalg.norm(X, axis=0)
        
        # Transpose to treat columns as vectors: shape (n_features, n_samples)
        vectors = x_norm.T

        d = vectors.shape[1]
        
        sel_ = [True] * vectors.shape[0]

        # Build FAISS index on columns as vectors
        index = faiss.IndexFlatIP(d)
        index.add(vectors)

        indices_to_skip = set()

        for idx, vector in enumerate(vectors) :
            if idx not in indices_to_skip:
                vector = vector.reshape(1, -1)
                # Perform range search to find all columns with similarity >= threshold
                lims, distances, labels = index.range_search(vector, thresh = self.threshold)
           
                similar_indices = labels[1:]
                index.remove_ids(similar_indices)
                indices_to_skip.update(similar_indices.tolist())

        for idx in indices_to_skip:
            sel_[idx] = False

        self.sel_ = sel_

        # ACT: udnerstend better what to return here
        return [
            new_idx
            for new_idx, idx in enumerate(np.where(sel_)[0])
            if idx in categorical_features
        ]

    @override
    def _transform(
        self, X: np.ndarray | torch.Tensor, *, is_test: bool = False
    ) -> np.ndarray:
        assert self.sel_ is not None, "You must call fit first"
        return X[:, self.sel_]
