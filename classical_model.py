"""Compatibility module for loading classical_best.pkl.

The original project saved a custom UnionVectorizer that stores
component vectorizers in a ``vectorizers`` dict and concatenates
their sparse transforms horizontally.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin


class UnionVectorizer(BaseEstimator, TransformerMixin):
    """Word + character TF-IDF feature union (pickle-compatible)."""

    def __init__(self, vectorizers=None):
        self.vectorizers = vectorizers or {}

    def fit(self, X, y=None):
        for name, vec in self.vectorizers.items():
            vec.fit(X, y)
        return self

    def transform(self, X):
        mats = []
        for name, vec in self.vectorizers.items():
            mats.append(vec.transform(X))
        if not mats:
            raise ValueError("UnionVectorizer has no component vectorizers")
        return sparse.hstack(mats).tocsr()

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


__all__ = ["UnionVectorizer"]
