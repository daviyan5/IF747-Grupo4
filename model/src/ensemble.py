#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Inspired from an implementation of the Diff-RF algorithm provided at
# https://github.com/pfmarteau/DiFF-RF

from multiprocessing import Pool
from functools import partial

import numpy as np
from tqdm import tqdm

from tree import Node
from utility import (
    generate_feature_distribution,
    similarity_score,
)
from sklearn.base import BaseEstimator, OutlierMixin


class DiFF_RF_Wrapper(BaseEstimator, OutlierMixin):
    def __init__(self, n_trees=100, sample_size=256, alpha=1.0):
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.alpha = alpha
        self.model = DiFF_RF(sample_size=self.sample_size, n_trees=self.n_trees)

    def fit(self, X, y=None):
        actual_sample_size = min(self.sample_size, len(X))
        self.model = DiFF_RF(sample_size=actual_sample_size, n_trees=self.n_trees)
        self.model.fit(X, n_jobs=-1)
        return self

    def decision_function(self, X, alpha=None):
        return self.model.anomaly_score(X, alpha=alpha if alpha else self.alpha)

    def predict(self, X):
        scores = self.decision_function(X)
        return (scores > 0.5).astype(int)


class DiFF_RF:
    """
    Distance-based and Frequency-based Forest (DiFF-RF) for anomaly detection.

    This algorithm builds an ensemble of decision trees to detect anomalies
    based on both distance metrics and frequency of occurrence patterns.
    """

    def __init__(self, sample_size: int, n_trees: int = 10):
        """
        Initialize the DiFF-RF model.
        """
        self.sample_size = sample_size
        self.n_trees = n_trees
        self.alpha = 1.0

        self.data = None
        self.trees = None
        self.feature_distribution = []
        self.test_size = 1

        self.pointwise_scores = np.zeros((1, self.n_trees))
        self.frequency_scores = np.zeros((1, self.n_trees))
        self.collective_scores = np.zeros((1, self.n_trees))

    @staticmethod
    def calculate_height_limit(sample_size: float) -> float:
        """
        Calculate the height limit for trees based on sample size.
        This is a common heuristic for isolation-based algorithms.
        """
        return 1.0 * np.ceil(np.log2(sample_size))

    def create_trees(
        self,
        data,
        feature_distribution: np.ndarray,
        sample_size: int,
        height_limit: float,
    ) -> Node:
        """
        Create a single tree for the forest.
        """
        rows = np.random.choice(len(data), sample_size, replace=False)
        return Node(
            data[rows, :],
            height_limit,
            feature_distribution,
            sample_size=sample_size,
        )

    def fit(self, data: np.ndarray, n_jobs: int = 1):
        """
        Fit the DiFF-RF model on the training data.
        """
        self.data = data

        self.sample_size = min(self.sample_size, len(data))

        height_limit = self.calculate_height_limit(self.sample_size)
        self.feature_distribution = generate_feature_distribution(data)

        if n_jobs > 1:
            create_tree_partial = partial(
                self.create_trees,
                feature_distribution=self.feature_distribution,
                sample_size=self.sample_size,
                height_limit=height_limit,
            )
            with Pool(n_jobs) as p:
                self.trees = list(
                    tqdm(
                        p.imap(create_tree_partial, [data for _ in range(self.n_trees)]),
                        total=self.n_trees,
                        desc="Creating Trees",
                        leave=False,
                    )
                )
        else:
            self.trees = [
                self.create_trees(data, self.feature_distribution, self.sample_size, height_limit)
                for _ in tqdm(range(self.n_trees), desc="Creating Trees", leave=False)
            ]
        return self

    def walk(self, data: np.ndarray) -> np.ndarray:
        """
        Process data through all trees in the forest.
        """
        self.pointwise_scores = np.zeros((len(data), self.n_trees))
        self.frequency_scores = np.zeros((len(data), self.n_trees))
        self.collective_scores = np.zeros((len(data), self.n_trees))

        for tree_idx, tree in enumerate(self.trees):
            # Uses a boolean mask to track which instances are in which node
            cur_idx = np.ones(len(data), dtype=bool)
            self.walk_tree(
                tree, tree_idx, cur_idx, data, self.feature_distribution, alpha=self.alpha
            )

    def walk_tree(self, node, tree_idx, cur_idx, data, feature_distribution, alpha):
        """
        Traverse a single tree for the given data points.
        """
        if node.is_leaf:
            instances = data[cur_idx]
            f = ((node.size + 1) / self.sample_size) / ((1 + len(instances)) / self.test_size)

            if alpha == 0:
                self.pointwise_scores[cur_idx, tree_idx] = 0
                self.frequency_scores[cur_idx, tree_idx] = -f
                self.collective_scores[cur_idx, tree_idx] = -f
            else:
                z = similarity_score(instances, node, alpha)
                self.pointwise_scores[cur_idx, tree_idx] = z
                self.frequency_scores[cur_idx, tree_idx] = -f
                self.collective_scores[cur_idx, tree_idx] = z * f
        else:
            left_idx = (data[:, node.split_feature] <= node.split_value) & cur_idx
            self.walk_tree(node.left, tree_idx, left_idx, data, feature_distribution, alpha)

            right_idx = (data[:, node.split_feature] > node.split_value) & cur_idx
            self.walk_tree(node.right, tree_idx, right_idx, data, feature_distribution, alpha)

    def anomaly_score(self, data: np.ndarray, alpha=1) -> dict:
        """
        Calculate anomaly scores for input data.
        """
        self.test_size = len(data)
        self.alpha = alpha

        self.walk(data)

        scores = {
            "pointwise": -self.pointwise_scores.mean(1),
            "frequency": self.frequency_scores.mean(1),
            "collective": -self.collective_scores.mean(1),
        }
        return scores

    def predict(
        self, data: np.ndarray, threshold: float, score_type: str = "collective"
    ) -> np.ndarray:
        """
        Predict anomalies in the input data using a given threshold.
        """
        if score_type not in ["pointwise", "frequency", "collective"]:
            raise RuntimeError(
                "Invalid score type. Please choose from: 'pointwise', 'frequency', 'collective'"
            )
        scores = self.anomaly_score(data)
        out = scores[score_type] >= threshold
        return out.astype(int)
