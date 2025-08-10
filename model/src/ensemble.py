#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Inspired from an implementation of the Diff-RF algorithm provided at
# https://github.com/pfmarteau/DiFF-RF

from multiprocessing import Pool
from functools import partial

import numpy as np
import pandas as pd

from tqdm import tqdm

from tree import Node
from utility import (
    generate_feature_distribution,
    optimize_clustering,
    default_clustering_params,
    similarity_score,
)


def calculate_alpha(data, n_trees, sample_size, n_iter=5):
    """
    Calculate optimal alpha parameter for the DiFF-RF model.

    This function searches for the best alpha value that minimizes
    the average rank of anomalies across multiple iterations.
    """
    possible_values = [1e-12, 1e-9, 1e-6, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.5, 1, 2, 5, 10, 100]
    r_alpha = {alpha: 0.0 for alpha in possible_values}

    reduction_value = 0.01
    # Choose len(data) * reduction_value values
    data_used = data.sample(frac=reduction_value, random_state=42)
    sample_size *= reduction_value
    sample_size = int(sample_size)

    num_parts = max(1, len(data_used) // sample_size)
    partitions = [data_used.loc[idx] for idx in np.array_split(data_used.index, num_parts)]

    # For each iteration
    for _ in tqdm(range(n_iter), desc="Iterations"):
        for i, p_i in tqdm(
            enumerate(partitions), desc="Partitions", total=len(partitions), leave=False
        ):
            x_i = pd.concat(
                [partitions[j] for j in range(len(partitions)) if j != i], ignore_index=True
            )

            for alpha in tqdm(
                possible_values, desc="Alpha values", total=len(possible_values), leave=False
            ):
                diff_rf = DiFF_RF_Plus(sample_size=sample_size, n_trees=n_trees)
                diff_rf.fit(x_i.values, n_jobs=16)

                scores_x = np.array(diff_rf.anomaly_score(x_i.values, alpha=alpha)["pointwise"])
                scores_p = np.array(diff_rf.anomaly_score(p_i.values, alpha=alpha)["pointwise"])

                delta = 0
                for perc in [95, 96, 97, 98, 99]:
                    quantile_value = np.percentile(scores_x, perc)
                    count = np.sum(scores_p > quantile_value)
                    delta += count * (100 - perc)
                r_alpha[alpha] += delta

    total_count = n_iter * len(partitions)
    for alpha in possible_values:
        r_alpha[alpha] /= total_count

    best_alpha = min(r_alpha, key=r_alpha.get)
    return best_alpha


def calculate_hyperparameters(data: np.ndarray) -> dict:
    """
    Calculate optimal hyperparameters for the DiFF-RF model.
    """
    n_trees = 256
    sample_size_ratio = 0.25
    sample_size = int(len(data) * sample_size_ratio)
    alpha = calculate_alpha(data, n_trees, sample_size)

    kwargs = {
        "sample_size": sample_size,
        "n_trees": n_trees,
        "alpha": alpha,
    }
    return kwargs


class DiFF_RF_Plus:
    """
    Distance-based and Frequency-based Forest (DiFF-RF) for anomaly detection.

    This algorithm builds an ensemble of decision trees to detect anomalies
    based on both distance metrics and frequency of occurrence patterns.
    """

    def __init__(self, sample_size: int, n_trees: int = 10, debug: bool = False):
        """
        Initialize the DiFF-RF model.
        """
        self.sample_size = sample_size
        self.n_trees = n_trees
        self.debug = debug
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
        """
        return 1.0 * np.ceil(np.log2(sample_size) - np.log2(np.log2(sample_size)))

    def create_trees(
        self,
        data,
        feature_distribution: np.ndarray,
        sample_size: int,
        height_limit: float,
        hyperparams: dict,
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
            hyperparams=hyperparams,
        )

    def fit(self, data: np.ndarray, n_jobs: int = 1, optimize_clusters: bool = False):
        """
        Fit the DiFF-RF model on the training data.
        """
        self.data = data

        self.sample_size = min(self.sample_size, len(data))

        height_limit = self.calculate_height_limit(self.sample_size)
        self.feature_distribution = generate_feature_distribution(data)

        if optimize_clusters:
            hyperparams = optimize_clustering(data, self.sample_size, height_limit)
        else:
            hyperparams = default_clustering_params()

        if n_jobs > 1:
            create_tree_partial = partial(
                self.create_trees,
                feature_distribution=self.feature_distribution,
                sample_size=self.sample_size,
                height_limit=height_limit,
                hyperparams=hyperparams,
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
                self.create_trees(
                    data, self.feature_distribution, self.sample_size, height_limit, hyperparams
                )
                for _ in tqdm(range(self.n_trees), desc="Creating Trees", leave=False)
            ]
        return self

    def walk(self, data: np.ndarray) -> np.ndarray:
        """
        Process data through all trees in the forest.
        """
        self.pointwise_scores.resize((len(data), self.n_trees))
        self.frequency_scores.resize((len(data), self.n_trees))
        self.collective_scores.resize((len(data), self.n_trees))

        # We can parallelize this loop for better performance
        for tree_idx, tree in enumerate(self.trees):
            # Uses boolean mask to track valid indices at each split
            cur_idx = np.ones(len(data)).astype(bool)
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
            left_idx = (data[:, node.split_feature] <= node.split_value) * cur_idx
            self.walk_tree(node.left, tree_idx, left_idx, data, feature_distribution, alpha)

            right_idx = (data[:, node.split_feature] > node.split_value) * cur_idx
            self.walk_tree(node.right, tree_idx, right_idx, data, feature_distribution, alpha)

    def anomaly_score(self, data: np.ndarray, alpha=1) -> dict:
        """
        Calculate anomaly scores for input data.
        """
        self.test_size = len(data)
        self.alpha = alpha

        # Evaluate the scores for each of the observations.
        self.walk(data)

        # Compute the scores from the path lengths
        scores = {
            "pointwise": -self.pointwise_scores.mean(1),
            "frequency": self.frequency_scores.mean(1),
            "collective": -self.collective_scores.mean(1),
        }
        return scores

    def predict(
        self, data: np.ndarray, threshold: float, score_type: str = "pointwise"
    ) -> np.ndarray:
        """
        Predict anomalies in the input data.
        """
        if score_type not in ["pointwise", "frequency", "collective"]:
            raise RuntimeError(
                "Invalid score type. Please choose from: 'pointwise', 'frequency', 'collective'"
            )
        scores = self.anomaly_score(data)
        out = scores[score_type] >= threshold
        return out * 1
