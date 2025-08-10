"""
Utility module for the DiFF-RF+ algorithm.
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Inspired from an implementation of the Diff-RF algorithm provided at
# https://github.com/pfmarteau/DiFF-RF

import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import pairwise_distances
from bayes_opt import BayesianOptimization
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import KFold

import hdbscan


def calculate_nbins(size: float) -> int:
    """
    Calculate the number of bins for histogram-based entropy calculation.
    """
    return int(size / 8) + 2


def generate_feature_distribution(data):
    """
    Generate a distribution of feature importances based on entropy.

    This function calculates the weight for each feature based on its entropy,
    giving more weight to features with lower entropy (more structured data).
    """
    nbins = calculate_nbins(len(data))

    feature_distribution = []
    for i in range(np.shape(data)[1]):
        feature_distribution.append(weight_feature(data[:, i], nbins))
    feature_distribution = np.array(feature_distribution)

    return feature_distribution / (feature_distribution.sum() + 1e-5)


def split_column(column):
    """
    Find a random split value within the range of a column.
    """
    xmin = column.min()
    xmax = column.max()
    return np.random.uniform(xmin, xmax)


def similarity_score(instances, node, alpha):
    """
    Calculate similarity score between instances and node centroids.

    For each instance, calculates how similar it is to the closest centroid
    in the node, based on a distance formula and the alpha parameter.
    """
    if len(instances) == 0:
        return 0

    d = np.shape(instances)[1]

    # If node has cluster centroids, use them
    if hasattr(node, "centroids") and node.centroids is not None and len(node.centroids) > 0:
        min_distances = np.zeros(len(instances))
        for i, instance in enumerate(instances):
            sq_distances = np.sum(
                ((instance - node.centroids) / node.centroid_stds) ** 2 / d, axis=1
            )
            min_distances[i] = np.min(sq_distances)

        return 2 ** (-alpha * min_distances)
    else:
        similarity = (instances - node.avg) / node.std
        return 2 ** (-alpha * (np.sum((similarity * similarity) / d, axis=1)))


def empirical_entropy(hist):
    """
    Calculate the empirical entropy from a histogram.
    """
    h = np.asarray(hist, dtype=np.float64)
    if h.sum() <= 0 or (h < 0).any():
        return 0
    h = h / h.sum()
    return -(h * np.ma.log2(h)).sum()


def weight_feature(s, nbins):
    """
    Calculate the weight of a feature based on its entropy.
    """
    wmin = 0.02
    mins = s.min()
    maxs = s.max()

    if not np.isfinite(mins) or not np.isfinite(maxs) or np.abs(mins - maxs) < 1e-10:
        return 1e-4
    if mins == maxs:
        return 1e-4

    hist, _ = np.histogram(s, bins=nbins)
    ent = empirical_entropy(hist) / np.log2(nbins)

    if np.isfinite(ent):
        return max(1 - ent, wmin)

    return wmin


def default_clustering_params():
    """
    Return default clustering parameters for HDBSCAN.
    """
    return {"min_cluster_size": 5, "min_samples": None, "cluster_selection_epsilon": 0.0}


def optimize_clustering(data, sample_size, tree_height):
    """
    Determine optimal clustering parameters based on data characteristics and tree structure
    using Bayesian optimization to find the best HDBSCAN parameters.
    """

    n_features = data.shape[1]
    n_samples = len(data)

    avg_points_per_leaf = max(1, sample_size / (2**tree_height))
    min_clustering_points = max(10, int(avg_points_per_leaf / 2))

    hyperparams = {"model": "hdbscan", "min_clustering_points": min_clustering_points}

    if n_samples < 100 or min_clustering_points < 5:
        hyperparams["min_cluster_size"] = max(3, int(avg_points_per_leaf / 5))
        hyperparams["min_samples"] = max(2, int(min_clustering_points / 2))
        hyperparams["cluster_selection_epsilon"] = 0.5
        return hyperparams

    try:
        sample_sets = np.zeros((sample_size, min_clustering_points, n_features))

        nbrs = NearestNeighbors(n_neighbors=min_clustering_points).fit(data)
        random_indices = np.random.choice(len(data), size=sample_size, replace=False)

        _, indices = nbrs.kneighbors(data[random_indices])
        sample_sets = data[indices]

        max_min_cluster_size = max(5, min(20, min_clustering_points))
        max_min_samples = max(5, min(15, min_clustering_points // 2))

        def black_box_function(min_cluster_size, min_samples, cluster_selection_epsilon):

            min_cluster_size = max(2, int(min_cluster_size))
            min_samples = min(max(1, int(min_samples)), min_cluster_size)

            scores = []

            opt_hyperparams = {
                "min_cluster_size": min_cluster_size,
                "min_samples": min_samples,
                "cluster_selection_epsilon": cluster_selection_epsilon,
            }

            for leaf_set in sample_sets:
                centroids, labels, centroid_stds = cluster_data(leaf_set, opt_hyperparams)
                scores.append((centroid_stds**2).sum())

            return -np.mean(scores)

        pbounds = {
            "min_cluster_size": (2, max_min_cluster_size),
            "min_samples": (1, max_min_samples),
            "cluster_selection_epsilon": (0.0, 1.0),
        }

        optimizer = BayesianOptimization(
            f=black_box_function, pbounds=pbounds, random_state=42, verbose=0
        )

        optimizer.maximize(init_points=2, n_iter=3)

        best_params = optimizer.max["params"]

        hyperparams["min_cluster_size"] = max(2, int(best_params["min_cluster_size"]))
        hyperparams["min_samples"] = max(1, int(best_params["min_samples"]))
        hyperparams["cluster_selection_epsilon"] = best_params["cluster_selection_epsilon"]

    except Exception as e:
        # If optimization fails, fall back to default values
        hyperparams["min_cluster_size"] = max(3, int(avg_points_per_leaf / 5))
        hyperparams["min_samples"] = max(2, int(min_clustering_points / 2))
        hyperparams["cluster_selection_epsilon"] = 0.5

    return hyperparams


def cluster_data(data, hyperparams=None):
    """
    Perform clustering on data points.
    """
    if len(data) <= 1:
        return data, np.zeros(len(data)), np.ones((1, data.shape[1]))

    hyperparams = hyperparams or {}

    min_cluster_size = hyperparams.get("min_cluster_size", 5)
    min_samples = hyperparams.get("min_samples", None)
    cluster_selection_epsilon = hyperparams.get("cluster_selection_epsilon", 0.0)

    clustering = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
    )
    labels = clustering.fit_predict(data)

    unique_labels = np.unique(labels)
    if -1 in unique_labels:
        unique_labels = unique_labels[unique_labels != -1]

    if len(unique_labels) == 0:
        return (
            np.array([np.mean(data, axis=0)]),
            np.zeros(len(data)),
            np.array([np.std(data, axis=0) + 1e-6]),
        )

    centroids = np.array([np.mean(data[labels == label], axis=0) for label in unique_labels])

    n_clusters = len(centroids)
    centroid_stds = np.ones_like(centroids) * 1e-6

    for i in range(n_clusters):
        cluster_points = data[labels == i]
        if len(cluster_points) > 1:
            centroid_stds[i] = np.std(cluster_points, axis=0)
            centroid_stds[i][centroid_stds[i] < 1e-6] = 1e-6
        else:
            centroid_stds[i] = np.ones(data.shape[1]) * 1e-6

    return centroids, labels, centroid_stds
