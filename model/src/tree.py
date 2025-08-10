#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Inspired from an implementation of the Diff-RF algorithm provided at
# https://github.com/pfmarteau/DiFF-RF

import random
import numpy as np

from utility import generate_feature_distribution, split_column


class Node:
    """
    Node in the DiFF-RF decision tree.

    This class represents either an internal node (with split criteria)
    or a leaf node (with statistical information about the data it contains).
    """

    def __init__(
        self,
        data,
        height_limit,
        feature_distribution=None,
        sample_size=None,
        height=0,
        is_leaf=False,
    ):
        """
        Initialize a node in the DiFF-RF tree.
        """
        self.size = len(data)
        self.height_limit = height_limit
        self.height = height
        self.feature_distribution = feature_distribution
        self.is_leaf = is_leaf
        self.n_nodes = 1
        self.sample_size = sample_size
        self.freq = self.size / sample_size if sample_size else None

        if is_leaf:
            self._initialize_leaf(data)
        else:
            self._initialize_internal(data)

    def _initialize_internal(self, data):
        """
        Initialize an internal node by finding a split and creating child nodes.
        """
        if len(data) > 32:
            self.feature_distribution = generate_feature_distribution(data)

        cols = np.arange(np.shape(data)[1], dtype="int")
        self.split_feature = random.choices(cols, weights=self.feature_distribution)[0]
        column = data[:, self.split_feature]

        self.split_value = split_column(column)

        mask = column <= self.split_value
        data_left = data[mask, :]

        next_height = self.height + 1
        limit_reached = (
            self.height_limit <= next_height
            or data_left.shape[0] <= 5
            or np.all(data_left.max(0) == data_left.min(0))
        )

        self.left = Node(
            data_left if data_left.shape[0] else data,
            self.height_limit,
            self.feature_distribution,
            self.sample_size,
            next_height,
            is_leaf=limit_reached,
        )

        data_right = data[~mask, :]
        limit_reached = (
            self.height_limit <= next_height
            or data_right.shape[0] <= 5
            or np.all(data_right.max(0) == data_right.min(0))
        )

        self.right = Node(
            data_right if data_right.shape[0] else data,
            self.height_limit,
            self.feature_distribution,
            self.sample_size,
            next_height,
            is_leaf=limit_reached,
        )

        self.n_nodes = 1 + self.left.n_nodes + self.right.n_nodes

    def _initialize_leaf(self, data):
        """
        Initialize a leaf node with basic statistical information.
        """
        self.avg = np.mean(data, axis=0)
        if len(data) > 1:
            self.std = np.std(data, axis=0)
            self.std[self.std < 1e-6] = 1e-6
        else:
            self.std = np.ones(data.shape[1])
