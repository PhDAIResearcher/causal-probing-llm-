"""
Control experiments for causal probing.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from typing import List


def randomization(vectors, labels):
    """
    Control 1: Randomize vectors - accuracy should drop
    """
    num_layers = vectors.shape[0]
    random_acc = []

    for layer in range(num_layers):
        X = vectors[layer]
        # Shuffle vectors (shuffle samples)
        X_random = np.random.permutation(X)

        clf = LogisticRegression(max_iter=2000)
        scores = cross_val_score(clf, X_random, labels, cv=5)
        random_acc.append(scores.mean())

    return random_acc
