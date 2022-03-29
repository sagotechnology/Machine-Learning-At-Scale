#!/usr/bin/env python
"""
Helper functions for w261 Machine Learning at Scale Week 3 Demo.
"""
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot_iris_data(samples, labels, names, title='Iris Dataset',
                   colors=['turquoise', 'red', 'gray']):
    """
    2D PCA Plot for iris dataset.
    """
    components = PCA(n_components=2).fit(samples).transform(samples)

    plt.figure(figsize=(10, 6))

    for color, i, target_name in zip(colors, [0, 1, 2], names):
        plt.scatter(components[labels == i, 0],
                    components[labels == i, 1],
                    color=color,
                    alpha=.8,
                    lw=2,
                    label=target_name)
    plt.legend(loc='best', 
               scatterpoints=1)
    plt.title(title)

    plt.show()