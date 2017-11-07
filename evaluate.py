import numpy as np
import tensorflow as tf

""" Steps for evaluation:

    - if untrained: train for n epochs, otherwise load weights
    - run on validation dataset, and collect:
        - accuracy
        - prediction error
        - discrimination error
        - equalized-odds discrimination error
        - precision parity error
    - visualize distribution of outputs vs. true
    - visualize embedding of data
        - plot PCA'd 2D representations with matplotlib, with different
            labels/protecteds
        - plot PCA'd 3D representations in tensorboard, again distinguishing
            between labels/protecteds
    - visualize learned clusters
        - run kmeans on the dataset, then train a decision tree to predict
            cluster assignment?

    Later work: create "evaluate_hyperparams.py" to show tradeoffs with varying
        hparams
"""
