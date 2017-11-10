import numpy as np
import tensorflow as tf
import os
from algos.baselines import SimpleNN
from utils.data_utils import get_adult_dataset

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
logdir = "/tmp/fairml-farm/test/"
tf.gfile.DeleteRecursively(logdir)
os.makedirs(logdir)
print("Loading Adult dataset...", end=" ")
train_dataset, validation_dataset, data_names = get_adult_dataset()
print("dataset loaded.")
inputsize = train_dataset["data"].shape[1]
print("Initializing classifier...")
with tf.variable_scope("c1"):
    classifier1 = SimpleNN(inputsize)
print("Training classifier...")
classifier1.train(train_dataset, logdir + "dropout", epochs=20, validation_dataset=validation_dataset)
tf.reset_default_graph()
print("Repeating the process with another classifier")
with tf.variable_scope("c2"):
    classifier2 = SimpleNN(inputsize, with_dropout=False)
classifier2.train(train_dataset, logdir + "no_dropout", epochs=20, validation_dataset=validation_dataset)
# classifier.save_model(os.path.join(logdir, "model.ckpt"))
print("To visualize, call\ntensorboard --logdir={}\nand navigate to\n"
      "http://0.0.0.0:6006/".format(logdir))
