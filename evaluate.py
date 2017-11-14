import numpy as np
import tensorflow as tf
import os
import subprocess
from algos.baselines import SimpleNN
from utils.data_utils import get_adult_dataset
import utils.tf_utils as U

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
masterdir = "/tmp/fairml-farm/"
logdir = masterdir + "test/"
datadir = masterdir + "data/"
try:
    tf.gfile.MakeDirs(datadir)
except tf.errors.OpError: # folder already exists
    pass
tf.gfile.DeleteRecursively(logdir)
os.makedirs(logdir)
print("Loading Adult dataset...")
train_dataset, validation_dataset, data_names = get_adult_dataset(datadir=datadir)
print("...dataset loaded.")
inputsize = train_dataset["data"].shape[1]
print("Launching Tensorboard")
tensorboard_process = U.launch_tensorboard(logdir)
print("Initializing classifier...")
layersizes = [100, 100]
# classifier1 = SimpleNN(inputsize, layersizes=layersizes)
# print("Training classifier...")
# classifier1.train(train_dataset, logdir + "dropout", epochs=50, validation_dataset=validation_dataset)
# tf.reset_default_graph()
print("Repeating the process with another classifier")
classifier2 = SimpleNN(inputsize, layersizes=layersizes, with_dropout=False)
print("initialized")
classifier2.train(train_dataset, logdir + "no_dropout", epochs=50, validation_dataset=validation_dataset)
# classifier.save_model(os.path.join(logdir, "model.ckpt"))
print("To visualize, navigate to http://0.0.0.0:6006/\nTo close Tensoboard,"
      " press ctrl+C")
tensorboard_process.join()
