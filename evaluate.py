import numpy as np
import tensorflow as tf
import os
from algos.baselines import SimpleNN, ParityNN
from utils.data_utils import adult_dataset
from utils.misc import increment_path
import utils.tf_utils as U
from utils.vis_utils import plot_embeddings

""" Steps for evaluation:

    - if untrained: train for n epochs, otherwise load weights
    - run on validation dataset, and collect:
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
train = True
run_id = False
# run_id = "run--00"
assert train or run_id, "if not training then specify directory"
experiment_name = "test"

masterdir = "/tmp/fairml-farm/"
datadir = masterdir + "data/"
try:
    tf.gfile.MakeDirs(datadir)
except tf.errors.OpError: # folder already exists
    pass
if run_id:
    logdir = os.path.join(masterdir, "logs", experiment_name, run_id)
else:
    logdir = increment_path(os.path.join(masterdir, "logs", experiment_name, "run"))
    os.makedirs(logdir)
print("Logging data to {}".format(logdir))
print("Loading Adult dataset...")
train_dataset, validation_dataset = adult_dataset(datadir=datadir)
# train_dataset, validation_dataset = adult_dataset(
    # datadir=datadir,
    # removable_columns=["sex", "capital-gain", "income>50k"],
    # objective=lambda s: int(s['capital-gain']>0))
print("...dataset loaded.")
inputsize = train_dataset["data"].shape[1]
print("Launching Tensorboard.\nTo visualize, navigate to "
      "http://0.0.0.0:6006/\nTo close Tensorboard,"
      " press ctrl+C")
tensorboard_process = U.launch_tensorboard(logdir)
print("Initializing classifier...")
layersizes = [100, 100]
c = SimpleNN()
c.build(inputsize, hparams={"layersizes":layersizes})
if train:
    print("Training network...")
    c.train(train_dataset, logdir, epochs=50,
            validation_dataset=validation_dataset)
    savepath = c.save_model(os.path.join(logdir, "model.ckpt"))
    assert savepath == tf.train.latest_checkpoint(logdir), "Paths unequal: {}, {}".format(savepath, tf.train.latest_checkpoint(logdir))
savepath = tf.train.latest_checkpoint(logdir)
c.load_model(savepath)
val_embeddings = c.compute_embedding(validation_dataset["data"])
plot_embeddings(val_embeddings,
                validation_dataset["label"],
                validation_dataset["protected"],
                plot3d=True,
                subsample=200,
                label_names=["income>50k", "income<=50k"],
                protected_names=["female", "male"])
tensorboard_process.join()
