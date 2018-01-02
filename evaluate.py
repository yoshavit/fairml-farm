import numpy as np
import tensorflow as tf
import os
from algos.baselines import SimpleNN, ParityNN, AdversariallyCensoredNN
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
print("...dataset loaded.")
inputsize = train_dataset["data"].shape[1]
print("Launching Tensorboard.\nTo visualize, navigate to "
      "http://0.0.0.0:6006/\nTo close Tensorboard,"
      " press ctrl+C")
tensorboard_process = U.launch_tensorboard(logdir)
print("Initializing classifier...")
layersizes = [100, 100]
c = ParityNN()
c.build(hparams={"inputsize": inputsize,
                            "layersizes":layersizes,
                            "adv_sees_label": True,
                           })
if train:
    print("Training network...")
    c.train(train_dataset, logdir, epochs=20,
            validation_dataset=validation_dataset)
    savepath = c.save_model(os.path.join(logdir, "model.ckpt"))
    assert savepath == tf.train.latest_checkpoint(logdir), "Paths unequal: {}, {}".format(savepath, tf.train.latest_checkpoint(logdir))

# ======= Plot out the learned embedding space ===============
visualize = False
if visualize:
    savepath = tf.train.latest_checkpoint(logdir)
    c.load_model(savepath)
    # get an equal number of male and female points
    n = validation_dataset["label"].shape[0]
    n_males = sum(validation_dataset["label"])
    limiting_gender = n_males > n - n_males # 1 if men, 0 if women
    n_limiting_gender = sum(validation_dataset["label"] == limiting_gender)
    n_per_gender = min(500, n_limiting_gender)
    inds = np.concatenate([
        np.where(validation_dataset["label"] == limiting_gender)[0][:n_per_gender],
        np.where(validation_dataset["label"] != limiting_gender)[0][:n_per_gender]],
        axis=0)
    vis_dataset = {k:v[inds, ...] for k, v in validation_dataset.items()}
    val_embeddings = c.compute_embedding(vis_dataset["data"])
    plot_embeddings(val_embeddings,
                    vis_dataset["label"],
                    vis_dataset["protected"],
                    plot3d=True,
                    subsample=False,
                    label_names=["income<=50k", "income>50k"],
                    protected_names=["female", "male"])
tensorboard_process.join()
