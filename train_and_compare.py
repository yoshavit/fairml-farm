import numpy as np
import tensorflow as tf
import os
from algos import construct_classifier
from utils.data_utils import adult_dataset
from utils.misc import increment_path
import utils.tf_utils as U
hparams_list = []

# ========== EXPERIMENT PARAMETERS ==========
hparams_list.append({
    "classifier_type": "simplenn",
})
hparams_list.append({
    "classifier_type": "paritynn",
})
hparams_list.append({
    "classifier_type": "adversariallycensorednn"
})
n_epochs = 20

experiment_name = "comparisontest"
# ===========================================
masterdir = "/tmp/fairml-farm/"
datadir = masterdir + "data/"
try:
    tf.gfile.MakeDirs(datadir)
except tf.errors.OpError: # folder already exists
    pass
experiment_dir = increment_path(os.path.join(masterdir, "logs",
                                             experiment_name, "exp"))
os.makedirs(experiment_dir)
print("Logging experiments data to {}".format(experiment_dir))
print("Loading Adult dataset...")
train_dataset, validation_dataset = adult_dataset(datadir=datadir)
print("...dataset loaded.")
inputsize = train_dataset["data"].shape[1]
print("Launching Tensorboard.\nTo visualize, navigate to "
      "http://0.0.0.0:6006/\nTo close Tensorboard,"
      " press ctrl+C")
tensorboard_process = U.launch_tensorboard(experiment_dir)
for hparams in hparams_list:
    if "experiment_name" in hparams:
        logdir = os.path.join(experiment_dir, hparams["experiment_name"])
    else:
        logdir = increment_path(os.path.join(experiment_dir,
                                             hparams["classifier_type"]))
    expname = logdir.split('/')[-1] # minor note: logdir shouldn't end with '/'
    print("Starting new experiment, logged at {}".format(logdir))
    with tf.Graph().as_default():
        classifier = construct_classifier(hparams)
        print("======= Experiment hyperparameters =======\n{}".format(
            classifier.hparams))
        print("======= Training for {} epochs ===========".format(n_epochs))
        classifier.train(train_dataset, logdir, epochs=n_epochs,
                         validation_dataset=validation_dataset)
tensorboard_process.join()
