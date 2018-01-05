import tensorflow as tf
import os
from algos import construct_classifier
from utils.data_utils import get_dataset
from utils.misc import increment_path
import utils.tf_utils as U
hparams_list = []

# Run a series of side-by-side training comparisons (visualized in tensorboard)
# to understand the influence of the different fairness regularizers on
# performance + fairness metrics.

# ========== EXPERIMENT PARAMETERS ==========
lamda = 5
hparams_list.append({
    "classifier_type": "simplenn",
})
for hparam in ["dpe_scalar", "fnpe_scalar", "fppe_scalar", "cpe_scalar"]:
    hparams_list.append({
        "experiment_name": "with_" + hparam,
        "classifier_type": "paritynn",
        "dpe_scalar": 0.0,
    })
    hparams_list[-1][hparam] = lamda

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
train_dataset, validation_dataset = get_dataset("adult", datadir=datadir)
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
print("Experiments complete!")
tensorboard_process.join()
