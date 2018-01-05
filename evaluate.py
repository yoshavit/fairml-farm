import os
import argparse
import numpy as np
from algos import construct_classifier, classifier_types
from utils.data_utils import get_dataset, dataset_names
from utils.misc import increment_path
from utils.tf_utils import launch_tensorboard
from utils.vis_utils import plot_embeddings

masterdir = "/tmp/fairml-farm/"
base_datadir = masterdir + "data/"
base_logdir = masterdir + "logs/"
parser = argparse.ArgumentParser(description="Evaluate an individual fairness "
                                 "algorithm.\nNOTE: classifier-specific "
                                 "arguments should be specified in area "
                                 "in the script itself.")
parser.add_argument("--experiment-name", default="default",
                    help="Name for the experiment base directory, "
                    "used as the extension relative to {}".format(base_logdir))
parser.add_argument("--load-dir",
                    help="Path to a previous experiment subdirectory, used to "
                    "load model weights, relative to {}.".format(base_logdir))
parser.add_argument("--train", action="store_true",
                    help="train the classifier")
parser.add_argument("-epochs", type=int, default=20)
parser.add_argument("--visualize", action="store_true", help="visualize "
                    "learned latent space")
parser.add_argument("-classifier", choices=[c.name for c in classifier_types],
                    default="simplenn",
                    help="Name of the type of fairness algorithm to use.")
parser.add_argument("-dataset", choices=dataset_names,
                    default="adult",
                    help="Name of dataset to train on.")
args = parser.parse_args()

loaddir = None
if args.load_dir is not None:
    loaddir = os.path.join(base_logdir, args.load_dir)
logdir = increment_path(os.path.join(base_logdir, args.experiment_name, "run"))
os.makedirs(logdir, exist_ok=True)
print("Logging data to {}".format(logdir))
print("Loading {} dataset...".format(args.dataset))
train_dataset, validation_dataset = get_dataset(args.dataset,
                                                base_datadir=base_datadir)
print("Launching Tensorboard.\nTo visualize, navigate to "
      "http://0.0.0.0:6006/\nTo close Tensorboard,"
      " press ctrl+C")
tensorboard_process = launch_tensorboard(logdir)
# ===== SPECIFY HYPERPARAMETERS (INCLUDING CLASSIFIER-TYPE) =====
inputsize = train_dataset["data"].shape[1]
layersizes = [100]
classifier_type = "paritynn"
hparams = {
    "classifier_type": classifier_type,
    "layersizes": layersizes,
    "inputsize": inputsize,
}
# ===============================================================
print("Initializing classifier...")
classifier = construct_classifier(hparams, loaddir=loaddir)
if args.train:
    print("Training network...")
    classifier.train(train_dataset, logdir, epochs=args.epochs,
                     validation_dataset=validation_dataset)
savepath = classifier.save_model(logdir)
if args.visualize: # Plot out the learned embedding space
    n = validation_dataset["label"].shape[0]
    # get an equal number of male and female points
    n_males = sum(validation_dataset["label"])
    limiting_gender = n_males > n - n_males # 1 if men, 0 if women
    n_limiting_gender = sum(validation_dataset["label"] == limiting_gender)
    max_points_per_gender = 500
    n_per_gender = min(max_points_per_gender, n_limiting_gender)
    inds = np.concatenate([
        np.where(validation_dataset["label"] == limiting_gender)[0][:n_per_gender],
        np.where(validation_dataset["label"] != limiting_gender)[0][:n_per_gender]],
        axis=0)
    vis_dataset = {k:v[inds, ...] for k, v in validation_dataset.items()}
    val_embeddings = classifier.compute_embedding(vis_dataset["data"])
    plot_embeddings(val_embeddings,
                    vis_dataset["label"],
                    vis_dataset["protected"],
                    plot3d=True,
                    subsample=False,
                    label_names=["income<=50k", "income>50k"],
                    protected_names=["female", "male"])
tensorboard_process.join()
