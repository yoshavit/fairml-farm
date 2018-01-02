import tensorflow as tf
import sys
sys.path.append('../')
print(sys.path)
from algos import construct_classifier
from utils.misc import increment_path
from toyexamples.synthetic_data import SquareBlock, ToyWorld

n_epochs = 10
hparams_list = [
    {"classifier_type": "paritynn",
     "dpe_scalar": 10**i,
     "layersizes": [],
     "inputsize": 2,
    }
    for i in range(-5, 2)
]
b1 = SquareBlock(0, [0,0], probpositive=0.8)
b2 = SquareBlock(1, [1,0], probpositive=0.5)
toyworld = ToyWorld()
toyworld.add_block(b1, .7)
toyworld.add_block(b2, .3)

train_dataset, validation_dataset = toyworld.dataset()
print("...dataset loaded.")
print("Launching Tensorboard.\nTo visualize, navigate to "
      "http://0.0.0.0:6006/\nTo close Tensorboard,"
      " press ctrl+C")
for hparams in hparams_list:
    print("Starting new experiment")
    with tf.Graph().as_default():
        classifier = construct_classifier(hparams)
        print("======= Experiment hyperparameters =======\n{}".format(
            classifier.hparams))
        print(tf.global_variables())
        raise RuntimeError
        print("======= Training for {} epochs ===========".format(n_epochs))
        classifier.train(train_dataset, epochs=n_epochs,
                         validation_dataset=validation_dataset)
