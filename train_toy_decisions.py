import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--varied-hparam', default='dpe_scalar',
                    choices=['dpe_scalar',
                             'fnpe_scalar',
                             'fppe_scalar',
                             'cpe_scalar'],
                    help="the hyperparameter to vary across experiments."
                    "dpe - demographic parity err, "
                    "fnpe - false negative parity err, "
                    "fppe - false positive parity err, "
                    "cpe - calibration parity err")
parser.add_argument('-epochs', type=int, default=200,
                    help="Number of epochs to train each classifier")
args = parser.parse_args()
import numpy as np
import tensorflow as tf
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from algos import construct_classifier
from toyexamples.synthetic_data import SquareBlock, ToyWorld

n_experiments = 10
varied_hparam_values = [10**i for i in np.linspace(-1, 2.3, n_experiments)]
lines_per_fn = 1
cmap = plt.cm.gist_rainbow
linecolors = [cmap(i) for i in np.linspace(0, 0.9, n_experiments)]

default_hparams = {
    "classifier_type": "paritynn",
    "dpe_scalar": 0.0,
    "layersizes": [],
    "inputsize": 2,
    "learning_rate": .1,
    "l2_penalty": 0.5,
    "batchsize": 128,
    "optimizer": tf.train.GradientDescentOptimizer,
}
hparams_list = []
for varied_hparam_value in varied_hparam_values:
    hparams_list.append(default_hparams.copy())
    hparams_list[-1][args.varied_hparam] = varied_hparam_value

uncertainty = 0.0
toyworld = ToyWorld()
bm1 = SquareBlock(0, [.0,.5], sizex=0.5, sizey=0.5, probpositive=1-uncertainty)
toyworld.add_block(bm1, 100)
bm2 = SquareBlock(0, [.0,.0], sizex=0.5, sizey=0.5, probpositive=0+uncertainty)
toyworld.add_block(bm2, 200)
bw1 = SquareBlock(1, [.0,.0], sizex=0.5, sizey=0.5, probpositive=0+uncertainty)
toyworld.add_block(bw1, 100)
bw2 = SquareBlock(1, [.5,.0], sizex=0.5, sizey=0.5, probpositive=1-uncertainty)
toyworld.add_block(bw2, 200)

handles = []
hs = toyworld.plot_points(n=300)
handles.extend(hs)

def contour_decision_function(classifier, X, Y):
    m, n = X.shape[0], X.shape[1]
    P = np.concatenate([np.reshape(X, [-1, 1]),
                        np.reshape(Y, [-1, 1])],
                       axis=1)
    Z = classifier.predict(P)
    return np.reshape(Z, [m, n])

test_lines = np.array([
    [1, 1, -1],
    [1,-1, 0],
    [0, 1, .0]])

train_dataset, validation_dataset = toyworld.dataset(n=2000)
for i, hparams in enumerate(hparams_list):
    print("Starting new experiment")
    # if i == len(test_lines): break
    with tf.Graph().as_default():
        with tf.Session() as sess:
            classifier = construct_classifier(hparams, sess=sess)
            print("======= Experiment hyperparameters"
                  "=======\n{}".format(classifier.hparams))
            print("======= Training for {} epochs"
                  "===========".format(args.epochs))
            classifier.train(train_dataset, epochs=args.epochs,
                             validation_dataset=validation_dataset)
            # w, b = classifier.extract_final_layer_weights()
            # tl = test_lines[i]
            # sess.run([tf.assign(w, np.expand_dims(tl[:2], axis=1)),
                      # tf.assign(b, np.expand_dims(tl[2], axis=0))])
            # classifier.validate(validation_dataset)
            # metric_dict = sess.run(classifier.metric_dict)
            # printed_metrics = ["val_crossentropy",
                               # "val_demographic_parity_error",
                               # "val_calibration_parity_error"]
            # msg = ", ".join(["{} = {:0.4}".format(pm, metric_dict[pm])
                             # for pm in printed_metrics])
            # print(msg)
            # plot resulting functions
            if lines_per_fn == 1:
                w, b = classifier.extract_final_layer_weights()
                w, b = sess.run([w, b])
                print("Decision boundary parameters:"
                      " a = {:3.3}, b = {:3.3}, c = {:3.3}".format(
                          w[0,0], w[1,0], b[0]))
                hs = toyworld.plot_line(w[0,0], w[1,0], b[0],
                                        label=hparams[args.varied_hparam],
                                        color=linecolors[i])
                handles.extend(hs)
            else:
                cdf = lambda X, Y: contour_decision_function(classifier, X, Y)
                hs = toyworld.plot_contour(cdf, lines=1, color=[linecolors[i]])
                handles.extend(hs)
plt.suptitle("Decision boundaries for varying {}".format(args.varied_hparam))
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()
