from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf
import time

import utils.tf_utils as U

class BaseClassifier(ABC):
    """Abstract class for binary classifiers.
    """
    @abstractmethod
    def load_model(filename):
        """ Loads the model parameters from file filename
        """
        raise NotImplementedError
    @abstractmethod
    def save_model(logdir):
        """ Saves the model parameters
        Returns the filename for the stored parameters
        """
        raise NotImplementedError
    @abstractmethod
    def predict(X):
        """
        Arguments:
            X: an n x d matrix of datapoints
        Returns:
            Yhat: an n x 1 vector of predicted classifications, each from 0 to 1
        """
        raise NotImplementedError
    @abstractmethod
    def compute_embedding(X):
        """
        Arguments:
            X: an n x d matrix of datapoints
        Returns:
            Z: an n x r lower-dimensional embedding of each of the datapoints
        """
        raise NotImplementedError

class SimpleNN(BaseClassifier):
    def __init__(self, sess=None):
        if sess is None:
            self.sess = tf.Session()
        self.built = False

    def check_built(self):
        assert self.built, "Classifier not yet built; Call obj.build(...) before using"

    def default_hparams(self):
        return {
            "layersizes": [100,00],
            "batchsize": 32,
            "with_dropout":True,
        }

    def build(self, inputsize, hparams={}):
        self.hparams = self.default_hparams()
        self.hparams.update(hparams)
        # ========== Build training pipeline ===================
        self.inputsize=inputsize
        self.dataset = tf.data.Dataset()
        self.dataset_X = tf.placeholder(dtype=tf.float32, shape=[None, self.inputsize],
                                      name="dataset_X")
        self.dataset_Y = tf.placeholder(tf.bool, shape=[None], name="dataset_Y")
        self.dataset_A = tf.placeholder(tf.bool, shape=[None], name="dataset_A")
        self.keep_prob = tf.placeholder(dtype=tf.float32)
        dataset = tf.data.Dataset.from_tensor_slices(
            (self.dataset_X, self.dataset_Y, self.dataset_A
            )).shuffle(100, reshuffle_each_iteration=True).batch(
                self.hparams["batchsize"])
        self.training_iterator = dataset.make_initializable_iterator()
        x, y, a = self.training_iterator.get_next()
        opt_ops, yhat, embedding, metrics, metric_names =\
                self.build_training_pipeline(x, y, a)
        # Make copies of the metrics, and store different avgs for the train
        # and validation copies
        train_metrics = [tf.identity(m) for m in metrics]
        ema = tf.train.ExponentialMovingAverage(0.99)
        self.global_step = tf.Variable(0, trainable=False)
        inc_global_step = tf.assign_add(self.global_step,
                                        self.hparams["batchsize"])
        with tf.control_dependencies(opt_ops + [inc_global_step]):
            self.train_op = U.ema_apply_wo_nans(ema, train_metrics)
        self.train_summaries = tf.summary.merge([
            tf.summary.scalar(metric_name, ema.average(metric), family="train")
            for metric_name, metric in zip(metric_names, train_metrics)])

        val_metrics =  [tf.identity(m) for m in metrics]
        # We also compute the moving-average for validation error
        self.validation_op = U.ema_apply_wo_nans(ema, val_metrics)
        # Reinitialize the moving average (i.e. return to 0.0) for each epoch
        # of the validation metrics
        self.restart_validation_op = tf.variables_initializer([
            ema.average(metric) for metric in val_metrics])
        self.val_summaries = tf.summary.merge([
            tf.summary.scalar(metric_name, ema.average(metric), family="val")
            for metric_name, metric in zip(metric_names, val_metrics)])

        # ======= prediction pipeline ================
        self.prediction_x = tf.placeholder(tf.float32, shape=[None, inputsize],
                                      name="prediction_x")
        self.prediction_iterator = tf.data.Dataset.from_tensor_slices(
            self.prediction_x).batch(self.hparams["batchsize"]).make_initializable_iterator()
        x = self.prediction_iterator.get_next()
        self.prediction_yhat, _, self.prediction_embedding = self.build_network(
            x, reuse=True)
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        self.built = True

    def build_training_pipeline(self, x, y, a):
        """Returns loss metrics, metric_names, TODO: more
        """
        yhat, yhat_logits, embedding = self.build_network(x)
        loss = self.build_loss(x, y, a, yhat_logits)
        metrics, metric_names = self.build_metrics(x, y, a, yhat_logits)
        metrics = [loss] + metrics
        metric_names = ['overall_loss'] + metric_names
        opt_ops = [tf.train.AdamOptimizer().minimize(loss)]
        return opt_ops, yhat, embedding, metrics, metric_names

    def build_loss(self, x, y, a, yhat_logits):
        crossentropy = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(y, tf.float32),
                                                    logits=yhat_logits))
        return crossentropy

    def build_metrics(self, x, y, a, yhat_logits):
        yhat = tf.sigmoid(yhat_logits)
        crossentropy = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(y, tf.float32),
                                                    logits=yhat_logits))
        accuracy = tf.reduce_mean(tf.cast(
            tf.equal(tf.greater(yhat_logits, 0.0), y),
            tf.float32))
        dpe = U.demographic_parity_discrimination(yhat, a)
        tppe, fppe = U.equalized_odds_discrimination(yhat, a, y)
        metrics = [crossentropy, accuracy, dpe, tppe, fppe]
        metric_names = ["crossentropy",
                        "accuracy",
                        "demographic_parity_error",
                        "true_positive_parity_error",
                        "false_positive_parity_error"]
        return metrics, metric_names

    def build_network(self, x, reuse=False):
        z = x
        with tf.variable_scope("classifier", reuse=reuse):
            for i, l in enumerate(self.hparams["layersizes"]):
                z = U.lrelu(U._linear(z, l, "fc{}".format(i)))
            embedding = z
            if self.hparams["with_dropout"]:
                z = tf.nn.dropout(z, self.keep_prob)
            yhat_logits = tf.squeeze(U._linear(z, 1, "output_logits"), axis=1)
            yhat = tf.sigmoid(yhat_logits)
            return yhat, yhat_logits, embedding

    def train(self, training_dataset, logdir, epochs=10, keep_prob=0.5,
              validation_dataset=None, validation_batches=None):
        self.check_built()
        # if validation_batches is None, runs all batches
        sw = tf.summary.FileWriter(logdir)
        for epoch in range(epochs):
            self.sess.run(self.training_iterator.initializer,
                          feed_dict={
                              self.dataset_X: training_dataset["data"],
                              self.dataset_Y: training_dataset["label"],
                              self.dataset_A: training_dataset["protected"]})
            print("Epoch {}...".format(epoch), end=" ")
            starttime = time.clock()
            steps = 0
            while True:
                steps += 1
                try:
                    self.sess.run(self.train_op, feed_dict={self.keep_prob: keep_prob})
                    if steps % 100 == 0:
                        sw.add_summary(self.sess.run(self.train_summaries),
                                       self.sess.run(self.global_step))
                except tf.errors.OutOfRangeError:
                    break
            sw.add_summary(self.sess.run(self.train_summaries),
                           self.sess.run(self.global_step))
            # validation stats
            if validation_dataset is not None:
                self.sess.run(self.training_iterator.initializer,
                              feed_dict={
                                  self.dataset_X: validation_dataset["data"],
                                  self.dataset_Y: validation_dataset["label"],
                                  self.dataset_A: validation_dataset["protected"]})
                steps = 0
                if validation_batches is None: validation_batches = float("inf")
                self.sess.run(self.restart_validation_op)
                while steps < validation_batches:
                    steps +=1
                    try:
                        self.sess.run(self.validation_op, feed_dict={self.keep_prob: 1.0})
                    except tf.errors.OutOfRangeError:
                        break
                sw.add_summary(self.sess.run(self.val_summaries),
                               self.sess.run(self.global_step))
            sw.flush()
            print("complete after {:0.2f} seconds.".format(time.clock() -
                                                      starttime))

    def load_model(self, filename):
        self.check_built()
        self.saver.restore(self.sess, filename)

    def save_model(self, filepath):
        self.check_built()
        filename = self.saver.save(self.sess, filepath)
        return filename

    def predict(self, x):
        """Compute the classifier output
        Args:
            X - n x d array of datapoints
        Returns:
            Z - n x r array of embeddings of datapoints
        """
        self.check_built()
        yhatbatched = []
        self.sess.run(self.prediction_iterator.initializer,
                      feed_dict={self.prediction_x: x})
        while True:
            try:
                yhatbatch = self.sess.run(self.prediction_yhat,
                                          feed_dict={self.keep_prob:1.0})
                yhatbatched.append(yhatbatch)
            except tf.errors.OutOfRangeError:
                break
        return np.concatenate(yhatbatched, axis=0)

    def compute_embedding(self, x):
        """Compute the low-dimensional embedding (learned by the classifier)
        of a set of datapoints. In this case, that is the last layer of the
        network before the output.
        Args:
            X - n x d array of datapoints
        Returns:
            Z - n x r array of embeddings of datapoints
        """
        self.check_built()
        yhatbatched = []
        self.sess.run(self.prediction_iterator.initializer,
                      feed_dict={self.prediction_x: x})
        while True:
            try:
                yhatbatch = self.sess.run(self.prediction_embedding,
                                          feed_dict={self.keep_prob:1.0})
                yhatbatched.append(yhatbatch)
            except tf.errors.OutOfRangeError:
                break
        return np.concatenate(yhatbatched, axis=0)

class ParityNN(SimpleNN):
    def default_hparams(self):
        hparams = super().default_hparams()
        hparams.update({
            "dpe_scalar": 1.0,
            "tppe_scalar": 0.0,
            "fppe_scalar": 0.0,
        })
        return hparams

    def build_loss(self, x, y, a, yhat_logits):
        crossentropy = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(y, tf.float32),
                                                    logits=yhat_logits))
        yhat = tf.sigmoid(yhat_logits)
        dpe = U.demographic_parity_discrimination(yhat, a)
        tppe, fppe = U.equalized_odds_discrimination(yhat, a, y)
        dpe, tppe, fppe = U.zero_nans(dpe, tppe, fppe)
        overall_loss = crossentropy +\
                self.hparams["dpe_scalar"]*dpe +\
                self.hparams["tppe_scalar"]*tppe +\
                self.hparams["fppe_scalar"]*fppe
        return overall_loss

class AdversariallyCensoredNN(SimpleNN):
    def default_hparams(self):
        hparams = super().default_hparams()
        hparams.update({
            "adversary_layers": [],
            "adv_loss_scalar": 1.0,
        })
        return hparams

    def build_training_pipeline(self, x, y, s):
        """Returns loss metrics, metric_names, TODO: more
        """
        yhat, yhat_logits, embedding = self.build_network(x)
        shat, shat_logits = self.build_adversary(x, y)
        loss = self.build_loss(x, y, s, yhat_logits)
        metrics, metric_names = self.build_metrics(x, y, s, yhat_logits)
        metrics = [loss] + metrics
        metric_names = ['overall_loss'] + metric_names
        opt_ops = [tf.train.AdamOptimizer().minimize(loss)]
        return opt_ops, yhat, embedding, metrics, metric_names
