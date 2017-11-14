from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf
import time

import utils.tf_utils as U

class BaseClassifier(ABC):
    """Abstract class for binary classifiers.
    """
    def __init__(self, inputsize):
        self.inputsize = inputsize
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
    def __init__(self, inputsize, sess=None, layersizes=[100, 100],
                 batchsize=32, with_dropout=True):
        super().__init__(inputsize)
        if sess is None:
            self.sess = tf.Session()
        self.batchsize=batchsize
        # ========== Build training pipeline ===================
        self.dataset = tf.data.Dataset()
        self.dataset_X = tf.placeholder(dtype=tf.float32, shape=[None, self.inputsize],
                                      name="dataset_X")
        self.dataset_Y = tf.placeholder(tf.bool, shape=[None], name="dataset_Y")
        self.dataset_S = tf.placeholder(tf.bool, shape=[None], name="dataset_S")
        self.keep_prob = tf.placeholder(dtype=tf.float32)
        dataset = tf.data.Dataset.from_tensor_slices(
            (self.dataset_X, self.dataset_Y, self.dataset_S
            )).shuffle(100, reshuffle_each_iteration=True).batch(self.batchsize)
        self.training_iterator = dataset.make_initializable_iterator()
        X, Y, S = self.training_iterator.get_next()
        self.training_Yhat, Yhat_logits, _ = self.build_network(X, layersizes,
                                                                with_dropout,
                                                                reuse=False)
        logloss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(Y, tf.float32),
                                                    logits=Yhat_logits))
        accuracy = tf.reduce_mean(tf.cast(
            tf.equal(tf.cast(tf.round(self.training_Yhat), tf.bool), Y),
            tf.float32))
        dpe = U.demographic_parity_discrimination(self.training_Yhat, S)
        tppe, fppe = U.equalized_odds_discrimination(self.training_Yhat, S, Y)
        metrics = [logloss, accuracy, dpe, tppe, fppe]
        metric_names = ["logloss", "accuracy",
                        "demographic_parity_error",
                        "true_positive_parity_error",
                        "false_positive_parity_error"]
        # Make copies of the metrics, and store different avgs for the train
        # and validation copies
        train_metrics = [tf.identity(m) for m in metrics]
        ema = tf.train.ExponentialMovingAverage(0.99)
        opt_op = tf.train.AdamOptimizer().minimize(logloss)
        self.global_step = tf.Variable(0, trainable=False)
        inc_global_step = tf.assign_add(self.global_step, batchsize)
        with tf.control_dependencies([opt_op, inc_global_step]):
            self.train_op = U.ema_apply_wo_nans(ema, train_metrics)
        self.train_summaries = tf.summary.merge([
            tf.summary.scalar(metric_name, ema.average(metric), family="train")
            for metric_name, metric in zip(metric_names, train_metrics)])

        val_metrics =  [tf.identity(m) for m in metrics]
        # Below is a workaround for computing avg metrics over numerous
        # validation samples - does not precisely calculate validation error,
        # but rather works over a certain window, with decay at beginning of
        # that window.
        self.validation_op = U.ema_apply_wo_nans(ema, val_metrics)
        # Reinitialize the moving average (i.e. return to 0.0) for each of the
        # validation metrics
        self.restart_validation_op = tf.variables_initializer([
            ema.average(metric) for metric in val_metrics])
        self.val_summaries = tf.summary.merge([
            tf.summary.scalar(metric_name, ema.average(metric), family="val")
            for metric_name, metric in zip(metric_names, val_metrics)])

        # ======= prediction pipeline ================
        self.prediction_X = tf.placeholder(tf.float32, shape=[None, inputsize],
                                      name="prediction_X")
        self.prediction_iterator = tf.data.Dataset.from_tensor_slices(
            self.prediction_X).batch(batchsize).make_initializable_iterator()
        X = self.prediction_iterator.get_next()
        self.prediction_Yhat, _, self.prediction_embedding = self.build_network(
            X, layersizes, with_dropout, reuse=True)
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

    def build_network(self, X, layersizes, with_dropout, reuse=False):
        z = X
        with tf.variable_scope("classifier", reuse=reuse):
            for i, l in enumerate(layersizes):
                z = U.lrelu(U._linear(z, l, "fc{}".format(i)))
            embedding = z
            if with_dropout:
                z = tf.nn.dropout(z, self.keep_prob)
            yhat_logits = tf.squeeze(U._linear(z, 1, "output_logits"), axis=1)
            yhat = tf.sigmoid(yhat_logits)
            return yhat, yhat_logits, embedding

    def train(self, training_dataset, logdir, epochs=10, keep_prob=0.5,
              validation_dataset=None, validation_batches=None):
        # if validation_batches is None, runs all batches
        sw = tf.summary.FileWriter(logdir)
        for epoch in range(epochs):
            self.sess.run(self.training_iterator.initializer,
                          feed_dict={
                              self.dataset_X: training_dataset["data"],
                              self.dataset_Y: training_dataset["label"],
                              self.dataset_S: training_dataset["protected"]})
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
                                  self.dataset_S: validation_dataset["protected"]})
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
        self.saver.restore(self.sess, filename)

    def save_model(self, filepath):
        filename = self.saver.save(self.sess, filepath)
        return filename

    def predict(self, X):
        """Compute the classifier output
        Args:
            X - n x d array of datapoints
        Returns:
            Z - n x r array of embeddings of datapoints
        """
        yhatbatched = []
        self.sess.run(self.prediction_iterator.initializer,
                      feed_dict={self.prediction_X: X})
        while True:
            try:
                yhatbatch = self.sess.run(self.prediction_Yhat,
                                          feed_dict={self.keep_prob:1.0})
                yhatbatched.append(yhatbatch)
            except tf.errors.OutOfRangeError:
                break
        return np.concatenate(yhatbatched, axis=0)

    def compute_embedding(self, X):
        """Compute the low-dimensional embedding (learned by the classifier)
        of a set of datapoints. In this case, that is the last layer of the
        network before the output.
        Args:
            X - n x d array of datapoints
        Returns:
            Z - n x r array of embeddings of datapoints
        """
        yhatbatched = []
        self.sess.run(self.prediction_iterator.initializer,
                      feed_dict={self.prediction_X: X})
        while True:
            try:
                yhatbatch = self.sess.run(self.prediction_embedding,
                                          feed_dict={self.keep_prob:1.0})
                yhatbatched.append(yhatbatch)
            except tf.errors.OutOfRangeError:
                break
        return np.concatenate(yhatbatched, axis=0)

