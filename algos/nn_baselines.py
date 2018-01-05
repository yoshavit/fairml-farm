import numpy as np
import tensorflow as tf
import time
import os
from algos.base import BaseClassifier
import utils.tf_utils as U

class SimpleNN(BaseClassifier):
    name = "simplenn"
    def __init__(self, sess=None):
        self.sess = sess if sess is not None else tf.Session()
        self.built = False

    def check_built(self):
        assert self.built, "Classifier not yet built; Call obj.build(...) before using"

    def default_hparams(self):
        return {
            "inputsize": 105, # for Adult dataset
            "learning_rate": 3e-4,
            "layersizes": [100,100],
            "batchsize": 32,
            "with_dropout": False,
            "l2_weight_penalty": 0.0,
            "optimizer": tf.train.AdamOptimizer,
        }

    def build(self, hparams={}):
        self.hparams = self.default_hparams()
        self.hparams.update(hparams)
        # ========== Build training pipeline ===================
        self.dataset = tf.data.Dataset()
        self.dataset_X = tf.placeholder(dtype=tf.float32, shape=[None,
                                                                 self.hparams["inputsize"]],
                                      name="dataset_X")
        self.dataset_Y = tf.placeholder(tf.bool, shape=[None], name="dataset_Y")
        self.dataset_A = tf.placeholder(tf.bool, shape=[None], name="dataset_A")
        self.keep_prob = tf.placeholder(dtype=tf.float32)
        dataset = tf.data.Dataset.from_tensor_slices(
            (self.dataset_X, self.dataset_Y, self.dataset_A
            )).shuffle(100, reshuffle_each_iteration=True).batch(
                self.hparams["batchsize"])
        self.dataset_iterator = dataset.make_initializable_iterator()
        x, y, a = self.dataset_iterator.get_next()
        opt_ops, yhat, embedding, metrics, metric_names =\
                self.build_training_pipeline(x, y, a)
        # Make copies of the metrics, and store different avgs for the train
        # and validation copies
        train_metrics = [tf.identity(m) for m in metrics]
        train_ema = tf.train.ExponentialMovingAverage(0.99)
        self.global_step = tf.Variable(0, trainable=False)
        inc_global_step = tf.assign_add(self.global_step,
                                        self.hparams["batchsize"])
        with tf.control_dependencies(opt_ops + [inc_global_step]):
            self.train_op = U.ema_apply_wo_nans(train_ema, train_metrics)
        self.train_summaries = tf.summary.merge([
            tf.summary.scalar(metric_name, train_ema.average(metric), family="train")
            for metric_name, metric in zip(metric_names, train_metrics)])

        val_metrics =  [tf.identity(m) for m in metrics]
        val_ema = tf.train.ExponentialMovingAverage(0.99)
        # We also compute the moving-average for validation error
        self.validation_op = U.ema_apply_wo_nans(val_ema, val_metrics)
        # Reinitialize the moving average (i.e. return to 0.0) for each epoch
        # of the validation metrics
        self.restart_validation_op = tf.variables_initializer([
            val_ema.average(metric) for metric in val_metrics])
        self.val_summaries = tf.summary.merge([
            tf.summary.scalar(metric_name, val_ema.average(metric), family="val")
            for metric_name, metric in zip(metric_names, val_metrics)])
        # alias for computing metrics w/o tensorboard
        self.metric_dict = {"val_" + metric_name:
                            val_ema.average(metric_tensor) for
                            metric_name, metric_tensor in
                            zip(metric_names, val_metrics)}
        self.metric_dict.update({"train_" + metric_name:
                                 train_ema.average(metric_tensor)
                                 for metric_name, metric_tensor in
                                 zip(metric_names, train_metrics)})


        # ======= prediction pipeline ================
        self.prediction_x = tf.placeholder(tf.float32, shape=[None,
                                                              self.hparams["inputsize"]],
                                      name="prediction_x")
        self.prediction_iterator = tf.data.Dataset.from_tensor_slices(
            self.prediction_x).batch(self.hparams["batchsize"]).make_initializable_iterator()
        x = self.prediction_iterator.get_next()
        self.prediction_yhat, _, self.prediction_embedding = self.build_network(
            x, reuse=True)
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        self.sw = None
        self.built = True

    def build_training_pipeline(self, x, y, a):
        """Returns loss metrics, metric_names, TODO: more
        """
        yhat, yhat_logits, embedding = self.build_network(x)
        loss = self.build_loss(y, a, yhat_logits)
        metrics, metric_names = self.build_metrics(y, a, yhat_logits)
        metrics = [loss] + metrics
        metric_names = ['overall_loss'] + metric_names
        opt_ops = [
            self.hparams["optimizer"](self.hparams["learning_rate"]).minimize(loss)
        ]
        return opt_ops, yhat, embedding, metrics, metric_names

    def build_loss(self, y, a, yhat_logits):
        crossentropy = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(y, tf.float32),
                                                    logits=yhat_logits))
        l2_penalty = tf.reduce_mean(tf.concat(
            [tf.reshape(var, [-1]) for var in
             tf.trainable_variables(scope="classifier")], axis=0))
        return crossentropy + self.hparams["l2_weight_penalty"]*l2_penalty

    def build_metrics(self, y, a, yhat_logits):
        yhat = tf.sigmoid(yhat_logits)
        crossentropy = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(y, tf.float32),
                                                    logits=yhat_logits))
        accuracy = tf.reduce_mean(tf.cast(
            tf.equal(tf.greater(yhat_logits, 0.0), y),
            tf.float32))
        dpe = U.demographic_parity_discrimination(yhat, a)
        fnpe, fppe = U.equalized_odds_discrimination(yhat, a, y)
        cpe = U.calibration_parity_loss(yhat, a, y, yhat_logits)
        metrics = [crossentropy, accuracy, dpe, fnpe, fppe, cpe]
        # order of the first two metrics is important for print-logging
        metric_names = ["crossentropy",
                        "accuracy",
                        "demographic_parity_error",
                        "false_negative_parity_error",
                        "false_positive_parity_error",
                        "calibration_parity_error"]
        return metrics, metric_names

    def build_network(self, x, reuse=False):
        r = x
        with tf.variable_scope("classifier", reuse=reuse):
            for i, l in enumerate(self.hparams["layersizes"]):
                r = U.lrelu(U._linear(r, l, "fc{}".format(i)))
            embedding = r
            if self.hparams["with_dropout"]:
                r = tf.nn.dropout(r, self.keep_prob)
            yhat_logits = tf.squeeze(U._linear(r, 1, "fc_final"), axis=1)
            yhat = tf.sigmoid(yhat_logits)
            return yhat, yhat_logits, embedding

    def train(self, training_dataset, logdir=None, epochs=10, keep_prob=0.5,
              validation_dataset=None, validation_batches=None):
        self.check_built()
        logging = logdir is not None
        if logging:
            if self.sw is None:
                self.sw = tf.summary.FileWriter(logdir)
            self.sw.reopen()

        for epoch in range(epochs):
            self.sess.run(self.dataset_iterator.initializer,
                          feed_dict={
                              self.dataset_X: training_dataset["data"],
                              self.dataset_Y: training_dataset["label"],
                              self.dataset_A: training_dataset["protected"]})
            print("Epoch {:3}...".format(epoch), end=" ")
            starttime = time.clock()
            steps = 0
            while True:
                steps += 1
                try:
                    self.sess.run(self.train_op, feed_dict={self.keep_prob: keep_prob})
                    if steps % 100 == 0 and logging:
                        self.sw.add_summary(self.sess.run(self.train_summaries),
                                            self.sess.run(self.global_step))
                except tf.errors.OutOfRangeError:
                    break
            if logging:
                self.sw.add_summary(self.sess.run(self.train_summaries),
                                    self.sess.run(self.global_step))
                self.sw.flush()
            if validation_dataset is not None:
                self.validate(validation_dataset, logdir, validation_batches,
                              close_sw_on_exit=False)
            endmsg = "complete after {:0.2f} seconds. ".format(time.clock() -
                                                               starttime)
            metric_dict = self.sess.run(self.metric_dict)
            # specify the metrics to be printed on the command line
            printed_metrics = ["val_crossentropy",
                               "train_accuracy",
                               "val_calibration_parity_error"]
            endmsg += ", ".join(["{} = {:0.2e}".format(pm, metric_dict[pm])
                                 for pm in printed_metrics])
            print(endmsg)
        if logging: self.sw.close()

    def validate(self, validation_dataset, logdir=None,
                 validation_batches=None, close_sw_on_exit=True):
        logging = logdir is not None
        if logging:
            if self.sw is None:
                self.sw = tf.summary.FileWriter(logdir)
            self.sw.reopen()
        self.sess.run(self.dataset_iterator.initializer,
                      feed_dict={
                          self.dataset_X: validation_dataset["data"],
                          self.dataset_Y: validation_dataset["label"],
                          self.dataset_A: validation_dataset["protected"]})
        steps = 0
        if validation_batches is None: validation_batches = float("inf")
        # self.sess.run(self.restart_validation_op)
        while steps < validation_batches:
            steps +=1
            try:
                self.sess.run(self.validation_op, feed_dict={self.keep_prob: 1.0})
            except tf.errors.OutOfRangeError:
                break
        if logging:
            self.sw.add_summary(self.sess.run(self.val_summaries),
                                self.sess.run(self.global_step))
            self.sw.flush()
            if close_sw_on_exit:
                self.sw.close()

    def load_model(self, filedir):
        self.check_built()
        filepath = os.path.join(filedir, "model.ckpt")
        self.saver.restore(self.sess, filepath)

    def save_model(self, filedir):
        self.check_built()
        filepath = self.saver.save(self.sess, filedir)
        return filepath

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

    def extract_final_layer_weights(self):
        if len(self.hparams["layersizes"]) > 0:
            raise ValueError("Final layer weights are not explanatory when"
                               "classifier is a multi-layer NN")
        # TODO: find a cleanerway to grab a variable by name
        weights = [var for var in tf.global_variables()
                   if "classifier/fc_final/W:0" in var.name][0]
        bias = [var for var in tf.global_variables()
                if "classifier/fc_final/b:0" in var.name][0]
        return weights, bias

class ParityNN(SimpleNN):
    name = "paritynn"
    def default_hparams(self):
        hparams = super().default_hparams()
        hparams.update({
            "dpe_scalar": 1.0,
            "fnpe_scalar": 0.0,
            "fppe_scalar": 0.0,
            "cpe_scalar": 0.0,
        })
        return hparams

    def build_loss(self, y, a, yhat_logits):
        crossentropy = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(y, tf.float32),
                                                    logits=yhat_logits))
        yhat = tf.sigmoid(yhat_logits)
        dpe = U.demographic_parity_discrimination(yhat, a)
        fnpe, fppe = U.equalized_odds_discrimination(yhat, a, y)
        dpe, fnpe, fppe = U.zero_nans(dpe, fnpe, fppe)
        cpe = U.calibration_parity_loss(yhat, a, y, yhat_logits)
        l2_penalty = tf.reduce_mean(tf.concat(
            [tf.reshape(var, [-1]) for var in
             tf.trainable_variables(scope="classifier")], axis=0))
        overall_loss = crossentropy +\
                self.hparams["dpe_scalar"]*dpe +\
                self.hparams["fnpe_scalar"]*fnpe +\
                self.hparams["fppe_scalar"]*fppe +\
                self.hparams["cpe_scalar"]*cpe +\
                self.hparams["l2_weight_penalty"]*l2_penalty
        return overall_loss
