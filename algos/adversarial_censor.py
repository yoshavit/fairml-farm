import tensorflow as tf
from algos.baselines import SimpleNN
import utils.tf_utils as U

class AdversariallyCensoredNN(SimpleNN):
    name = "adversariallycensorednn"
    def default_hparams(self):
        hparams = super().default_hparams()
        hparams.update({
            "adv_learning_rate": 3e-4,
            "adv_layersizes": [],
            "adv_loss_scalar": 1.0,
            "adv_sees_label": False,
        })
        return hparams

    def build_training_pipeline(self, x, y, a):
        """Returns loss metrics, metric_names, TODO: more
        """
        yhat, yhat_logits, z = self.build_network(x)
        ahat, ahat_logits = self.build_adversary(z, y)
        loss = self.build_loss(y, a, yhat_logits, ahat_logits)
        metrics, metric_names = self.build_metrics(y, a, yhat_logits,
                                                   ahat_logits)
        metrics = [loss] + metrics
        metric_names = ['overall_loss'] + metric_names
        nrml_params = [v for v in tf.global_variables() if "adversary" not in v.name]
        adv_params = [v for v in tf.global_variables() if "adversary" in v.name]
        nrml_opt_op = self.hparams["optimizer"](self.hparams["learning_rate"]
                                               ).minimize(loss, var_list=nrml_params)
        adv_opt_op = self.hparams["optimizer"](self.hparams["adv_learning_rate"]
                                              ).minimize(-1*loss, var_list=adv_params)
        opt_ops = [nrml_opt_op, adv_opt_op]
        return opt_ops, yhat, z, metrics, metric_names

    def build_adversary(self, z, y, reuse=False):
        r = z
        with tf.variable_scope("adversary", reuse=reuse):
            if self.hparams["adv_sees_label"]:
                r = tf.concat([r, tf.expand_dims(tf.cast(y, tf.float32), axis=1)],
                              axis=1)
            for i, l in enumerate(self.hparams["adv_layersizes"]):
                r = U.lrelu(U._linear(r, l, "fc{}".format(i)))
            ahat_logits = tf.squeeze(U._linear(r, 1, "output_logits"), axis=1)
            ahat = tf.sigmoid(ahat_logits)
        return ahat, ahat_logits

    def build_loss(self, y, a, yhat_logits, ahat_logits):
        primary_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(y, tf.float32),
                                                    logits=yhat_logits))
        adv_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(a, tf.float32),
                                                    logits=ahat_logits))
        overall_loss = primary_loss - self.hparams["adv_loss_scalar"]*adv_loss
        return overall_loss

    def build_metrics(self, y, a, yhat_logits, ahat_logits):
        metrics, metric_names = super().build_metrics(y, a, yhat_logits)
        adv_crossentropy = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(a, tf.float32),
                                                    logits=ahat_logits))
        adv_accuracy = tf.reduce_mean(tf.cast(
            tf.equal(tf.greater(ahat_logits, 0.0), a),
            tf.float32))
        metrics += [adv_crossentropy,
                    adv_accuracy]
        metric_names += ["adv_crossentropy",
                         "adv_accuracy"]
        return metrics, metric_names
