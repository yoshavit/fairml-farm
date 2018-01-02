import tensorflow as tf
def _weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.get_variable("W", initializer=initial)
def _bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.get_variable("b", initializer=initial)
def _linear(x, output_size, name):
    """Construct a fully-connected linear layer.
    """
    with tf.variable_scope(name):
        W = _weight_variable([x.get_shape().as_list()[1], output_size])
        b = _bias_variable([output_size])
        output = tf.matmul(x, W) + b
    return output
def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x, name=name)

def zero_nans(*tensors):
    return [tf.where(tf.is_nan(t), 0.0, t) for t in tensors]

def ema_apply_wo_nans(ema, tensors):
    """Updates ExponentialMovingAverage (ema) with current values of tensors
    (similar to ema.apply(tensors) ), while ignoring tensors with NaN values.
    """
    return [tf.cond(tf.is_nan(t),
                    true_fn = lambda: tf.no_op(),
                    false_fn = lambda: ema.apply([t]))
            for t in tensors]

# ====== Tensorboard Ops ====================
from multiprocessing import Process
import subprocess

def launch_tensorboard(logdir, tensorboard_path=None):
    if tensorboard_path is None:
        import platform
        assert platform.node != 'Yonadavs-MacBook-Air.local', "New users must specify path to tensorboard"
        tensorboard_path = '/Users/yonadav/anaconda/envs/tensorflow3.5/bin/tensorboard'
    def _call_tensorboard():
        subprocess.call("{} --logdir={}".format(tensorboard_path, logdir),
                        shell=True)
    tensorboard_process = Process(target=_call_tensorboard)
    tensorboard_process.start()
    return tensorboard_process

# ======= Bias-specific Ops =========================

def demographic_parity_discrimination(yhat, a):
    """Computes the difference in the mean prediction between protected
    classes. (https://www.cs.toronto.edu/~toni/Papers/icml-final.pdf Eq. 14)
    Args:
        yhat - n x 1 tensor of predictions
        a - n x 1 tf.bool tensor marking whether the individual is from the
            protected class
    Returns:
        A scalar tensor of the difference in mean prediction between classes.
    """
    yhat_a0, yhat_a1 = tf.dynamic_partition(yhat, tf.cast(a, tf.int32), 2)
    disc = tf.abs(tf.reduce_mean(yhat_a0) - tf.reduce_mean(yhat_a1))
    return disc

def equalized_odds_discrimination(yhat, a, y):
    """Computes the difference in the mean prediction between protected classes,
    conditioned on the true outcome. Equivalent to the deviation from
    'Equalized Odds' defined in https://arxiv.org/pdf/1610.02413.pdf
    Args:
        yhat - n x 1 tensor of predictions
        a - n x 1 tf.bool tensor marking whether the individual is from the
            protected class
        y - n x 1 tf.bool tensor marking the individual's true outcome
    Returns:
        false_negative_parity_error - a scalar tensor of the mean deviation for
            negative outcomes
        false_positive_parity_error - a scalar tensor of the mean deviation for
            negative outcomes
    """
    partitions = tf.cast(y, tf.int32)*2 + tf.cast(a, tf.int32)
    yhat_y0_a0, yhat_y0_a1, yhat_y1_a0, yhat_y1_a1 = tf.dynamic_partition(
        yhat, partitions, 4)
    false_negative_parity_error = tf.abs(tf.reduce_mean(yhat_y0_a0) -
                                        tf.reduce_mean(yhat_y0_a1))
    false_positive_parity_error = tf.abs(tf.reduce_mean(yhat_y1_a0) -
                                        tf.reduce_mean(yhat_y1_a1))
    return false_negative_parity_error, false_positive_parity_error

def crossentropy(yhat, y):
    return -(tf.log(yhat)*y + tf.log(1-yhat)*(1-y))

def calibration_parity_loss(yhat, a, y, yhat_logits=None):
    """Computes the abs difference in the mean loss between protected
    classes. (https://www.cs.toronto.edu/~toni/Papers/icml-final.pdf Eq. 14)
    Args:
        yhat - n x 1 tensor of predictions
        a - n x 1 tf.bool tensor marking whether the individual is from the
            protected class
        yhat_logits - optional, n x 1 tensor of prediction logits used to
            compute the crossentropy more efficiently if provided
    Returns:
        A scalar tensor of the difference in mean loss between classes.
    """
    # TODO: implement check if there are no members of one of the classes
    a = tf.cast(a, tf.int32)
    y_a0, y_a1 = tf.dynamic_partition(tf.cast(y, tf.float32),
                                      a, 2)
    if yhat_logits is not None:
        yhat_logits_a0, yhat_logits_a1 = tf.dynamic_partition(yhat_logits, a, 2)
        disc = tf.abs(
            tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=y_a0,
                                                        logits=yhat_logits_a0))
            - tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=y_a1,
                                                        logits=yhat_logits_a1)))
    else:
        yhat_a0, yhat_a1 = tf.dynamic_partition(yhat, a, 2)
        disc = tf.abs(
            tf.reduce_mean(crossentropy(yhat_a0, y_a0))
            - tf.reduce_mean(crossentropy(yhat_a1, y_a1))
        )
    return disc
