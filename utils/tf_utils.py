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

# ======= Bias-specific Ops =========================

def demographic_parity_discrimination(Yhat, S):
    """Computes the difference in the mean prediction between protected
    classes. (https://www.cs.toronto.edu/~toni/Papers/icml-final.pdf Eq. 14)
    Args:
        Yhat - n x 1 tensor of predictions
        S - n x 1 tf.bool tensor marking whether the individual is from the
            protected class
    Returns:
        A scalar tensor of the difference in mean prediction between classes.
    """
    Yhat_s0, Yhat_s1 = tf.dynamic_partition(Yhat, tf.cast(S, tf.int32), 2)
    disc = tf.abs(tf.reduce_mean(Yhat_s0) - tf.reduce_mean(Yhat_s1))
    return disc

def equalized_odds_discrimination(Yhat, S, Y):
    """Computes the difference in the mean prediction between protected classes,
    conditioned on the true outcome. Equivalent to the deviation from
    'Equalized Odds' defined in https://arxiv.org/pdf/1610.02413.pdf
    Args:
        Yhat - n x 1 tensor of predictions
        S - n x 1 tf.bool tensor marking whether the individual is from the
            protected class
        Y - n x 1 tf.bool tensor marking the individual's true outcome
    Returns:
        true_positive_parity_error - a scalar tensor of the mean deviation for
            positive outcomes
        false_positive_parity_error - a scalar tensor of the mean deviation for
            negative outcomes
    """
    partitions = tf.cast(Y, tf.int32)*2 + tf.cast(S, tf.int32)
    Yhat_y0_s0, Yhat_y0_s1, Yhat_y1_s0, Yhat_y1_s1 = tf.dynamic_partition(
        Yhat, partitions, 4)
    true_positive_parity_error = tf.abs(tf.reduce_mean(Yhat_y0_s0) -
                                        tf.reduce_mean(Yhat_y0_s1))
    false_positive_parity_error = tf.abs(tf.reduce_mean(Yhat_y1_s0) -
                                        tf.reduce_mean(Yhat_y1_s1))
    return true_positive_parity_error, false_positive_parity_error



