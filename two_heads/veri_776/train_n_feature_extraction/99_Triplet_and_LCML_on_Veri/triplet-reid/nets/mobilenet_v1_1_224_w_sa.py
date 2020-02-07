import tensorflow as tf

from nets.mobilenet_v1_w_sa import mobilenet_v1_w_sa
from tensorflow.contrib import slim


def endpoints(image, is_training):
    if image.get_shape().ndims != 4:
        raise ValueError('Input must be of size [batch, height, width, 3]')

    image = tf.divide(image, 255.0)

    with tf.contrib.slim.arg_scope(mobilenet_v1_w_sa_arg_scope(batch_norm_decay=0.9, weight_decay=0.0)):
        _, endpoints = mobilenet_v1_w_sa(image, num_classes=1001, is_training=is_training)

    endpoints['model_output'] = endpoints['global_pool'] = tf.reduce_mean(
        endpoints['Conv2d_13_pointwise'], [1, 2], name='global_pool', keep_dims=False)

    return endpoints, 'MobilenetV1'


# This is copied and modified from mobilenet_v1_w_sa.py.
def mobilenet_v1_w_sa_arg_scope(
    is_training=True,
    weight_decay=0.00004,
    stddev=0.09,
    regularize_depthwise=False,
    batch_norm_decay=0.9997,
    batch_norm_epsilon=0.001,
    batch_norm_updates_collections=tf.GraphKeys.UPDATE_OPS,
    normalizer_fn=slim.batch_norm):
  """Defines the default MobilenetV1 arg scope.

  Args:
    is_training: Whether or not we're training the model. If this is set to
      None, the parameter is not added to the batch_norm arg_scope.
    weight_decay: The weight decay to use for regularizing the model.
    stddev: The standard deviation of the trunctated normal weight initializer.
    regularize_depthwise: Whether or not apply regularization on depthwise.
    batch_norm_decay: Decay for batch norm moving average.
    batch_norm_epsilon: Small float added to variance to avoid dividing by zero
      in batch norm.
    batch_norm_updates_collections: Collection for the update ops for
      batch norm.
    normalizer_fn: Normalization function to apply after convolution.

  Returns:
    An `arg_scope` to use for the mobilenet v1 model.
  """
  batch_norm_params = {
      'center': True,
      'scale': True,
      'decay': batch_norm_decay,
      'epsilon': batch_norm_epsilon,
      'updates_collections': batch_norm_updates_collections,
  }
  if is_training is not None:
    batch_norm_params['is_training'] = is_training

  # Set weight_decay for weights in Conv and DepthSepConv layers.
  weights_init = tf.truncated_normal_initializer(stddev=stddev)
  regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
  if regularize_depthwise:
    depthwise_regularizer = regularizer
  else:
    depthwise_regularizer = None
  with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                      weights_initializer=weights_init,
                      activation_fn=tf.nn.relu6, normalizer_fn=normalizer_fn):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      with slim.arg_scope([slim.conv2d], weights_regularizer=regularizer):
        with slim.arg_scope([slim.separable_conv2d],
                            weights_regularizer=depthwise_regularizer) as sc:
          return sc
