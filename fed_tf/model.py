import tensorflow as tf
import numpy as np
from tensorflow.compat.v1.train import AdamOptimizer
import os

class ConvNet(tf.Module):
    def __init__(self, graph, input_size, num_classes, num_channels, learning_rate, args):
        """
            Construct the AlexNet model.
            graph: The tf computation graph (`tf.Graph`)
            input_size: The size of input image (`int`)
            num_classes: The number of output classes (`int`)
            num_channels: The number of input channels (`int`)
            learning_rate: Learning rate for optimizer (`float`)
            args: Training arguments (Namespace)
        """
        self.config = tf.ConfigProto()
        if args.cuda:
            os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
            self.config.gpu_options.allow_growth = True

        with graph.as_default():
            self.x = tf.placeholder(tf.float32, [None, input_size, input_size, num_channels], name='x')
            self.y = tf.placeholder(tf.float32, [None, num_classes], name='y')

            # 1st Layer: Conv (w ReLu) -> Lrn -> Pool
            conv1 = conv(self.x, 5, 5, 32, 1, 1, name='conv1')
            pool1 = max_pool(conv1, 2, 2, 2, 2, padding='VALID', name='pool1')

            # 2nd Layer: Conv (w ReLu)  -> Lrn -> Pool
            conv2 = conv(pool1, 5, 5, 64, 1, 1, name='conv2')
            pool2 = max_pool(conv2, 2, 2, 2, 2, padding='VALID', name='pool2')

            if input_size == 32:
                # 3rd Layer: Flatten -> FC (w ReLu) -> Dropout
                # flattened = tf.reshape(pool5, [-1, 4*4*64])
                # fc1 = fc(flattened, 5*5*64, 4096, name='fc6')
                flattened = tf.reshape(pool2, [-1, 8*8*64])
                fc1 = fc_layer(flattened, 8*8*64, 2048, name='fc1')
                # 4th Layer: FC and return unscaled activations
                logits = fc_layer(fc1, 2048, num_classes, relu=False, name='fc8')
            else:
                raise ValueError("Invalid dataset!")

            # loss and optimizer
            self.loss_op = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
                                                            labels=self.y))
            optimizer = AdamOptimizer(
                learning_rate=learning_rate)
            self.train_op = optimizer.minimize(self.loss_op)

            # Evaluate model
            self.y_hat = tf.nn.softmax(logits)


def conv(x, filter_height, filter_width, num_filters,
            stride_y, stride_x, name, padding='SAME', groups=1):
    """Create a convolution layer.
    Adapted from: https://github.com/ethereon/caffe-tensorflow
    """
    # Get number of input channels
    input_channels = int(x.get_shape()[-1])

    # Create lambda function for the convolution
    convolve = lambda i, k: tf.nn.conv2d(
        i, k, strides=[1, stride_y, stride_x, 1], padding=padding)

    with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases of the conv layer
        weights = tf.get_variable('weights',
                                    shape=[
                                        filter_height, filter_width,
                                        input_channels / groups, num_filters
                                    ])
        biases = tf.get_variable('biases', shape=[num_filters])

    if groups == 1:
        conv = convolve(x, weights)

    # In the cases of multiple groups, split inputs & weights and
    else:
        # Split input and weights and convolve them separately
        input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
        weight_groups = tf.split(axis=3,
                                    num_or_size_splits=groups,
                                    value=weights)
        output_groups = [
            convolve(i, k) for i, k in zip(input_groups, weight_groups)
        ]

        # Concat the convolved output together again
        conv = tf.concat(axis=3, values=output_groups)

    # Add biases
    bias = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

    # Apply relu function
    relu = tf.nn.relu(bias, name=scope.name)

    return relu


def fc_layer(x, input_size, output_size, name, relu=True, k=20):
    """Create a fully connected layer."""

    with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases.
        W = tf.get_variable('weights', shape=[input_size, output_size])
        b = tf.get_variable('biases', shape=[output_size])
        # Matrix multiply weights and inputs and add biases.
        z = tf.nn.bias_add(tf.matmul(x, W), b, name=scope.name)

    if relu:
        # Apply ReLu non linearity.
        a = tf.nn.relu(z)
        return a

    else:
        return z


def max_pool(x, \
                filter_height, filter_width,
                stride_y, stride_x,
                name, padding='SAME'):
    """Create a max pooling layer."""
    return tf.nn.max_pool2d(x,
        ksize=[1, filter_height, filter_width, 1],
        strides=[1, stride_y, stride_x, 1],
        padding=padding,
        name=name)


def lrn(x, radius, alpha, beta, name, bias=1.0):
    """Create a local response normalization layer."""
    return tf.nn.local_response_normalization(x,
        depth_radius=radius,
        alpha=alpha,
        beta=beta,
        bias=bias,
        name=name)


def dropout(x, rate):
    """Create a dropout layer."""
    return tf.nn.dropout(x, rate=rate)
