import tensorflow as tf
from tensorflow.python.training import moving_averages
import os


def resnet20(graph, args, optimizer_fn, input_size, num_classes, in_channels):
    return ResNet(graph, args, optimizer_fn, [3, 3, 3], input_size, num_classes, in_channels)


def resnet32(graph, args, optimizer_fn, input_size, num_classes, in_channels):
    return ResNet(graph, args, optimizer_fn, [5, 5, 5], input_size, num_classes, in_channels)


def resnet44(graph, args, optimizer_fn, input_size, num_classes, in_channels):
    return ResNet(graph, args, optimizer_fn, [7, 7, 7], input_size, in_channels, num_classes)


def resnet56(graph, args, optimizer_fn, input_size, num_classes, in_channels):
    return ResNet(graph, args, optimizer_fn, [9, 9, 9], input_size, in_channels, num_classes)


def resnet110(graph, args, optimizer_fn, input_size, num_classes, in_channels):
    return ResNet(graph, args, optimizer_fn, [18, 18, 18], input_size, in_channels, num_classes)


def resnet1202(graph, args, optimizer_fn, input_size, num_classes, in_channels):
    return ResNet(graph, args, optimizer_fn, [200, 200, 200], input_size, in_channels, num_classes)


class ResNet(tf.Module):
    def __init__(self, graph, args, optimizer_fn, num_blocks, input_size=32, num_classes=10, in_channels=3):
        self.config = tf.compat.v1.ConfigProto()
        if args.cuda:
            os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
            self.config.gpu_options.allow_growth = True
        
        self.graph = graph
        self.optimizer = optimizer_fn()
        self.num_blocks = num_blocks
        self.input_size = input_size
        self.in_channels = in_channels 
        self.num_classes= num_classes
        self._build_model()

    def _build_model(self):
        with self.graph.as_default():
            self.x = tf.compat.v1.placeholder(tf.float32, [None, self.input_size, self.input_size, self.in_channels], name="x")
            self.y = tf.compat.v1.placeholder(tf.float32, [None, self.num_classes], name="y")
                
            self.in_planes = 16
            
            conv1 = conv_layer(self.x, 16, 3, 1, scope="conv1") # -> (batch, 32, 32, 16)
            bn1 = tf.nn.relu(batch_norm(conv1, scope="bn1"))

            block2 = self._layer(bn1, 16, self.num_blocks[0], init_stride=1, scope="layer2") # -> (batch, 32, 32, 16)
            block3 = self._layer(block2, 32, self.num_blocks[1], init_stride=2, scope="layer3") # -> (batch, 16, 16, 32)
            block4 = self._layer(block3, 64, self.num_blocks[2], init_stride=2, scope="layer4") # -> (batch, 8, 8, 64)

            avgpool5 = pool_layer(block4, block4.get_shape()[-1], block4.get_shape()[-1], name="avgpool5", pooling_Mode = 'avg_pool') # -> (batch, 1, 1, 64)
            spatialsqueeze = tf.squeeze(avgpool5, [1, 2], name="spatial_squeeze") # -> (batch, 64)
            self.logits = fc_layer(spatialsqueeze, self.num_classes, "fc6") # -> (batch, num_classes)
            
            with tf.compat.v1.variable_scope('loss'):
                losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y)
                self.loss_op = tf.reduce_mean(losses)

            with tf.compat.v1.variable_scope('optimizer'):
                self.train_op = self.optimizer.minimize(self.loss_op)

    def _layer(self, x, planes, num_blocks, init_stride=2, scope="block"):
        self.expansion = 1
        with tf.compat.v1.variable_scope(scope):
            out = self._block(x, self.in_planes, planes, stride=init_stride, scope="block1")
            self.in_planes = planes * self.expansion
            strides = [1]*(num_blocks-1)
            
            for i, stride in enumerate(strides):
                out = self._block(out, self.in_planes, planes, stride, scope=("block%s" % (i + 2)))
                self.in_planes = planes * self.expansion

            return out

    def _block(self, x, in_planes, planes, stride=1, scope="block"):
        with tf.compat.v1.variable_scope(scope):
            out = conv_layer(x, planes, kernel_size=3, stride=stride, scope="conv_1")
            out = batch_norm(out, scope="bn_1")
            out = tf.nn.relu(out)
            
            out = conv_layer(out, planes, kernel_size=3, stride=1, scope="conv_2")
            out = batch_norm(out, scope="bn_2")

            if stride != 1 or in_planes != planes: #inplaces !+ plances
                shortcut = conv_layer(x, self.expansion * planes, kernel_size=1, stride=stride, scope="conv_3")
                shortcut = batch_norm(shortcut, scope="bn_3")
            else:
                shortcut = x

            out += shortcut
            out = tf.nn.relu(out)

            return out
        

def variable_weight(name, shape, initializer, trainable=True):
    return tf.compat.v1.get_variable(name, shape=shape, dtype=tf.float32,
                           initializer=initializer, trainable=trainable)


def conv_layer(x, num_outputs, kernel_size, stride=1, scope="conv2d"):
    input_channels = x.get_shape()[-1]

    with tf.compat.v1.variable_scope(scope):
        kernel = variable_weight("kernel", [kernel_size, kernel_size, input_channels, num_outputs], 
            tf.contrib.layers.xavier_initializer_conv2d())
            
        return tf.nn.conv2d(x, kernel, strides=[1, stride, stride, 1], padding="SAME")


def fc_layer(x, num_outputs, scope="fc"):
    input_channels = x.get_shape()[-1]
    
    with tf.compat.v1.variable_scope(scope):
        W = variable_weight("weight", [input_channels, num_outputs], 
            tf.contrib.layers.xavier_initializer())
        b = variable_weight("bias", [num_outputs,], 
            tf.zeros_initializer())
        
        return tf.compat.v1.nn.xw_plus_b(x, W, b)


# batch norm layer
def batch_norm(x, decay=0.999, epsilon=1e-03, scope="scope"):
    x_shape = x.get_shape()
    input_channels = x_shape[-1]
    reduce_dims = list(range(len(x_shape) - 1))

    with tf.compat.v1.variable_scope(scope):
        beta = variable_weight("beta", [input_channels,], 
                                initializer=tf.zeros_initializer())
        gamma = variable_weight("gamma", [input_channels,], 
                                initializer=tf.ones_initializer())
        # for inference
        moving_mean = variable_weight("moving_mean", [input_channels,],
                                initializer=tf.zeros_initializer(), trainable=False)
        moving_variance = variable_weight("moving_variance", [input_channels], 
                                initializer=tf.ones_initializer(), trainable=False)

    mean, variance = tf.nn.moments(x, axes=reduce_dims)
    update_move_mean = moving_averages.assign_moving_average(moving_mean, mean, decay=decay)
    update_move_variance = moving_averages.assign_moving_average(moving_variance, variance, decay=decay)
    tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.UPDATE_OPS, update_move_mean)
    tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.UPDATE_OPS, update_move_variance)

    return tf.nn.batch_normalization(x, mean, variance, beta, gamma, epsilon)


def pool_layer(x, pool_size, pool_stride, name, padding='SAME', pooling_Mode='max_pool'):
    if pooling_Mode=='max_pool':
        return tf.nn.max_pool(x, [1, pool_size, pool_size, 1], [1, pool_stride, pool_stride, 1], padding = padding, name = name)

    if pooling_Mode=='avg_pool':
        return tf.nn.avg_pool2d(x, [1, pool_size, pool_size, 1], [1, pool_stride, pool_stride, 1], padding = padding, name = name)


