# Converted to TensorFlow .caffemodel
# with the DeepLab-ResNet configuration.
# The batch normalisation layer is provided by
# the slim library (https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim).

import tensorflow as tf

from kaffe.tensorflow import Network


class ThreeDNetwork(Network):
    def setup(self, is_training, num_classes):
        '''Network definition.

        Args:
          is_training: whether to update the running mean and variance of the batch normalisation layer.
                       If the batch size is small, it is better to keep the running mean and variance of
                       the-pretrained model frozen.
          num_classes: number of classes to predict (including background).
            '''

        with tf.device('/gpu:0'):
            (self.feed('data')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv1')
             .conv(7, 7, 96, 2, 2, biased=False, relu=False, name='conv1')
             .max_pool(3, 3, 2, 2, name='pool1')
             )

            (self.feed('pool1')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv1a_dense1')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv1a_dense1')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv1b_dense1')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv1b_dense1')
             )

            (self.feed('pool1',
                       'conv1b_dense1')
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv2a_dense1')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv2a_dense1')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv2b_dense1')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv2b_dense1')
             )

            (self.feed('pool1',
                       'conv1b_dense1',
                       'conv2b_dense1')
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv3a_dense1')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv3a_dense1')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv3b_dense1')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv3b_dense1')
             )

            (self.feed('pool1',
                       'conv1b_dense1',
                       'conv2b_dense1',
                       'conv3b_dense1')
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv4a_dense1')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv4a_dense1')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv4b_dense1')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv4b_dense1')
             )

            (self.feed('pool1',
                       'conv1b_dense1',
                       'conv2b_dense1',
                       'conv3b_dense1',
                       'conv4b_dense1'
                       )
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv5a_dense1')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv5a_dense1')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv5b_dense1')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv5b_dense1')
             )

            (self.feed('pool1',
                       'conv1b_dense1',
                       'conv2b_dense1',
                       'conv3b_dense1',
                       'conv4b_dense1',
                       'conv5b_dense1'
                       )
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv6a_dense1')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv6a_dense1')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv6b_dense1')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv6b_dense1')
             )

            (self.feed('pool1',
                       'conv1b_dense1',
                       'conv2b_dense1',
                       'conv3b_dense1',
                       'conv4b_dense1',
                       'conv5b_dense1',
                       'conv6b_dense1'
                       )
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv_transition1')
             .conv(1, 1, (96 + 48 * 6) / 2, 1, 1, biased=False, relu=False, name='conv_transition1')
             .avg_pool(2, 2, 2, 2, name='pool_transition1')
             )

            (self.feed('pool_transition1')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv1a_dense2')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv1a_dense2')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv1b_dense2')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv1b_dense2')
             )

            (self.feed('pool_transition1',
                       'conv1b_dense2')
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv2a_dense2')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv2a_dense2')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv2b_dense2')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv2b_dense2')
             )

            (self.feed('pool_transition1',
                       'conv1b_dense2',
                       'conv2b_dense2')
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv3a_dense2')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv3a_dense2')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv3b_dense2')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv3b_dense2')
             )

            (self.feed('pool_transition1',
                       'conv1b_dense2',
                       'conv2b_dense2',
                       'conv3b_dense2')
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv4a_dense2')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv4a_dense2')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv4b_dense2')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv4b_dense2')
             )

            (self.feed('pool_transition1',
                       'conv1b_dense2',
                       'conv2b_dense2',
                       'conv3b_dense2',
                       'conv4b_dense2'
                       )
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv5a_dense2')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv5a_dense2')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv5b_dense2')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv5b_dense2')
             )

            (self.feed('pool_transition1',
                       'conv1b_dense2',
                       'conv2b_dense2',
                       'conv3b_dense2',
                       'conv4b_dense2',
                       'conv5b_dense2'
                       )
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv6a_dense2')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv6a_dense2')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv6b_dense2')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv6b_dense2')
             )

            (self.feed('pool_transition1',
                       'conv1b_dense2',
                       'conv2b_dense2',
                       'conv3b_dense2',
                       'conv4b_dense2',
                       'conv5b_dense2',
                       'conv6b_dense2'
                       )
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv7a_dense2')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv7a_dense2')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv7b_dense2')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv7b_dense2')
             )

            (self.feed('pool_transition1',
                       'conv1b_dense2',
                       'conv2b_dense2',
                       'conv3b_dense2',
                       'conv4b_dense2',
                       'conv5b_dense2',
                       'conv6b_dense2',
                       'conv7b_dense2'
                       )
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv8a_dense2')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv8a_dense2')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv8b_dense2')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv8b_dense2')
             )

            (self.feed('pool_transition1',
                       'conv1b_dense2',
                       'conv2b_dense2',
                       'conv3b_dense2',
                       'conv4b_dense2',
                       'conv5b_dense2',
                       'conv6b_dense2',
                       'conv7b_dense2',
                       'conv8b_dense2'
                       )
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv9a_dense2')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv9a_dense2')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv9b_dense2')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv9b_dense2')
             )

            (self.feed('pool_transition1',
                       'conv1b_dense2',
                       'conv2b_dense2',
                       'conv3b_dense2',
                       'conv4b_dense2',
                       'conv5b_dense2',
                       'conv6b_dense2',
                       'conv7b_dense2',
                       'conv8b_dense2',
                       'conv9b_dense2'
                       )
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv10a_dense2')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv10a_dense2')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv10b_dense2')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv10b_dense2')
             )

            (self.feed('pool_transition1',
                       'conv1b_dense2',
                       'conv2b_dense2',
                       'conv3b_dense2',
                       'conv4b_dense2',
                       'conv5b_dense2',
                       'conv6b_dense2',
                       'conv7b_dense2',
                       'conv8b_dense2',
                       'conv9b_dense2',
                       'conv10b_dense2'
                       )
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv11a_dense2')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv11a_dense2')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv11b_dense2')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv11b_dense2')
             )

            (self.feed('pool_transition1',
                       'conv1b_dense2',
                       'conv2b_dense2',
                       'conv3b_dense2',
                       'conv4b_dense2',
                       'conv5b_dense2',
                       'conv6b_dense2',
                       'conv7b_dense2',
                       'conv8b_dense2',
                       'conv9b_dense2',
                       'conv10b_dense2',
                       'conv11b_dense2'
                       )
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv12a_dense2')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv12a_dense2')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv12b_dense2')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv12b_dense2')
             )

            (self.feed('pool_transition1',
                       'conv1b_dense2',
                       'conv2b_dense2',
                       'conv3b_dense2',
                       'conv4b_dense2',
                       'conv5b_dense2',
                       'conv6b_dense2',
                       'conv7b_dense2',
                       'conv8b_dense2',
                       'conv9b_dense2',
                       'conv10b_dense2',
                       'conv11b_dense2',
                       'conv12b_dense2'
                       )
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv_transition2')
             .conv(1, 1, (96 + 48 * 12) / 2, 1, 1, biased=False, relu=False, name='conv_transition2')
             .avg_pool(2, 2, 2, 2, name='pool_transition2')
             )

            (self.feed('pool_transition2')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv1a_dense3')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv1a_dense3')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv1b_dense3')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv1b_dense3')
             )

            (self.feed('pool_transition2',
                       'conv1b_dense3')
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv2a_dense3')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv2a_dense3')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv2b_dense3')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv2b_dense3')
             )

            (self.feed('pool_transition2',
                       'conv1b_dense3',
                       'conv2b_dense3')
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv3a_dense3')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv3a_dense3')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv3b_dense3')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv3b_dense3')
             )

            (self.feed('pool_transition2',
                       'conv1b_dense3',
                       'conv2b_dense3',
                       'conv3b_dense3')
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv4a_dense3')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv4a_dense3')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv4b_dense3')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv4b_dense3')
             )

            (self.feed('pool_transition2',
                       'conv1b_dense3',
                       'conv2b_dense3',
                       'conv3b_dense3',
                       'conv4b_dense3'
                       )
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv5a_dense3')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv5a_dense3')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv5b_dense3')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv5b_dense3')
             )

            (self.feed('pool_transition2',
                       'conv1b_dense3',
                       'conv2b_dense3',
                       'conv3b_dense3',
                       'conv4b_dense3',
                       'conv5b_dense3'
                       )
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv6a_dense3')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv6a_dense3')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv6b_dense3')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv6b_dense3')
             )

            (self.feed('pool_transition2',
                       'conv1b_dense3',
                       'conv2b_dense3',
                       'conv3b_dense3',
                       'conv4b_dense3',
                       'conv5b_dense3',
                       'conv6b_dense3'
                       )
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv7a_dense3')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv7a_dense3')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv7b_dense3')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv7b_dense3')
             )

            (self.feed('pool_transition2',
                       'conv1b_dense3',
                       'conv2b_dense3',
                       'conv3b_dense3',
                       'conv4b_dense3',
                       'conv5b_dense3',
                       'conv6b_dense3',
                       'conv7b_dense3'
                       )
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv8a_dense3')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv8a_dense3')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv8b_dense3')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv8b_dense3')
             )

            (self.feed('pool_transition2',
                       'conv1b_dense3',
                       'conv2b_dense3',
                       'conv3b_dense3',
                       'conv4b_dense3',
                       'conv5b_dense3',
                       'conv6b_dense3',
                       'conv7b_dense3',
                       'conv8b_dense3'
                       )
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv9a_dense3')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv9a_dense3')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv9b_dense3')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv9b_dense3')
             )

            (self.feed('pool_transition2',
                       'conv1b_dense3',
                       'conv2b_dense3',
                       'conv3b_dense3',
                       'conv4b_dense3',
                       'conv5b_dense3',
                       'conv6b_dense3',
                       'conv7b_dense3',
                       'conv8b_dense3',
                       'conv9b_dense3'
                       )
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv10a_dense3')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv10a_dense3')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv10b_dense3')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv10b_dense3')
             )

            (self.feed('pool_transition2',
                       'conv1b_dense3',
                       'conv2b_dense3',
                       'conv3b_dense3',
                       'conv4b_dense3',
                       'conv5b_dense3',
                       'conv6b_dense3',
                       'conv7b_dense3',
                       'conv8b_dense3',
                       'conv9b_dense3',
                       'conv10b_dense3'
                       )
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv11a_dense3')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv11a_dense3')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv11b_dense3')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv11b_dense3')
             )

            (self.feed('pool_transition2',
                       'conv1b_dense3',
                       'conv2b_dense3',
                       'conv3b_dense3',
                       'conv4b_dense3',
                       'conv5b_dense3',
                       'conv6b_dense3',
                       'conv7b_dense3',
                       'conv8b_dense3',
                       'conv9b_dense3',
                       'conv10b_dense3',
                       'conv11b_dense3'
                       )
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv12a_dense3')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv12a_dense3')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv12b_dense3')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv12b_dense3')
             )

            (self.feed('pool_transition2',
                       'conv1b_dense3',
                       'conv2b_dense3',
                       'conv3b_dense3',
                       'conv4b_dense3',
                       'conv5b_dense3',
                       'conv6b_dense3',
                       'conv7b_dense3',
                       'conv8b_dense3',
                       'conv9b_dense3',
                       'conv10b_dense3',
                       'conv11b_dense3',
                       'conv12b_dense3'
                       )
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv13a_dense3')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv13a_dense3')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv13b_dense3')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv13b_dense3')
             )

            (self.feed('pool_transition2',
                       'conv1b_dense3',
                       'conv2b_dense3',
                       'conv3b_dense3',
                       'conv4b_dense3',
                       'conv5b_dense3',
                       'conv6b_dense3',
                       'conv7b_dense3',
                       'conv8b_dense3',
                       'conv9b_dense3',
                       'conv10b_dense3',
                       'conv11b_dense3',
                       'conv12b_dense3',
                       'conv13b_dense3')
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv14a_dense3')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv14a_dense3')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv14b_dense3')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv14b_dense3')
             )

            (self.feed('pool_transition2',
                       'conv1b_dense3',
                       'conv2b_dense3',
                       'conv3b_dense3',
                       'conv4b_dense3',
                       'conv5b_dense3',
                       'conv6b_dense3',
                       'conv7b_dense3',
                       'conv8b_dense3',
                       'conv9b_dense3',
                       'conv10b_dense3',
                       'conv11b_dense3',
                       'conv12b_dense3',
                       'conv13b_dense3',
                       'conv14b_dense3')
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv15a_dense3')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv15a_dense3')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv15b_dense3')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv15b_dense3')
             )

            (self.feed('pool_transition2',
                       'conv1b_dense3',
                       'conv2b_dense3',
                       'conv3b_dense3',
                       'conv4b_dense3',
                       'conv5b_dense3',
                       'conv6b_dense3',
                       'conv7b_dense3',
                       'conv8b_dense3',
                       'conv9b_dense3',
                       'conv10b_dense3',
                       'conv11b_dense3',
                       'conv12b_dense3',
                       'conv13b_dense3',
                       'conv14b_dense3',
                       'conv15b_dense3')
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv16a_dense3')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv16a_dense3')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv16b_dense3')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv16b_dense3')
             )

            (self.feed('pool_transition2',
                       'conv1b_dense3',
                       'conv2b_dense3',
                       'conv3b_dense3',
                       'conv4b_dense3',
                       'conv5b_dense3',
                       'conv6b_dense3',
                       'conv7b_dense3',
                       'conv8b_dense3',
                       'conv9b_dense3',
                       'conv10b_dense3',
                       'conv11b_dense3',
                       'conv12b_dense3',
                       'conv13b_dense3',
                       'conv14b_dense3',
                       'conv15b_dense3',
                       'conv16b_dense3'
                       )
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv17a_dense3')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv17a_dense3')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv17b_dense3')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv17b_dense3')
             )

            (self.feed('pool_transition2',
                       'conv1b_dense3',
                       'conv2b_dense3',
                       'conv3b_dense3',
                       'conv4b_dense3',
                       'conv5b_dense3',
                       'conv6b_dense3',
                       'conv7b_dense3',
                       'conv8b_dense3',
                       'conv9b_dense3',
                       'conv10b_dense3',
                       'conv11b_dense3',
                       'conv12b_dense3',
                       'conv13b_dense3',
                       'conv14b_dense3',
                       'conv15b_dense3',
                       'conv16b_dense3',
                       'conv17b_dense3'
                       )
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv18a_dense3')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv18a_dense3')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv18b_dense3')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv18b_dense3')
             )

            (self.feed('pool_transition2',
                       'conv1b_dense3',
                       'conv2b_dense3',
                       'conv3b_dense3',
                       'conv4b_dense3',
                       'conv5b_dense3',
                       'conv6b_dense3',
                       'conv7b_dense3',
                       'conv8b_dense3',
                       'conv9b_dense3',
                       'conv10b_dense3',
                       'conv11b_dense3',
                       'conv12b_dense3',
                       'conv13b_dense3',
                       'conv14b_dense3',
                       'conv15b_dense3',
                       'conv16b_dense3',
                       'conv17b_dense3',
                       'conv18b_dense3'
                       )
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv19a_dense3')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv19a_dense3')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv19b_dense3')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv19b_dense3')
             )

            (self.feed('pool_transition2',
                       'conv1b_dense3',
                       'conv2b_dense3',
                       'conv3b_dense3',
                       'conv4b_dense3',
                       'conv5b_dense3',
                       'conv6b_dense3',
                       'conv7b_dense3',
                       'conv8b_dense3',
                       'conv9b_dense3',
                       'conv10b_dense3',
                       'conv11b_dense3',
                       'conv12b_dense3',
                       'conv13b_dense3',
                       'conv14b_dense3',
                       'conv15b_dense3',
                       'conv16b_dense3',
                       'conv17b_dense3',
                       'conv18b_dense3',
                       'conv19b_dense3'
                       )
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv20a_dense3')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv20a_dense3')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv20b_dense3')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv20b_dense3')
             )

            (self.feed('pool_transition2',
                       'conv1b_dense3',
                       'conv2b_dense3',
                       'conv3b_dense3',
                       'conv4b_dense3',
                       'conv5b_dense3',
                       'conv6b_dense3',
                       'conv7b_dense3',
                       'conv8b_dense3',
                       'conv9b_dense3',
                       'conv10b_dense3',
                       'conv11b_dense3',
                       'conv12b_dense3',
                       'conv13b_dense3',
                       'conv14b_dense3',
                       'conv15b_dense3',
                       'conv16b_dense3',
                       'conv17b_dense3',
                       'conv18b_dense3',
                       'conv19b_dense3',
                       'conv20b_dense3'
                       )
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv21a_dense3')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv21a_dense3')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv21b_dense3')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv21b_dense3')
             )

            (self.feed('pool_transition2',
                       'conv1b_dense3',
                       'conv2b_dense3',
                       'conv3b_dense3',
                       'conv4b_dense3',
                       'conv5b_dense3',
                       'conv6b_dense3',
                       'conv7b_dense3',
                       'conv8b_dense3',
                       'conv9b_dense3',
                       'conv10b_dense3',
                       'conv11b_dense3',
                       'conv12b_dense3',
                       'conv13b_dense3',
                       'conv14b_dense3',
                       'conv15b_dense3',
                       'conv16b_dense3',
                       'conv17b_dense3',
                       'conv18b_dense3',
                       'conv19b_dense3',
                       'conv20b_dense3',
                       'conv21b_dense3'
                       )
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv22a_dense3')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv22a_dense3')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv22b_dense3')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv22b_dense3')
             )

            (self.feed('pool_transition2',
                       'conv1b_dense3',
                       'conv2b_dense3',
                       'conv3b_dense3',
                       'conv4b_dense3',
                       'conv5b_dense3',
                       'conv6b_dense3',
                       'conv7b_dense3',
                       'conv8b_dense3',
                       'conv9b_dense3',
                       'conv10b_dense3',
                       'conv11b_dense3',
                       'conv12b_dense3',
                       'conv13b_dense3',
                       'conv14b_dense3',
                       'conv15b_dense3',
                       'conv16b_dense3',
                       'conv17b_dense3',
                       'conv18b_dense3',
                       'conv19b_dense3',
                       'conv20b_dense3',
                       'conv21b_dense3',
                       'conv22b_dense3'
                       )
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv23a_dense3')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv23a_dense3')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv23b_dense3')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv23b_dense3')
             )

            (self.feed('pool_transition2',
                       'conv1b_dense3',
                       'conv2b_dense3',
                       'conv3b_dense3',
                       'conv4b_dense3',
                       'conv5b_dense3',
                       'conv6b_dense3',
                       'conv7b_dense3',
                       'conv8b_dense3',
                       'conv9b_dense3',
                       'conv10b_dense3',
                       'conv11b_dense3',
                       'conv12b_dense3',
                       'conv13b_dense3',
                       'conv14b_dense3',
                       'conv15b_dense3',
                       'conv16b_dense3',
                       'conv17b_dense3',
                       'conv18b_dense3',
                       'conv19b_dense3',
                       'conv20b_dense3',
                       'conv21b_dense3',
                       'conv22b_dense3',
                       'conv23b_dense3'
                       )
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv24a_dense3')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv24a_dense3')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv24b_dense3')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv24b_dense3')
             )

            (self.feed('pool_transition2',
                       'conv1b_dense3',
                       'conv2b_dense3',
                       'conv3b_dense3',
                       'conv4b_dense3',
                       'conv5b_dense3',
                       'conv6b_dense3',
                       'conv7b_dense3',
                       'conv8b_dense3',
                       'conv9b_dense3',
                       'conv10b_dense3',
                       'conv11b_dense3',
                       'conv12b_dense3',
                       'conv13b_dense3',
                       'conv14b_dense3',
                       'conv15b_dense3',
                       'conv16b_dense3',
                       'conv17b_dense3',
                       'conv18b_dense3',
                       'conv19b_dense3',
                       'conv20b_dense3',
                       'conv21b_dense3',
                       'conv22b_dense3',
                       'conv23b_dense3',
                       'conv24b_dense3')
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv25a_dense3')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv25a_dense3')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv25b_dense3')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv25b_dense3')
             )

            (self.feed('pool_transition2',
                       'conv1b_dense3',
                       'conv2b_dense3',
                       'conv3b_dense3',
                       'conv4b_dense3',
                       'conv5b_dense3',
                       'conv6b_dense3',
                       'conv7b_dense3',
                       'conv8b_dense3',
                       'conv9b_dense3',
                       'conv10b_dense3',
                       'conv11b_dense3',
                       'conv12b_dense3',
                       'conv13b_dense3',
                       'conv14b_dense3',
                       'conv15b_dense3',
                       'conv16b_dense3',
                       'conv17b_dense3',
                       'conv18b_dense3',
                       'conv19b_dense3',
                       'conv20b_dense3',
                       'conv21b_dense3',
                       'conv22b_dense3',
                       'conv23b_dense3',
                       'conv24b_dense3',
                       'conv25b_dense3')
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv26a_dense3')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv26a_dense3')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv26b_dense3')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv26b_dense3')
             )

            (self.feed('pool_transition2',
                       'conv1b_dense3',
                       'conv2b_dense3',
                       'conv3b_dense3',
                       'conv4b_dense3',
                       'conv5b_dense3',
                       'conv6b_dense3',
                       'conv7b_dense3',
                       'conv8b_dense3',
                       'conv9b_dense3',
                       'conv10b_dense3',
                       'conv11b_dense3',
                       'conv12b_dense3',
                       'conv13b_dense3',
                       'conv14b_dense3',
                       'conv15b_dense3',
                       'conv16b_dense3',
                       'conv17b_dense3',
                       'conv18b_dense3',
                       'conv19b_dense3',
                       'conv20b_dense3',
                       'conv21b_dense3',
                       'conv22b_dense3',
                       'conv23b_dense3',
                       'conv24b_dense3',
                       'conv25b_dense3',
                       'conv26b_dense3')
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv27a_dense3')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv27a_dense3')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv27b_dense3')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv27b_dense3')
             )

            (self.feed('pool_transition2',
                       'conv1b_dense3',
                       'conv2b_dense3',
                       'conv3b_dense3',
                       'conv4b_dense3',
                       'conv5b_dense3',
                       'conv6b_dense3',
                       'conv7b_dense3',
                       'conv8b_dense3',
                       'conv9b_dense3',
                       'conv10b_dense3',
                       'conv11b_dense3',
                       'conv12b_dense3',
                       'conv13b_dense3',
                       'conv14b_dense3',
                       'conv15b_dense3',
                       'conv16b_dense3',
                       'conv17b_dense3',
                       'conv18b_dense3',
                       'conv19b_dense3',
                       'conv20b_dense3',
                       'conv21b_dense3',
                       'conv22b_dense3',
                       'conv23b_dense3',
                       'conv24b_dense3',
                       'conv25b_dense3',
                       'conv26b_dense3',
                       'conv27b_dense3')
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv28a_dense3')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv28a_dense3')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv28b_dense3')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv28b_dense3')
             )

            (self.feed('pool_transition2',
                       'conv1b_dense3',
                       'conv2b_dense3',
                       'conv3b_dense3',
                       'conv4b_dense3',
                       'conv5b_dense3',
                       'conv6b_dense3',
                       'conv7b_dense3',
                       'conv8b_dense3',
                       'conv9b_dense3',
                       'conv10b_dense3',
                       'conv11b_dense3',
                       'conv12b_dense3',
                       'conv13b_dense3',
                       'conv14b_dense3',
                       'conv15b_dense3',
                       'conv16b_dense3',
                       'conv17b_dense3',
                       'conv18b_dense3',
                       'conv19b_dense3',
                       'conv20b_dense3',
                       'conv21b_dense3',
                       'conv22b_dense3',
                       'conv23b_dense3',
                       'conv24b_dense3',
                       'conv25b_dense3',
                       'conv26b_dense3',
                       'conv27b_dense3',
                       'conv28b_dense3'
                       )
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv29a_dense3')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv29a_dense3')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv29b_dense3')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv29b_dense3')
             )

            (self.feed('pool_transition2',
                       'conv1b_dense3',
                       'conv2b_dense3',
                       'conv3b_dense3',
                       'conv4b_dense3',
                       'conv5b_dense3',
                       'conv6b_dense3',
                       'conv7b_dense3',
                       'conv8b_dense3',
                       'conv9b_dense3',
                       'conv10b_dense3',
                       'conv11b_dense3',
                       'conv12b_dense3',
                       'conv13b_dense3',
                       'conv14b_dense3',
                       'conv15b_dense3',
                       'conv16b_dense3',
                       'conv17b_dense3',
                       'conv18b_dense3',
                       'conv19b_dense3',
                       'conv20b_dense3',
                       'conv21b_dense3',
                       'conv22b_dense3',
                       'conv23b_dense3',
                       'conv24b_dense3',
                       'conv25b_dense3',
                       'conv26b_dense3',
                       'conv27b_dense3',
                       'conv28b_dense3',
                       'conv29b_dense3'
                       )
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv30a_dense3')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv30a_dense3')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv30b_dense3')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv30b_dense3')
             )

            (self.feed('pool_transition2',
                       'conv1b_dense3',
                       'conv2b_dense3',
                       'conv3b_dense3',
                       'conv4b_dense3',
                       'conv5b_dense3',
                       'conv6b_dense3',
                       'conv7b_dense3',
                       'conv8b_dense3',
                       'conv9b_dense3',
                       'conv10b_dense3',
                       'conv11b_dense3',
                       'conv12b_dense3',
                       'conv13b_dense3',
                       'conv14b_dense3',
                       'conv15b_dense3',
                       'conv16b_dense3',
                       'conv17b_dense3',
                       'conv18b_dense3',
                       'conv19b_dense3',
                       'conv20b_dense3',
                       'conv21b_dense3',
                       'conv22b_dense3',
                       'conv23b_dense3',
                       'conv24b_dense3',
                       'conv25b_dense3',
                       'conv26b_dense3',
                       'conv27b_dense3',
                       'conv28b_dense3',
                       'conv29b_dense3',
                       'conv30b_dense3'
                       )
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv31a_dense3')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv31a_dense3')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv31b_dense3')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv31b_dense3')
             )

            (self.feed('pool_transition2',
                       'conv1b_dense3',
                       'conv2b_dense3',
                       'conv3b_dense3',
                       'conv4b_dense3',
                       'conv5b_dense3',
                       'conv6b_dense3',
                       'conv7b_dense3',
                       'conv8b_dense3',
                       'conv9b_dense3',
                       'conv10b_dense3',
                       'conv11b_dense3',
                       'conv12b_dense3',
                       'conv13b_dense3',
                       'conv14b_dense3',
                       'conv15b_dense3',
                       'conv16b_dense3',
                       'conv17b_dense3',
                       'conv18b_dense3',
                       'conv19b_dense3',
                       'conv20b_dense3',
                       'conv21b_dense3',
                       'conv22b_dense3',
                       'conv23b_dense3',
                       'conv24b_dense3',
                       'conv25b_dense3',
                       'conv26b_dense3',
                       'conv27b_dense3',
                       'conv28b_dense3',
                       'conv29b_dense3',
                       'conv30b_dense3',
                       'conv31b_dense3'
                       )
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv32a_dense3')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv32a_dense3')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv32b_dense3')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv32b_dense3')
             )

            (self.feed('pool_transition2',
                       'conv1b_dense3',
                       'conv2b_dense3',
                       'conv3b_dense3',
                       'conv4b_dense3',
                       'conv5b_dense3',
                       'conv6b_dense3',
                       'conv7b_dense3',
                       'conv8b_dense3',
                       'conv9b_dense3',
                       'conv10b_dense3',
                       'conv11b_dense3',
                       'conv12b_dense3',
                       'conv13b_dense3',
                       'conv14b_dense3',
                       'conv15b_dense3',
                       'conv16b_dense3',
                       'conv17b_dense3',
                       'conv18b_dense3',
                       'conv19b_dense3',
                       'conv20b_dense3',
                       'conv21b_dense3',
                       'conv22b_dense3',
                       'conv23b_dense3',
                       'conv24b_dense3',
                       'conv25b_dense3',
                       'conv26b_dense3',
                       'conv27b_dense3',
                       'conv28b_dense3',
                       'conv29b_dense3',
                       'conv30b_dense3',
                       'conv31b_dense3',
                       'conv32b_dense3'
                       )
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv33a_dense3')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv33a_dense3')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv33b_dense3')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv33b_dense3')
             )

            (self.feed('pool_transition2',
                       'conv1b_dense3',
                       'conv2b_dense3',
                       'conv3b_dense3',
                       'conv4b_dense3',
                       'conv5b_dense3',
                       'conv6b_dense3',
                       'conv7b_dense3',
                       'conv8b_dense3',
                       'conv9b_dense3',
                       'conv10b_dense3',
                       'conv11b_dense3',
                       'conv12b_dense3',
                       'conv13b_dense3',
                       'conv14b_dense3',
                       'conv15b_dense3',
                       'conv16b_dense3',
                       'conv17b_dense3',
                       'conv18b_dense3',
                       'conv19b_dense3',
                       'conv20b_dense3',
                       'conv21b_dense3',
                       'conv22b_dense3',
                       'conv23b_dense3',
                       'conv24b_dense3',
                       'conv25b_dense3',
                       'conv26b_dense3',
                       'conv27b_dense3',
                       'conv28b_dense3',
                       'conv29b_dense3',
                       'conv30b_dense3',
                       'conv31b_dense3',
                       'conv32b_dense3',
                       'conv33b_dense3'
                       )
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv34a_dense3')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv34a_dense3')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv34b_dense3')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv34b_dense3')
             )

            (self.feed('pool_transition2',
                       'conv1b_dense3',
                       'conv2b_dense3',
                       'conv3b_dense3',
                       'conv4b_dense3',
                       'conv5b_dense3',
                       'conv6b_dense3',
                       'conv7b_dense3',
                       'conv8b_dense3',
                       'conv9b_dense3',
                       'conv10b_dense3',
                       'conv11b_dense3',
                       'conv12b_dense3',
                       'conv13b_dense3',
                       'conv14b_dense3',
                       'conv15b_dense3',
                       'conv16b_dense3',
                       'conv17b_dense3',
                       'conv18b_dense3',
                       'conv19b_dense3',
                       'conv20b_dense3',
                       'conv21b_dense3',
                       'conv22b_dense3',
                       'conv23b_dense3',
                       'conv24b_dense3',
                       'conv25b_dense3',
                       'conv26b_dense3',
                       'conv27b_dense3',
                       'conv28b_dense3',
                       'conv29b_dense3',
                       'conv30b_dense3',
                       'conv31b_dense3',
                       'conv32b_dense3',
                       'conv33b_dense3',
                       'conv34b_dense3'
                       )
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv35a_dense3')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv35a_dense3')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv35b_dense3')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv35b_dense3')
             )

            (self.feed('pool_transition2',
                       'conv1b_dense3',
                       'conv2b_dense3',
                       'conv3b_dense3',
                       'conv4b_dense3',
                       'conv5b_dense3',
                       'conv6b_dense3',
                       'conv7b_dense3',
                       'conv8b_dense3',
                       'conv9b_dense3',
                       'conv10b_dense3',
                       'conv11b_dense3',
                       'conv12b_dense3',
                       'conv13b_dense3',
                       'conv14b_dense3',
                       'conv15b_dense3',
                       'conv16b_dense3',
                       'conv17b_dense3',
                       'conv18b_dense3',
                       'conv19b_dense3',
                       'conv20b_dense3',
                       'conv21b_dense3',
                       'conv22b_dense3',
                       'conv23b_dense3',
                       'conv24b_dense3',
                       'conv25b_dense3',
                       'conv26b_dense3',
                       'conv27b_dense3',
                       'conv28b_dense3',
                       'conv29b_dense3',
                       'conv30b_dense3',
                       'conv31b_dense3',
                       'conv32b_dense3',
                       'conv33b_dense3',
                       'conv34b_dense3',
                       'conv35b_dense3'
                       )
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv36a_dense3')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv36a_dense3')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv36b_dense3')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv36b_dense3')
             )

            (self.feed('pool_transition2',
                       'conv1b_dense3',
                       'conv2b_dense3',
                       'conv3b_dense3',
                       'conv4b_dense3',
                       'conv5b_dense3',
                       'conv6b_dense3',
                       'conv7b_dense3',
                       'conv8b_dense3',
                       'conv9b_dense3',
                       'conv10b_dense3',
                       'conv11b_dense3',
                       'conv12b_dense3',
                       'conv13b_dense3',
                       'conv14b_dense3',
                       'conv15b_dense3',
                       'conv16b_dense3',
                       'conv17b_dense3',
                       'conv18b_dense3',
                       'conv19b_dense3',
                       'conv20b_dense3',
                       'conv21b_dense3',
                       'conv22b_dense3',
                       'conv23b_dense3',
                       'conv24b_dense3',
                       'conv25b_dense3',
                       'conv26b_dense3',
                       'conv27b_dense3',
                       'conv28b_dense3',
                       'conv29b_dense3',
                       'conv30b_dense3',
                       'conv31b_dense3',
                       'conv32b_dense3',
                       'conv33b_dense3',
                       'conv34b_dense3',
                       'conv35b_dense3',
                       'conv36b_dense3'
                       )
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv_transition3')
             .conv(1, 1, (96 + 48 * 36) / 2, 1, 1, biased=False, relu=False, name='conv_transition3')
             .avg_pool(2, 2, 2, 2, name='pool_transition3')
             )

            (self.feed('pool_transition3')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv1a_dense4')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv1a_dense4')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv1b_dense4')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv1b_dense4')
             )

            (self.feed('pool_transition3',
                       'conv1b_dense4')
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv2a_dense4')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv2a_dense4')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv2b_dense4')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv2b_dense4')
             )

            (self.feed('pool_transition3',
                       'conv1b_dense4',
                       'conv2b_dense4')
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv3a_dense4')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv3a_dense4')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv3b_dense4')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv3b_dense4')
             )

            (self.feed('pool_transition3',
                       'conv1b_dense4',
                       'conv2b_dense4',
                       'conv3b_dense4')
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv4a_dense4')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv4a_dense4')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv4b_dense4')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv4b_dense4')
             )

            (self.feed('pool_transition3',
                       'conv1b_dense4',
                       'conv2b_dense4',
                       'conv3b_dense4',
                       'conv4b_dense4'
                       )
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv5a_dense4')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv5a_dense4')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv5b_dense4')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv5b_dense4')
             )

            (self.feed('pool_transition3',
                       'conv1b_dense4',
                       'conv2b_dense4',
                       'conv3b_dense4',
                       'conv4b_dense4',
                       'conv5b_dense4'
                       )
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv6a_dense4')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv6a_dense4')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv6b_dense4')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv6b_dense4')
             )

            (self.feed('pool_transition3',
                       'conv1b_dense4',
                       'conv2b_dense4',
                       'conv3b_dense4',
                       'conv4b_dense4',
                       'conv5b_dense4',
                       'conv6b_dense4'
                       )
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv7a_dense4')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv7a_dense4')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv7b_dense4')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv7b_dense4')
             )

            (self.feed('pool_transition3',
                       'conv1b_dense4',
                       'conv2b_dense4',
                       'conv3b_dense4',
                       'conv4b_dense4',
                       'conv5b_dense4',
                       'conv6b_dense4',
                       'conv7b_dense4'
                       )
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv8a_dense4')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv8a_dense4')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv8b_dense4')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv8b_dense4')
             )

            (self.feed('pool_transition3',
                       'conv1b_dense4',
                       'conv2b_dense4',
                       'conv3b_dense4',
                       'conv4b_dense4',
                       'conv5b_dense4',
                       'conv6b_dense4',
                       'conv7b_dense4',
                       'conv8b_dense4'
                       )
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv9a_dense4')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv9a_dense4')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv9b_dense4')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv9b_dense4')
             )

            (self.feed('pool_transition3',
                       'conv1b_dense4',
                       'conv2b_dense4',
                       'conv3b_dense4',
                       'conv4b_dense4',
                       'conv5b_dense4',
                       'conv6b_dense4',
                       'conv7b_dense4',
                       'conv8b_dense4',
                       'conv9b_dense4'
                       )
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv10a_dense4')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv10a_dense4')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv10b_dense4')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv10b_dense4')
             )

            (self.feed('pool_transition3',
                       'conv1b_dense4',
                       'conv2b_dense4',
                       'conv3b_dense4',
                       'conv4b_dense4',
                       'conv5b_dense4',
                       'conv6b_dense4',
                       'conv7b_dense4',
                       'conv8b_dense4',
                       'conv9b_dense4',
                       'conv10b_dense4'
                       )
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv11a_dense4')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv11a_dense4')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv11b_dense4')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv11b_dense4')
             )

            (self.feed('pool_transition3',
                       'conv1b_dense4',
                       'conv2b_dense4',
                       'conv3b_dense4',
                       'conv4b_dense4',
                       'conv5b_dense4',
                       'conv6b_dense4',
                       'conv7b_dense4',
                       'conv8b_dense4',
                       'conv9b_dense4',
                       'conv10b_dense4',
                       'conv11b_dense4'
                       )
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv12a_dense4')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv12a_dense4')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv12b_dense4')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv12b_dense4')
             )

            (self.feed('pool_transition3',
                       'conv1b_dense4',
                       'conv2b_dense4',
                       'conv3b_dense4',
                       'conv4b_dense4',
                       'conv5b_dense4',
                       'conv6b_dense4',
                       'conv7b_dense4',
                       'conv8b_dense4',
                       'conv9b_dense4',
                       'conv10b_dense4',
                       'conv11b_dense4',
                       'conv12b_dense4'
                       )
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv13a_dense4')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv13a_dense4')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv13b_dense4')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv13b_dense4')
             )

            (self.feed('pool_transition3',
                       'conv1b_dense4',
                       'conv2b_dense4',
                       'conv3b_dense4',
                       'conv4b_dense4',
                       'conv5b_dense4',
                       'conv6b_dense4',
                       'conv7b_dense4',
                       'conv8b_dense4',
                       'conv9b_dense4',
                       'conv10b_dense4',
                       'conv11b_dense4',
                       'conv12b_dense4',
                       'conv13b_dense4')
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv14a_dense4')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv14a_dense4')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv14b_dense4')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv14b_dense4')
             )

            (self.feed('pool_transition3',
                       'conv1b_dense4',
                       'conv2b_dense4',
                       'conv3b_dense4',
                       'conv4b_dense4',
                       'conv5b_dense4',
                       'conv6b_dense4',
                       'conv7b_dense4',
                       'conv8b_dense4',
                       'conv9b_dense4',
                       'conv10b_dense4',
                       'conv11b_dense4',
                       'conv12b_dense4',
                       'conv13b_dense4',
                       'conv14b_dense4')
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv15a_dense4')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv15a_dense4')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv15b_dense4')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv15b_dense4')
             )

            (self.feed('pool_transition3',
                       'conv1b_dense4',
                       'conv2b_dense4',
                       'conv3b_dense4',
                       'conv4b_dense4',
                       'conv5b_dense4',
                       'conv6b_dense4',
                       'conv7b_dense4',
                       'conv8b_dense4',
                       'conv9b_dense4',
                       'conv10b_dense4',
                       'conv11b_dense4',
                       'conv12b_dense4',
                       'conv13b_dense4',
                       'conv14b_dense4',
                       'conv15b_dense4')
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv16a_dense4')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv16a_dense4')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv16b_dense4')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv16b_dense4')
             )

            (self.feed('pool_transition3',
                       'conv1b_dense4',
                       'conv2b_dense4',
                       'conv3b_dense4',
                       'conv4b_dense4',
                       'conv5b_dense4',
                       'conv6b_dense4',
                       'conv7b_dense4',
                       'conv8b_dense4',
                       'conv9b_dense4',
                       'conv10b_dense4',
                       'conv11b_dense4',
                       'conv12b_dense4',
                       'conv13b_dense4',
                       'conv14b_dense4',
                       'conv15b_dense4',
                       'conv16b_dense4'
                       )
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv17a_dense4')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv17a_dense4')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv17b_dense4')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv17b_dense4')
             )

            (self.feed('pool_transition3',
                       'conv1b_dense4',
                       'conv2b_dense4',
                       'conv3b_dense4',
                       'conv4b_dense4',
                       'conv5b_dense4',
                       'conv6b_dense4',
                       'conv7b_dense4',
                       'conv8b_dense4',
                       'conv9b_dense4',
                       'conv10b_dense4',
                       'conv11b_dense4',
                       'conv12b_dense4',
                       'conv13b_dense4',
                       'conv14b_dense4',
                       'conv15b_dense4',
                       'conv16b_dense4',
                       'conv17b_dense4'
                       )
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv18a_dense4')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv18a_dense4')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv18b_dense4')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv18b_dense4')
             )

            (self.feed('pool_transition3',
                       'conv1b_dense4',
                       'conv2b_dense4',
                       'conv3b_dense4',
                       'conv4b_dense4',
                       'conv5b_dense4',
                       'conv6b_dense4',
                       'conv7b_dense4',
                       'conv8b_dense4',
                       'conv9b_dense4',
                       'conv10b_dense4',
                       'conv11b_dense4',
                       'conv12b_dense4',
                       'conv13b_dense4',
                       'conv14b_dense4',
                       'conv15b_dense4',
                       'conv16b_dense4',
                       'conv17b_dense4',
                       'conv18b_dense4'
                       )
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv19a_dense4')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv19a_dense4')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv19b_dense4')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv19b_dense4')
             )

            (self.feed('pool_transition3',
                       'conv1b_dense4',
                       'conv2b_dense4',
                       'conv3b_dense4',
                       'conv4b_dense4',
                       'conv5b_dense4',
                       'conv6b_dense4',
                       'conv7b_dense4',
                       'conv8b_dense4',
                       'conv9b_dense4',
                       'conv10b_dense4',
                       'conv11b_dense4',
                       'conv12b_dense4',
                       'conv13b_dense4',
                       'conv14b_dense4',
                       'conv15b_dense4',
                       'conv16b_dense4',
                       'conv17b_dense4',
                       'conv18b_dense4',
                       'conv19b_dense4'
                       )
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv20a_dense4')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv20a_dense4')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv20b_dense4')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv20b_dense4')
             )

            (self.feed('pool_transition3',
                       'conv1b_dense4',
                       'conv2b_dense4',
                       'conv3b_dense4',
                       'conv4b_dense4',
                       'conv5b_dense4',
                       'conv6b_dense4',
                       'conv7b_dense4',
                       'conv8b_dense4',
                       'conv9b_dense4',
                       'conv10b_dense4',
                       'conv11b_dense4',
                       'conv12b_dense4',
                       'conv13b_dense4',
                       'conv14b_dense4',
                       'conv15b_dense4',
                       'conv16b_dense4',
                       'conv17b_dense4',
                       'conv18b_dense4',
                       'conv19b_dense4',
                       'conv20b_dense4'
                       )
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv21a_dense4')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv21a_dense4')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv21b_dense4')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv21b_dense4')
             )

            (self.feed('pool_transition3',
                       'conv1b_dense4',
                       'conv2b_dense4',
                       'conv3b_dense4',
                       'conv4b_dense4',
                       'conv5b_dense4',
                       'conv6b_dense4',
                       'conv7b_dense4',
                       'conv8b_dense4',
                       'conv9b_dense4',
                       'conv10b_dense4',
                       'conv11b_dense4',
                       'conv12b_dense4',
                       'conv13b_dense4',
                       'conv14b_dense4',
                       'conv15b_dense4',
                       'conv16b_dense4',
                       'conv17b_dense4',
                       'conv18b_dense4',
                       'conv19b_dense4',
                       'conv20b_dense4',
                       'conv21b_dense4'
                       )
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv22a_dense4')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv22a_dense4')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv22b_dense4')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv22b_dense4')
             )

            (self.feed('pool_transition3',
                       'conv1b_dense4',
                       'conv2b_dense4',
                       'conv3b_dense4',
                       'conv4b_dense4',
                       'conv5b_dense4',
                       'conv6b_dense4',
                       'conv7b_dense4',
                       'conv8b_dense4',
                       'conv9b_dense4',
                       'conv10b_dense4',
                       'conv11b_dense4',
                       'conv12b_dense4',
                       'conv13b_dense4',
                       'conv14b_dense4',
                       'conv15b_dense4',
                       'conv16b_dense4',
                       'conv17b_dense4',
                       'conv18b_dense4',
                       'conv19b_dense4',
                       'conv20b_dense4',
                       'conv21b_dense4',
                       'conv22b_dense4'
                       )
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv23a_dense4')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv23a_dense4')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv23b_dense4')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv23b_dense4')
             )

            (self.feed('pool_transition3',
                       'conv1b_dense4',
                       'conv2b_dense4',
                       'conv3b_dense4',
                       'conv4b_dense4',
                       'conv5b_dense4',
                       'conv6b_dense4',
                       'conv7b_dense4',
                       'conv8b_dense4',
                       'conv9b_dense4',
                       'conv10b_dense4',
                       'conv11b_dense4',
                       'conv12b_dense4',
                       'conv13b_dense4',
                       'conv14b_dense4',
                       'conv15b_dense4',
                       'conv16b_dense4',
                       'conv17b_dense4',
                       'conv18b_dense4',
                       'conv19b_dense4',
                       'conv20b_dense4',
                       'conv21b_dense4',
                       'conv22b_dense4',
                       'conv23b_dense4'
                       )
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv24a_dense4')
             .conv(1, 1, 192, 1, 1, biased=False, relu=False, name='conv24a_dense4')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv24b_dense4')
             .conv(3, 3, 48, 1, 1, biased=False, relu=False, name='conv24b_dense4')
             )

            (self.feed('pool_transition3',
                       'conv1b_dense4',
                       'conv2b_dense4',
                       'conv3b_dense4',
                       'conv4b_dense4',
                       'conv5b_dense4',
                       'conv6b_dense4',
                       'conv7b_dense4',
                       'conv8b_dense4',
                       'conv9b_dense4',
                       'conv10b_dense4',
                       'conv11b_dense4',
                       'conv12b_dense4',
                       'conv13b_dense4',
                       'conv14b_dense4',
                       'conv15b_dense4',
                       'conv16b_dense4',
                       'conv17b_dense4',
                       'conv18b_dense4',
                       'conv19b_dense4',
                       'conv20b_dense4',
                       'conv21b_dense4',
                       'conv22b_dense4',
                       'conv23b_dense4',
                       'conv24b_dense4')
             .concat(axis=-1)
             .resize(14, 14, name='bilinear_upsample1')
             )

            (self.feed('pool_transition2',
                       'conv1b_dense3',
                       'conv2b_dense3',
                       'conv3b_dense3',
                       'conv4b_dense3',
                       'conv5b_dense3',
                       'conv6b_dense3',
                       'conv7b_dense3',
                       'conv8b_dense3',
                       'conv9b_dense3',
                       'conv10b_dense3',
                       'conv11b_dense3',
                       'conv12b_dense3',
                       'conv13b_dense3',
                       'conv14b_dense3',
                       'conv15b_dense3',
                       'conv16b_dense3',
                       'conv17b_dense3',
                       'conv18b_dense3',
                       'conv19b_dense3',
                       'conv20b_dense3',
                       'conv21b_dense3',
                       'conv22b_dense3',
                       'conv23b_dense3',
                       'conv24b_dense3',
                       'conv25b_dense3',
                       'conv26b_dense3',
                       'conv27b_dense3',
                       'conv28b_dense3',
                       'conv29b_dense3',
                       'conv30b_dense3',
                       'conv31b_dense3',
                       'conv32b_dense3',
                       'conv33b_dense3',
                       'conv34b_dense3',
                       'conv35b_dense3',
                       'conv36b_dense3',
                       'bilinear_upsample1')
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv_upsample1')
             .conv(3, 3, 768, 1, 1, biased=False, relu=False, name='conv_upsample1')
             .resize(28, 28, name='bilinear_upsample2')
             )

            (self.feed('pool_transition1',
                       'conv1b_dense2',
                       'conv2b_dense2',
                       'conv3b_dense2',
                       'conv4b_dense2',
                       'conv5b_dense2',
                       'conv6b_dense2',
                       'conv7b_dense2',
                       'conv8b_dense2',
                       'conv9b_dense2',
                       'conv10b_dense2',
                       'conv11b_dense2',
                       'conv12b_dense2',
                       'bilinear_upsample2')
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv_upsample2')
             .conv(3, 3, 384, 1, 1, biased=False, relu=False, name='conv_upsample2')
             .resize(56, 56, name='bilinear_upsample3')
             )

            (self.feed('pool1',
                       'conv1b_dense1',
                       'conv2b_dense1',
                       'conv3b_dense1',
                       'conv4b_dense1',
                       'conv5b_dense1',
                       'conv6b_dense1',
                       'bilinear_upsample3')
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv_upsample3')
             .conv(3, 3, 96, 1, 1, biased=False, relu=False, name='conv_upsample3')
             .resize(112, 112, name='bilinear_upsample4')
             )

            (self.feed('conv1',
                       'bilinear_upsample4')
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv_upsample4')
             .conv(3, 3, 96, 1, 1, biased=False, relu=False, name='conv_upsample4')
             .resize(224, 224, name='bilinear_upsample5')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv_upsample5')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='conv_upsample5')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv2')
             .conv(1, 1, 3, 1, 1, biased=False, relu=False, name='conv2')
             .softmax(axis=-1, name='conv2_softmax')
             )

        with tf.device('/gpu:1'):
            (self.feed('conv2',
                       'data')
             .concat(axis=-1)
             .expand_dims(axis=0)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='3d_bn_conv1')
             .conv3D(7, 7, 7, 96, 2, 2, 2, biased=False, relu=False, name='3d_conv1')
             .max_pool3D(2, 2, 2, 2, 2, 2, name='3d_pool1')
             )

            (self.feed('3d_pool1')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='3d_bn_conv1a_dense1')
             .conv3D(1, 1, 1, 128, 1, 1, 1, biased=False, relu=False, name='3d_conv1a_dense1')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='3d_bn_conv1b_dense1')
             .conv3D(3, 3, 3, 32, 1, 1, 1, biased=False, relu=False, name='3d_conv1b_dense1')
             )

            (self.feed('3d_pool1',
                       '3d_conv1b_dense1')
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='3d_bn_conv2a_dense1')
             .conv3D(1, 1, 1, 128, 1, 1, 1, biased=False, relu=False, name='3d_conv2a_dense1')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='3d_bn_conv2b_dense1')
             .conv3D(3, 3, 3, 32, 1, 1, 1, biased=False, relu=False, name='3d_conv2b_dense1')
             )

            (self.feed('3d_pool1',
                       '3d_conv1b_dense1',
                       '3d_conv2b_dense1')
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='3d_bn_conv3a_dense1')
             .conv3D(1, 1, 1, 128, 1, 1, 1, biased=False, relu=False, name='3d_conv3a_dense1')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='3d_bn_conv3b_dense1')
             .conv3D(3, 3, 3, 32, 1, 1, 1, biased=False, relu=False, name='3d_conv3b_dense1')
             )

            (self.feed('3d_pool1',
                       '3d_conv1b_dense1',
                       '3d_conv2b_dense1',
                       '3d_conv3b_dense1',
                       )
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='3d_bn_conv_transition1')
             .conv3D(1, 1, 1, (96 + 32 * 3) / 2, 1, 1, 1, biased=False, relu=False, name='3d_conv_transition1')
             .avg_pool3D(2, 2, 1, 2, 2, 1, name='3d_pool_transition1')
             )

            (self.feed('3d_pool_transition1')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='3d_bn_conv1a_dense2')
             .conv3D(1, 1, 1, 128, 1, 1, 1, biased=False, relu=False, name='3d_conv1a_dense2')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='3d_bn_conv1b_dense2')
             .conv3D(3, 3, 3, 32, 1, 1, 1, biased=False, relu=False, name='3d_conv1b_dense2')
             )

            (self.feed('3d_pool_transition1',
                       '3d_conv1b_dense2')
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='3d_bn_conv2a_dense2')
             .conv3D(1, 1, 1, 128, 1, 1, 1, biased=False, relu=False, name='3d_conv2a_dense2')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='3d_bn_conv2b_dense2')
             .conv3D(3, 3, 3, 32, 1, 1, 1, biased=False, relu=False, name='3d_conv2b_dense2')
             )

            (self.feed('3d_pool_transition1',
                       '3d_conv1b_dense2',
                       '3d_conv2b_dense2')
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='3d_bn_conv3a_dense2')
             .conv3D(1, 1, 1, 128, 1, 1, 1, biased=False, relu=False, name='3d_conv3a_dense2')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='3d_bn_conv3b_dense2')
             .conv3D(3, 3, 3, 32, 1, 1, 1, biased=False, relu=False, name='3d_conv3b_dense2')
             )

            (self.feed('3d_pool_transition1',
                       '3d_conv1b_dense2',
                       '3d_conv2b_dense2',
                       '3d_conv3b_dense2')
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='3d_bn_conv4a_dense2')
             .conv3D(1, 1, 1, 128, 1, 1, 1, biased=False, relu=False, name='3d_conv4a_dense2')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='3d_bn_conv4b_dense2')
             .conv3D(3, 3, 3, 32, 1, 1, 1, biased=False, relu=False, name='3d_conv4b_dense2')
             )

            (self.feed('3d_pool_transition1',
                       '3d_conv1b_dense2',
                       '3d_conv2b_dense2',
                       '3d_conv3b_dense2',
                       '3d_conv4b_dense2'
                       )
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='3d_bn_conv_transition2')
             .conv3D(1, 1, 1, ((96 + 32 * 3) / 2 + 32 * 4) / 2, 1, 1, 1, biased=False, relu=False,
                     name='3d_conv_transition2')
             .avg_pool3D(2, 2, 1, 2, 2, 1, name='3d_pool_transition2')
             )

            (self.feed('3d_pool_transition2')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='3d_bn_conv1a_dense3')
             .conv3D(1, 1, 1, 128, 1, 1, 1, biased=False, relu=False, name='3d_conv1a_dense3')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='3d_bn_conv1b_dense3')
             .conv3D(3, 3, 3, 32, 1, 1, 1, biased=False, relu=False, name='3d_conv1b_dense3')
             )

            (self.feed('3d_pool_transition2',
                       '3d_conv1b_dense3')
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='3d_bn_conv2a_dense3')
             .conv3D(1, 1, 1, 128, 1, 1, 1, biased=False, relu=False, name='3d_conv2a_dense3')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='3d_bn_conv2b_dense3')
             .conv3D(3, 3, 3, 32, 1, 1, 1, biased=False, relu=False, name='3d_conv2b_dense3')
             )

            (self.feed('3d_pool_transition2',
                       '3d_conv1b_dense3',
                       '3d_conv2b_dense3')
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='3d_bn_conv3a_dense3')
             .conv3D(1, 1, 1, 128, 1, 1, 1, biased=False, relu=False, name='3d_conv3a_dense3')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='3d_bn_conv3b_dense3')
             .conv3D(3, 3, 3, 32, 1, 1, 1, biased=False, relu=False, name='3d_conv3b_dense3')
             )

            (self.feed('3d_pool_transition2',
                       '3d_conv1b_dense3',
                       '3d_conv2b_dense3',
                       '3d_conv3b_dense3')
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='3d_bn_conv4a_dense3')
             .conv3D(1, 1, 1, 128, 1, 1, 1, biased=False, relu=False, name='3d_conv4a_dense3')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='3d_bn_conv4b_dense3')
             .conv3D(3, 3, 3, 32, 1, 1, 1, biased=False, relu=False, name='3d_conv4b_dense3')
             )

            (self.feed('3d_pool_transition2',
                       '3d_conv1b_dense3',
                       '3d_conv2b_dense3',
                       '3d_conv3b_dense3',
                       '3d_conv4b_dense3'
                       )
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='3d_bn_conv5a_dense3')
             .conv3D(1, 1, 1, 128, 1, 1, 1, biased=False, relu=False, name='3d_conv5a_dense3')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='3d_bn_conv5b_dense3')
             .conv3D(3, 3, 3, 32, 1, 1, 1, biased=False, relu=False, name='3d_conv5b_dense3')
             )

            (self.feed('3d_pool_transition2',
                       '3d_conv1b_dense3',
                       '3d_conv2b_dense3',
                       '3d_conv3b_dense3',
                       '3d_conv4b_dense3',
                       '3d_conv5b_dense3'
                       )
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='3d_bn_conv6a_dense3')
             .conv3D(1, 1, 1, 128, 1, 1, 1, biased=False, relu=False, name='3d_conv6a_dense3')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='3d_bn_conv6b_dense3')
             .conv3D(3, 3, 3, 32, 1, 1, 1, biased=False, relu=False, name='3d_conv6b_dense3')
             )

            (self.feed('3d_pool_transition2',
                       '3d_conv1b_dense3',
                       '3d_conv2b_dense3',
                       '3d_conv3b_dense3',
                       '3d_conv4b_dense3',
                       '3d_conv5b_dense3',
                       '3d_conv6b_dense3'
                       )
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='3d_bn_conv7a_dense3')
             .conv3D(1, 1, 1, 128, 1, 1, 1, biased=False, relu=False, name='3d_conv7a_dense3')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='3d_bn_conv7b_dense3')
             .conv3D(3, 3, 3, 32, 1, 1, 1, biased=False, relu=False, name='3d_conv7b_dense3')
             )

            (self.feed('3d_pool_transition2',
                       '3d_conv1b_dense3',
                       '3d_conv2b_dense3',
                       '3d_conv3b_dense3',
                       '3d_conv4b_dense3',
                       '3d_conv5b_dense3',
                       '3d_conv6b_dense3',
                       '3d_conv7b_dense3'
                       )
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='3d_bn_conv8a_dense3')
             .conv3D(1, 1, 1, 128, 1, 1, 1, biased=False, relu=False, name='3d_conv8a_dense3')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='3d_bn_conv8b_dense3')
             .conv3D(3, 3, 3, 32, 1, 1, 1, biased=False, relu=False, name='3d_conv8b_dense3')
             )

            (self.feed('3d_pool_transition2',
                       '3d_conv1b_dense3',
                       '3d_conv2b_dense3',
                       '3d_conv3b_dense3',
                       '3d_conv4b_dense3',
                       '3d_conv5b_dense3',
                       '3d_conv6b_dense3',
                       '3d_conv7b_dense3',
                       '3d_conv8b_dense3'
                       )
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='3d_bn_conv9a_dense3')
             .conv3D(1, 1, 1, 128, 1, 1, 1, biased=False, relu=False, name='3d_conv9a_dense3')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='3d_bn_conv9b_dense3')
             .conv3D(3, 3, 3, 32, 1, 1, 1, biased=False, relu=False, name='3d_conv9b_dense3')
             )

            (self.feed('3d_pool_transition2',
                       '3d_conv1b_dense3',
                       '3d_conv2b_dense3',
                       '3d_conv3b_dense3',
                       '3d_conv4b_dense3',
                       '3d_conv5b_dense3',
                       '3d_conv6b_dense3',
                       '3d_conv7b_dense3',
                       '3d_conv8b_dense3',
                       '3d_conv9b_dense3'
                       )
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='3d_bn_conv10a_dense3')
             .conv3D(1, 1, 1, 128, 1, 1, 1, biased=False, relu=False, name='3d_conv10a_dense3')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='3d_bn_conv10b_dense3')
             .conv3D(3, 3, 3, 32, 1, 1, 1, biased=False, relu=False, name='3d_conv10b_dense3')
             )

            (self.feed('3d_pool_transition2',
                       '3d_conv1b_dense3',
                       '3d_conv2b_dense3',
                       '3d_conv3b_dense3',
                       '3d_conv4b_dense3',
                       '3d_conv5b_dense3',
                       '3d_conv6b_dense3',
                       '3d_conv7b_dense3',
                       '3d_conv8b_dense3',
                       '3d_conv9b_dense3',
                       '3d_conv10b_dense3'
                       )
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='3d_bn_conv11a_dense3')
             .conv3D(1, 1, 1, 128, 1, 1, 1, biased=False, relu=False, name='3d_conv11a_dense3')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='3d_bn_conv11b_dense3')
             .conv3D(3, 3, 3, 32, 1, 1, 1, biased=False, relu=False, name='3d_conv11b_dense3')
             )

            (self.feed('3d_pool_transition2',
                       '3d_conv1b_dense3',
                       '3d_conv2b_dense3',
                       '3d_conv3b_dense3',
                       '3d_conv4b_dense3',
                       '3d_conv5b_dense3',
                       '3d_conv6b_dense3',
                       '3d_conv7b_dense3',
                       '3d_conv8b_dense3',
                       '3d_conv9b_dense3',
                       '3d_conv10b_dense3',
                       '3d_conv11b_dense3'
                       )
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='3d_bn_conv12a_dense3')
             .conv3D(1, 1, 1, 128, 1, 1, 1, biased=False, relu=False, name='3d_conv12a_dense3')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='3d_bn_conv12b_dense3')
             .conv3D(3, 3, 3, 32, 1, 1, 1, biased=False, relu=False, name='3d_conv12b_dense3')
             )

            (self.feed('3d_pool_transition2',
                       '3d_conv1b_dense3',
                       '3d_conv2b_dense3',
                       '3d_conv3b_dense3',
                       '3d_conv4b_dense3',
                       '3d_conv5b_dense3',
                       '3d_conv6b_dense3',
                       '3d_conv7b_dense3',
                       '3d_conv8b_dense3',
                       '3d_conv9b_dense3',
                       '3d_conv10b_dense3',
                       '3d_conv11b_dense3',
                       '3d_conv12b_dense3'
                       )
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='3d_bn_conv_transition3')
             .conv3D(1, 1, 1, (((96 + 32 * 3) / 2 + 32 * 4) / 2 + 32 * 12) / 2, 1, 1, 1, biased=False, relu=False,
                     name='3d_conv_transition3')
             .avg_pool3D(2, 2, 1, 2, 2, 1, name='3d_pool_transition3')
             )

            (self.feed('3d_pool_transition3')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='3d_bn_conv1a_dense4')
             .conv3D(1, 1, 1, 128, 1, 1, 1, biased=False, relu=False, name='3d_conv1a_dense4')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='3d_bn_conv1b_dense4')
             .conv3D(3, 3, 3, 32, 1, 1, 1, biased=False, relu=False, name='3d_conv1b_dense4')
             )

            (self.feed('3d_pool_transition3',
                       '3d_conv1b_dense4')
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='3d_bn_conv2a_dense4')
             .conv3D(1, 1, 1, 128, 1, 1, 1, biased=False, relu=False, name='3d_conv2a_dense4')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='3d_bn_conv2b_dense4')
             .conv3D(3, 3, 3, 32, 1, 1, 1, biased=False, relu=False, name='3d_conv2b_dense4')
             )

            (self.feed('3d_pool_transition3',
                       '3d_conv1b_dense4',
                       '3d_conv2b_dense4')
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='3d_bn_conv3a_dense4')
             .conv3D(1, 1, 1, 128, 1, 1, 1, biased=False, relu=False, name='3d_conv3a_dense4')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='3d_bn_conv3b_dense4')
             .conv3D(3, 3, 3, 32, 1, 1, 1, biased=False, relu=False, name='3d_conv3b_dense4')
             )

            (self.feed('3d_pool_transition3',
                       '3d_conv1b_dense4',
                       '3d_conv2b_dense4',
                       '3d_conv3b_dense4')
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='3d_bn_conv4a_dense4')
             .conv3D(1, 1, 1, 128, 1, 1, 1, biased=False, relu=False, name='3d_conv4a_dense4')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='3d_bn_conv4b_dense4')
             .conv3D(3, 3, 3, 32, 1, 1, 1, biased=False, relu=False, name='3d_conv4b_dense4')
             )

            (self.feed('3d_pool_transition3',
                       '3d_conv1b_dense4',
                       '3d_conv2b_dense4',
                       '3d_conv3b_dense4',
                       '3d_conv4b_dense4'
                       )
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='3d_bn_conv5a_dense4')
             .conv3D(1, 1, 1, 128, 1, 1, 1, biased=False, relu=False, name='3d_conv5a_dense4')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='3d_bn_conv5b_dense4')
             .conv3D(3, 3, 3, 32, 1, 1, 1, biased=False, relu=False, name='3d_conv5b_dense4')
             )

            (self.feed('3d_pool_transition3',
                       '3d_conv1b_dense4',
                       '3d_conv2b_dense4',
                       '3d_conv3b_dense4',
                       '3d_conv4b_dense4',
                       '3d_conv5b_dense4'
                       )
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='3d_bn_conv6a_dense4')
             .conv3D(1, 1, 1, 128, 1, 1, 1, biased=False, relu=False, name='3d_conv6a_dense4')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='3d_bn_conv6b_dense4')
             .conv3D(3, 3, 3, 32, 1, 1, 1, biased=False, relu=False, name='3d_conv6b_dense4')
             )

            (self.feed('3d_pool_transition3',
                       '3d_conv1b_dense4',
                       '3d_conv2b_dense4',
                       '3d_conv3b_dense4',
                       '3d_conv4b_dense4',
                       '3d_conv5b_dense4',
                       '3d_conv6b_dense4'
                       )
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='3d_bn_conv7a_dense4')
             .conv3D(1, 1, 1, 128, 1, 1, 1, biased=False, relu=False, name='3d_conv7a_dense4')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='3d_bn_conv7b_dense4')
             .conv3D(3, 3, 3, 32, 1, 1, 1, biased=False, relu=False, name='3d_conv7b_dense4')
             )

            (self.feed('3d_pool_transition3',
                       '3d_conv1b_dense4',
                       '3d_conv2b_dense4',
                       '3d_conv3b_dense4',
                       '3d_conv4b_dense4',
                       '3d_conv5b_dense4',
                       '3d_conv6b_dense4',
                       '3d_conv7b_dense4'
                       )
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='3d_bn_conv8a_dense4')
             .conv3D(1, 1, 1, 128, 1, 1, 1, biased=False, relu=False, name='3d_conv8a_dense4')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='3d_bn_conv8b_dense4')
             .conv3D(3, 3, 3, 32, 1, 1, 1, biased=False, relu=False, name='3d_conv8b_dense4')
             )

            (self.feed('3d_pool_transition3',
                       '3d_conv1b_dense4',
                       '3d_conv2b_dense4',
                       '3d_conv3b_dense4',
                       '3d_conv4b_dense4',
                       '3d_conv5b_dense4',
                       '3d_conv6b_dense4',
                       '3d_conv7b_dense4',
                       '3d_conv8b_dense4'
                       )
             .concat(axis=-1)
             .deconv3D(2, 2, 1, 504, 2, 2, 1, 14, 14, 3, relu=False, name='3d_bilinear_upsample1')
             )

            (self.feed('pool_transition2',
                       'conv1b_dense3',
                       'conv2b_dense3',
                       'conv3b_dense3',
                       'conv4b_dense3',
                       'conv5b_dense3',
                       'conv6b_dense3',
                       'conv7b_dense3',
                       'conv8b_dense3',
                       'conv9b_dense3',
                       'conv10b_dense3',
                       'conv11b_dense3',
                       'conv12b_dense3',
                       'conv13b_dense3',
                       'conv14b_dense3',
                       'conv15b_dense3',
                       'conv16b_dense3',
                       'conv17b_dense3',
                       'conv18b_dense3',
                       'conv19b_dense3',
                       'conv20b_dense3',
                       'conv21b_dense3',
                       'conv22b_dense3',
                       'conv23b_dense3',
                       'conv24b_dense3',
                       'conv25b_dense3',
                       'conv26b_dense3',
                       'conv27b_dense3',
                       'conv28b_dense3',
                       'conv29b_dense3',
                       'conv30b_dense3',
                       'conv31b_dense3',
                       'conv32b_dense3',
                       'conv33b_dense3',
                       'conv34b_dense3',
                       'conv35b_dense3',
                       'conv36b_dense3'
                       )
             .concat(axis=-1)
             .expand_dims(axis=0)
             .reshape([1, 3, 14, 14, -1], name='2d_dense3')
             )

            (self.feed('3d_pool_transition2',
                       '3d_conv1b_dense3',
                       '3d_conv2b_dense3',
                       '3d_conv3b_dense3',
                       '3d_conv4b_dense3',
                       '3d_conv5b_dense3',
                       '3d_conv6b_dense3',
                       '3d_conv7b_dense3',
                       '3d_conv8b_dense3',
                       '3d_conv9b_dense3',
                       '3d_conv10b_dense3',
                       '3d_conv11b_dense3',
                       '3d_conv12b_dense3',
                       '3d_bilinear_upsample1',
                       '2d_dense3'
                       )
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='3d_bn_conv_upsample1')
             .deconv3D(2, 2, 1, 504, 2, 2, 1, 28, 28, 3, relu=False, name='3d_bilinear_upsample2')
             )

            (self.feed('pool_transition1',
                       'conv1b_dense2',
                       'conv2b_dense2',
                       'conv3b_dense2',
                       'conv4b_dense2',
                       'conv5b_dense2',
                       'conv6b_dense2',
                       'conv7b_dense2',
                       'conv8b_dense2',
                       'conv9b_dense2',
                       'conv10b_dense2',
                       'conv11b_dense2',
                       'conv12b_dense2'
                       )
             .concat(axis=-1)
             .expand_dims(axis=0)
             .reshape([1, 3, 28, 28, -1], name='2d_dense2')
             )

            (self.feed('3d_pool_transition1',
                       '3d_conv1b_dense2',
                       '3d_conv2b_dense2',
                       '3d_conv3b_dense2',
                       '3d_conv4b_dense2',
                       '3d_bilinear_upsample2',
                       '2d_dense2'
                       )
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='3d_bn_conv_upsample2')
             .deconv3D(2, 2, 1, 224, 2, 2, 1, 56, 56, 3, name='3d_bilinear_upsample3')
             )

            (self.feed('pool1',
                       'conv1b_dense1',
                       'conv2b_dense1',
                       'conv3b_dense1',
                       'conv4b_dense1',
                       'conv5b_dense1',
                       'conv6b_dense1'
                       )
             .concat(axis=-1)
             .expand_dims(axis=0)
             .reshape([1, 3, 56, 56, -1], name='2d_dense1')
             )

            (self.feed('3d_pool1',
                       '3d_conv1b_dense1',
                       '3d_conv2b_dense1',
                       '3d_conv3b_dense1',
                       '3d_bilinear_upsample3',
                       '2d_dense1'
                       )
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='3d_bn_conv_upsample3')
             .deconv3D(2, 2, 2, 192, 2, 2, 2, 112, 112, 6, name='3d_bilinear_upsample4')
             )

            (self.feed('conv1')
             .expand_dims(axis=0)
             .reshape([1, 6, 112, 112, -1], name='2d_conv1')
             )

            (self.feed('2d_conv1',
                       '3d_conv1',
                       '3d_bilinear_upsample4'
                       )
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='3d_bn_conv_upsample4')
             .deconv3D(2, 2, 2, 96, 2, 2, 2, 224, 224, 12, name='3d_bilinear_upsample5')
             )

            (self.feed('bilinear_upsample5')
             .expand_dims(axis=0)
             .reshape([1, 12, 224, 224, -1], name='2d_bilinear_upsample5')
             )

            (self.feed('2d_bilinear_upsample5',
                       '3d_bilinear_upsample5'
                       )
             .add()
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='3d_bn_conv_upsample5')
             .conv3D(3, 3, 3, 64, 1, 1, 1, biased=False, relu=False, name='3d_conv_upsample5')
             )

            (self.feed('conv2')
             .expand_dims(axis=0)
             .reshape([-1, 12, 224, 224, num_classes], name='2d_conv2')
             )

            (self.feed('3d_conv_upsample5',
                       '2d_conv2'
                       )
             .concat(axis=-1)
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='3d_bn_conv2')
             .conv3D(1, 1, 1, num_classes, 1, 1, 1, biased=False, relu=False, name='3d_conv2')
             )

            pass
