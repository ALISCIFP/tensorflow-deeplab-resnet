# Converted to TensorFlow .caffemodel
# with the DeepLab-ResNet configuration.
# The batch normalisation layer is provided by
# the slim library (https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim).

import tensorflow as tf

from kaffe.tensorflow import Network


class DeepLabResNetModel(Network):
    def setup(self, is_training, num_classes):
        '''Network definition.

        Args:
          is_training: whether to update the running mean and variance of the batch normalisation layer.
                       If the batch size is small, it is better to keep the running mean and variance of
                       the-pretrained model frozen.
          num_classes: number of classes to predict (including background).
        '''

        (self.feed('data')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv1')
         .conv3D(7, 7, 7, 96, 2, 2, 2, biased=False, relu=False, name='conv1')
         .max_pool3D(3, 3, 3, 2, 2, 2, name='pool1')
         )

        (self.feed('pool1')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv1a_dense1')
         .conv3D(1, 1, 1, 128, 1, 1, 1, biased=False, relu=False, name='conv1a_dense1')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv1b_dense1')
         .conv3D(3, 3, 3, 32, 1, 1, 1, biased=False, relu=False, name='conv1b_dense1')
         )

        (self.feed('pool1',
                   'conv1b_dense1')
         .concat(axis=-1)
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv2a_dense1')
         .conv3D(1, 1, 1, 128, 1, 1, 1, biased=False, relu=False, name='conv2a_dense1')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv2b_dense1')
         .conv3D(3, 3, 3, 32, 1, 1, 1, biased=False, relu=False, name='conv2b_dense1')
         )

        (self.feed('pool1',
                   'conv1b_dense1',
                   'conv2b_dense1')
         .concat(axis=-1)
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv3a_dense1')
         .conv3D(1, 1, 1, 128, 1, 1, 1, biased=False, relu=False, name='conv3a_dense1')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv3b_dense1')
         .conv3D(3, 3, 3, 32, 1, 1, 1, biased=False, relu=False, name='conv3b_dense1')
         )


        (self.feed('pool1',
                   'conv1b_dense1',
                   'conv2b_dense1',
                   'conv3b_dense1',
                   )
         .concat(axis=-1)
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv_transition1')
         .conv3D(1, 1, 1, (96 + 32 * 3) / 2, 1, 1, 1, biased=False, relu=False, name='conv_transition1')
         .avg_pool3D(2, 2, 1, 2, 2, 1, name='pool_transition1')
         )

        (self.feed('pool_transition1')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv1a_dense2')
         .conv3D(1, 1, 1, 128, 1, 1, 1, biased=False, relu=False, name='conv1a_dense2')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv1b_dense2')
         .conv3D(3, 3, 3, 32, 1, 1, 1, biased=False, relu=False, name='conv1b_dense2')
         )

        (self.feed('pool_transition1',
                   'conv1b_dense2')
         .concat(axis=-1)
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv2a_dense2')
         .conv3D(1, 1, 1, 128, 1, 1, 1, biased=False, relu=False, name='conv2a_dense2')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv2b_dense2')
         .conv3D(3, 3, 3, 32, 1, 1, 1, biased=False, relu=False, name='conv2b_dense2')
         )

        (self.feed('pool_transition1',
                   'conv1b_dense2',
                   'conv2b_dense2')
         .concat(axis=-1)
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv3a_dense2')
         .conv3D(1, 1, 1, 128, 1, 1, 1, biased=False, relu=False, name='conv3a_dense2')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv3b_dense2')
         .conv3D(3, 3, 3, 32, 1, 1, 1, biased=False, relu=False, name='conv3b_dense2')
         )

        (self.feed('pool_transition1',
                   'conv1b_dense2',
                   'conv2b_dense2',
                   'conv3b_dense2')
         .concat(axis=-1)
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv4a_dense2')
         .conv3D(1, 1, 1, 128, 1, 1, 1, biased=False, relu=False, name='conv4a_dense2')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv4b_dense2')
         .conv3D(3, 3, 3, 32, 1, 1, 1, biased=False, relu=False, name='conv4b_dense2')
         )

        (self.feed('pool_transition1',
                   'conv1b_dense2',
                   'conv2b_dense2',
                   'conv3b_dense2',
                   'conv4b_dense2'
                   )
         .concat(axis=-1)
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv_transition2')
         .conv3D(1, 1, 1, ((96 + 32 * 3) / 2 + 32 * 4) / 2, 1, 1, 1, biased=False, relu=False, name='conv_transition2')
         .avg_pool3D(2, 2, 1, 2, 2, 1, name='pool_transition2')
         )

        (self.feed('pool_transition2')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv1a_dense3')
         .conv3D(1, 1, 1, 128, 1, 1, 1, biased=False, relu=False, name='conv1a_dense3')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv1b_dense3')
         .conv3D(3, 3, 3, 32, 1, 1, 1, biased=False, relu=False, name='conv1b_dense3')
         )

        (self.feed('pool_transition2',
                   'conv1b_dense3')
         .concat(axis=-1)
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv2a_dense3')
         .conv3D(1, 1, 1, 192, 1, 1, 1, biased=False, relu=False, name='conv2a_dense3')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv2b_dense3')
         .conv3D(3, 3, 3, 48, 1, 1, 1, biased=False, relu=False, name='conv2b_dense3')
         )

        (self.feed('pool_transition2',
                   'conv1b_dense3',
                   'conv2b_dense3')
         .concat(axis=-1)
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv3a_dense3')
         .conv3D(1, 1, 1, 128, 1, 1, 1, biased=False, relu=False, name='conv3a_dense3')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv3b_dense3')
         .conv3D(3, 3, 3, 32, 1, 1, 1, biased=False, relu=False, name='conv3b_dense3')
         )

        (self.feed('pool_transition2',
                   'conv1b_dense3',
                   'conv2b_dense3',
                   'conv3b_dense3')
         .concat(axis=-1)
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv4a_dense3')
         .conv3D(1, 1, 1, 128, 1, 1, 1, biased=False, relu=False, name='conv4a_dense3')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv4b_dense3')
         .conv3D(3, 3, 3, 32, 1, 1, 1, biased=False, relu=False, name='conv4b_dense3')
         )

        (self.feed('pool_transition2',
                   'conv1b_dense3',
                   'conv2b_dense3',
                   'conv3b_dense3',
                   'conv4b_dense3'
                   )
         .concat(axis=-1)
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv5a_dense3')
         .conv3D(1, 1, 1, 128, 1, 1, 1, biased=False, relu=False, name='conv5a_dense3')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv5b_dense3')
         .conv3D(3, 3, 3, 32, 1, 1, 1, biased=False, relu=False, name='conv5b_dense3')
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
         .conv3D(1, 1, 1, 128, 1, 1, 1, biased=False, relu=False, name='conv6a_dense3')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv6b_dense3')
         .conv3D(3, 3, 3, 32, 1, 1, 1, biased=False, relu=False, name='conv6b_dense3')
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
         .conv3D(1, 1, 1, 128, 1, 1, 1, biased=False, relu=False, name='conv7a_dense3')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv7b_dense3')
         .conv3D(3, 3, 3, 32, 1, 1, 1, biased=False, relu=False, name='conv7b_dense3')
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
         .conv3D(1, 1, 1, 128, 1, 1, 1, biased=False, relu=False, name='conv8a_dense3')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv8b_dense3')
         .conv3D(3, 3, 3, 32, 1, 1, 1, biased=False, relu=False, name='conv8b_dense3')
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
         .conv3D(1, 1, 1, 128, 1, 1, 1, biased=False, relu=False, name='conv9a_dense3')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv9b_dense3')
         .conv3D(3, 3, 3, 32, 1, 1, 1, biased=False, relu=False, name='conv9b_dense3')
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
         .conv3D(1, 1, 1, 128, 1, 1, 1, biased=False, relu=False, name='conv10a_dense3')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv10b_dense3')
         .conv3D(3, 3, 3, 32, 1, 1, 1, biased=False, relu=False, name='conv10b_dense3')
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
         .conv3D(1, 1, 1, 128, 1, 1, 1, biased=False, relu=False, name='conv11a_dense3')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv11b_dense3')
         .conv3D(3, 3, 3, 32, 1, 1, 1, biased=False, relu=False, name='conv11b_dense3')
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
         .conv3D(1, 1, 1, 128, 1, 1, 1, biased=False, relu=False, name='conv12a_dense3')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv12b_dense3')
         .conv3D(3, 3, 3, 32, 1, 1, 1, biased=False, relu=False, name='conv12b_dense3')
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
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv_transition3')
         .conv3D(1, 1, 1, (((96 + 32 * 3) / 2 + 32 * 4) / 2 + 32 * 12) / 2, 1, 1, 1, biased=False, relu=False,
                 name='conv_transition3')
         .avg_pool3D(2, 2, 1, 2, 2, 1, name='pool_transition3')
         )

        (self.feed('pool_transition3')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv1a_dense4')
         .conv3D(1, 1, 1, 128, 1, 1, 1, biased=False, relu=False, name='conv1a_dense4')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv1b_dense4')
         .conv3D(3, 3, 3, 32, 1, 1, 1, biased=False, relu=False, name='conv1b_dense4')
         )

        (self.feed('pool_transition3',
                   'conv1b_dense4')
         .concat(axis=-1)
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv2a_dense4')
         .conv3D(1, 1, 1, 128, 1, 1, 1, biased=False, relu=False, name='conv2a_dense4')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv2b_dense4')
         .conv3D(3, 3, 3, 32, 1, 1, 1, biased=False, relu=False, name='conv2b_dense4')
         )

        (self.feed('pool_transition3',
                   'conv1b_dense4',
                   'conv2b_dense4')
         .concat(axis=-1)
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv3a_dense4')
         .conv3D(1, 1, 1, 128, 1, 1, 1, biased=False, relu=False, name='conv3a_dense4')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv3b_dense4')
         .conv3D(3, 3, 3, 32, 1, 1, 1, biased=False, relu=False, name='conv3b_dense4')
         )

        (self.feed('pool_transition3',
                   'conv1b_dense4',
                   'conv2b_dense4',
                   'conv3b_dense4')
         .concat(axis=-1)
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv4a_dense4')
         .conv3D(1, 1, 1, 128, 1, 1, 1, biased=False, relu=False, name='conv4a_dense4')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv4b_dense4')
         .conv3D(3, 3, 3, 32, 1, 1, 1, biased=False, relu=False, name='conv4b_dense4')
         )

        (self.feed('pool_transition3',
                   'conv1b_dense4',
                   'conv2b_dense4',
                   'conv3b_dense4',
                   'conv4b_dense4'
                   )
         .concat(axis=-1)
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv5a_dense4')
         .conv3D(1, 1, 1, 128, 1, 1, 1, biased=False, relu=False, name='conv5a_dense4')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv5b_dense4')
         .conv3D(3, 3, 3, 32, 1, 1, 1, biased=False, relu=False, name='conv5b_dense4')
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
         .conv3D(1, 1, 1, 128, 1, 1, 1, biased=False, relu=False, name='conv6a_dense4')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv6b_dense4')
         .conv3D(3, 3, 3, 32, 1, 1, 1, biased=False, relu=False, name='conv6b_dense4')
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
         .conv3D(1, 1, 1, 128, 1, 1, 1, biased=False, relu=False, name='conv7a_dense4')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv7b_dense4')
         .conv3D(3, 3, 3, 32, 1, 1, 1, biased=False, relu=False, name='conv7b_dense4')
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
         .conv3D(1, 1, 1, 128, 1, 1, 1, biased=False, relu=False, name='conv8a_dense4')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv8b_dense4')
         .conv3D(3, 3, 3, 32, 1, 1, 1, biased=False, relu=False, name='conv8b_dense4')
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
         .deconv3D(2, 2, 1, 504, 2, 2, 1, 14, 14, 3, relu=False, name='bilinear_upsample1')
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
                   'bilinear_upsample1')
         .concat(axis=-1)
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv_upsample1')
         # .conv3D(3, 3, 768, 1, 1, biased=False, relu=False, name='conv_upsample1')
         .deconv3D(2, 2, 1, 224, 2, 2, 1, 28, 28, 3, relu=False, name='bilinear_upsample2')
         )

        (self.feed('pool_transition1',
                   'conv1b_dense2',
                   'conv2b_dense2',
                   'conv3b_dense2',
                   'conv4b_dense2',
                   'bilinear_upsample2')
         .concat(axis=-1)
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv_upsample2')
         # .conv3D(3, 3, 384, 1, 1, biased=False, relu=False, name='conv_upsample2')
         .deconv3D(2, 2, 1, 192, 2, 2, 1, 56, 56, 3, name='bilinear_upsample3')
         )

        (self.feed('pool1',
                   'conv1b_dense1',
                   'conv2b_dense1',
                   'conv3b_dense1',
                   'bilinear_upsample3')
         .concat(axis=-1)
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv_upsample3')
         # .conv3D(3, 3, 96, 1, 1, biased=False, relu=False, name='conv_upsample3')
         .deconv3D(2, 2, 1, 96, 2, 2, 1, 112, 112, 3, name='bilinear_upsample4')
         )

        (self.feed('bilinear_upsample4')
         .concat(axis=-1)
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv_upsample4')
         # .conv3D(3, 3, 96, 1, 1, biased=False, relu=False, name='conv_upsample4')
         .deconv3D(2, 2, 1, 64, 2, 2, 1, 224, 224, 3, name='bilinear_upsample5')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv_upsample5')
         #.conv3D(3, 3, 64, 1, 1, biased=False, relu=False, name='conv_upsample5')
         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv2')
         .conv3D(1, 1, 1, 3, 1, 1, 1, biased=False, relu=False, name='conv2')
         )

        pass
