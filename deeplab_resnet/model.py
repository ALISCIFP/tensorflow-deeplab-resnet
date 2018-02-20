# Converted to TensorFlow .caffemodel
# with the DeepLab-ResNet configuration.
# The batch normalisation layer is provided by
# the slim library (https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim).

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
        # 320x320x64
        (self.feed('data')
         .conv(3, 3, 64 / 4, 1, 1, biased=False, relu=False, name='conv1_a')
         .prelu(name='conv1_a_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv1_a_bn'))

        (self.feed('data',
                   'conv1_a_bn')
         .concat(axis=-1)
         .conv(3, 3, 64 / 4, 1, 1, biased=False, relu=False, name='conv1_b')
         .prelu(name='conv1_b_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv1_b_bn'))

        (self.feed('data',
                   'conv1_a_bn',
                   'conv1_b_bn')
         .concat(axis=-1)
         .conv(3, 3, 64 / 4, 1, 1, biased=False, relu=False, name='conv1_c')
         .prelu(name='conv1_c_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv1_c_bn'))

        (self.feed('data',
                   'conv1_a_bn',
                   'conv1_b_bn',
                   'conv1_c_bn')
         .concat(axis=-1)
         .conv(3, 3, 64 / 4, 1, 1, biased=False, relu=False, name='conv1_d')
         .prelu(name='conv1_d_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv1_d_bn'))

        (self.feed('conv1_a_bn',
                   'conv1_b_bn',
                   'conv1_c_bn',
                   'conv1_d_bn')
         .concat(name='conv1_bn', axis=-1))

        # 320x320x64
        (self.feed('conv1_bn')
         .conv(3, 3, 64 / 4, 1, 1, biased=False, relu=False, name='conv2_a')
         .prelu(name='conv2_a_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv2_a_bn'))

        (self.feed('conv1_bn',
                   'conv2_a_bn')
         .concat(axis=-1)
         .conv(3, 3, 64 / 4, 1, 1, biased=False, relu=False, name='conv2_b')
         .prelu(name='conv2_b_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv2_b_bn'))

        (self.feed('conv1_bn',
                   'conv2_a_bn',
                   'conv2_b_bn')
         .concat(axis=-1)
         .conv(3, 3, 64 / 4, 1, 1, biased=False, relu=False, name='conv2_c')
         .prelu(name='conv2_c_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv2_c_bn'))

        (self.feed('conv1_bn',
                   'conv2_a_bn',
                   'conv2_b_bn',
                   'conv2_c_bn')
         .concat(axis=-1)
         .conv(3, 3, 64 / 4, 1, 1, biased=False, relu=False, name='conv2_d')
         .prelu(name='conv2_d_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv2_d_bn'))

        (self.feed('conv2_a_bn',
                   'conv2_b_bn',
                   'conv2_c_bn',
                   'conv2_d_bn')
         .concat(name='conv2_bn', axis=-1))

        # 320x320x128
        (self.feed('conv2_bn')
         .conv(2, 2, 128, 1, 1, biased=False, relu=False, name='conv2_to_3')
         .prelu(name='conv2_to_3_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv2_to_3_bn')
         .max_pool(2, 2, 2, 2, name='conv2_pool'))

        # 160x160x128
        (self.feed('conv2_pool')
         .conv(3, 3, 128 / 4, 1, 1, biased=False, relu=False, name='conv3_a')
         .prelu(name='conv3_a_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv3_a_bn'))

        (self.feed('conv2_pool',
                   'conv3_a_bn')
         .concat(axis=-1)
         .conv(3, 3, 128 / 4, 1, 1, biased=False, relu=False, name='conv3_b')
         .prelu(name='conv3_b_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv3_b_bn'))

        (self.feed('conv2_pool',
                   'conv3_a_bn',
                   'conv3_b_bn')
         .concat(axis=-1)
         .conv(3, 3, 128 / 4, 1, 1, biased=False, relu=False, name='conv3_c')
         .prelu(name='conv3_c_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv3_c_bn'))

        (self.feed('conv2_pool',
                   'conv3_a_bn',
                   'conv3_b_bn',
                   'conv3_c_bn')
         .concat(axis=-1)
         .conv(3, 3, 128 / 4, 1, 1, biased=False, relu=False, name='conv3_d')
         .prelu(name='conv3_d_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv3_d_bn'))

        (self.feed('conv3_a_bn',
                   'conv3_b_bn',
                   'conv3_c_bn',
                   'conv3_d_bn')
         .concat(name='conv3_bn', axis=-1))

        # 160x160x128
        (self.feed('conv3_bn')
         .conv(3, 3, 128 / 4, 1, 1, biased=False, relu=False, name='conv4_a')
         .prelu(name='conv4_a_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv4_a_bn'))

        (self.feed('conv3_bn',
                   'conv4_a_bn')
         .concat(axis=-1)
         .conv(3, 3, 128 / 4, 1, 1, biased=False, relu=False, name='conv4_b')
         .prelu(name='conv4_b_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv4_b_bn'))

        (self.feed('conv3_bn',
                   'conv4_a_bn',
                   'conv4_b_bn')
         .concat(axis=-1)
         .conv(3, 3, 128 / 4, 1, 1, biased=False, relu=False, name='conv4_c')
         .prelu(name='conv4_c_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv4_c_bn'))

        (self.feed('conv3_bn',
                   'conv4_a_bn',
                   'conv4_b_bn',
                   'conv4_c_bn')
         .concat(axis=-1)
         .conv(3, 3, 128 / 4, 1, 1, biased=False, relu=False, name='conv4_d')
         .prelu(name='conv4_d_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv4_d_bn'))

        (self.feed('conv4_a_bn',
                   'conv4_b_bn',
                   'conv4_c_bn',
                   'conv4_d_bn')
         .concat(name='conv4_bn', axis=-1))

        (self.feed('conv2_pool',
                   'conv4_bn')
         .add(name='conv4_sum'))

        # 160x160x256
        (self.feed('conv4_sum')
         .conv(2, 2, 256, 1, 1, biased=False, relu=False, name='conv4_to_5')
         .prelu(name='conv4_to_5_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv4_to_5_bn')
         .max_pool(2, 2, 2, 2, name='conv4_pool'))

        # 80x80x256
        (self.feed('conv4_pool')
         .conv(3, 3, 256 / 4, 1, 1, biased=False, relu=False, name='conv5_a')
         .prelu(name='conv5_a_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv5_a_bn'))

        (self.feed('conv4_pool',
                   'conv5_a_bn')
         .concat(axis=-1)
         .conv(3, 3, 256 / 4, 1, 1, biased=False, relu=False, name='conv5_b')
         .prelu(name='conv5_b_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv5_b_bn'))

        (self.feed('conv4_pool',
                   'conv5_a_bn',
                   'conv5_b_bn')
         .concat(axis=-1)
         .conv(3, 3, 256 / 4, 1, 1, biased=False, relu=False, name='conv5_c')
         .prelu(name='conv5_c_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv5_c_bn'))

        (self.feed('conv4_pool',
                   'conv5_a_bn',
                   'conv5_b_bn',
                   'conv5_c_bn')
         .concat(axis=-1)
         .conv(3, 3, 256 / 4, 1, 1, biased=False, relu=False, name='conv5_d')
         .prelu(name='conv5_d_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv5_d_bn'))

        (self.feed('conv5_a_bn',
                   'conv5_b_bn',
                   'conv5_c_bn',
                   'conv5_d_bn')
         .concat(name='conv5_bn', axis=-1))

        # 80x80x256
        (self.feed('conv5_bn')
         .conv(3, 3, 256 / 4, 1, 1, biased=False, relu=False, name='conv6_a')
         .prelu(name='conv6_a_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv6_a_bn'))

        (self.feed('conv5_bn',
                   'conv6_a_bn')
         .concat(axis=-1)
         .conv(3, 3, 256 / 4, 1, 1, biased=False, relu=False, name='conv6_b')
         .prelu(name='conv6_b_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv6_b_bn'))

        (self.feed('conv5_bn',
                   'conv6_a_bn',
                   'conv6_b_bn')
         .concat(axis=-1)
         .conv(3, 3, 256 / 4, 1, 1, biased=False, relu=False, name='conv6_c')
         .prelu(name='conv6_c_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv6_c_bn'))

        (self.feed('conv5_bn',
                   'conv6_a_bn',
                   'conv6_b_bn',
                   'conv6_c_bn')
         .concat(axis=-1)
         .conv(3, 3, 256 / 4, 1, 1, biased=False, relu=False, name='conv6_d')
         .prelu(name='conv6_d_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv6_d_bn'))

        (self.feed('conv6_a_bn',
                   'conv6_b_bn',
                   'conv6_c_bn',
                   'conv6_d_bn')
         .concat(name='conv6_bn', axis=-1))

        # 80x80x256
        (self.feed('conv6_bn')
         .conv(3, 3, 256 / 4, 1, 1, biased=False, relu=False, name='conv7_a')
         .prelu(name='conv7_a_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv7_a_bn'))

        (self.feed('conv6_bn',
                   'conv7_a_bn')
         .concat(axis=-1)
         .conv(3, 3, 256 / 4, 1, 1, biased=False, relu=False, name='conv7_b')
         .prelu(name='conv7_b_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv7_b_bn'))

        (self.feed('conv6_bn',
                   'conv7_a_bn',
                   'conv7_b_bn')
         .concat(axis=-1)
         .conv(3, 3, 256 / 4, 1, 1, biased=False, relu=False, name='conv7_c')
         .prelu(name='conv7_c_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv7_c_bn'))

        (self.feed('conv6_bn',
                   'conv7_a_bn',
                   'conv7_b_bn',
                   'conv7_c_bn')
         .concat(axis=-1)
         .conv(3, 3, 256 / 4, 1, 1, biased=False, relu=False, name='conv7_d')
         .prelu(name='conv7_d_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv7_d_bn'))

        (self.feed('conv7_a_bn',
                   'conv7_b_bn',
                   'conv7_c_bn',
                   'conv7_d_bn')
         .concat(name='conv7_bn', axis=-1))

        (self.feed('conv4_pool',
                   'conv7_bn')
         .add(name='conv7_sum'))

        # 80x80x512
        (self.feed('conv7_sum')
         .conv(2, 2, 512, 1, 1, biased=False, relu=False, name='conv7_to_8')
         .prelu(name='conv7_to_8_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv7_to_8_bn')
         .max_pool(2, 2, 2, 2, name='conv7_pool'))

        # 40x40x512
        (self.feed('conv7_pool')
         .conv(3, 3, 512 / 4, 1, 1, biased=False, relu=False, name='conv8_a')
         .prelu(name='conv8_a_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv8_a_bn'))

        (self.feed('conv7_pool',
                   'conv8_a_bn')
         .concat(axis=-1)
         .conv(3, 3, 512 / 4, 1, 1, biased=False, relu=False, name='conv8_b')
         .prelu(name='conv8_b_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv8_b_bn'))

        (self.feed('conv7_pool',
                   'conv8_a_bn',
                   'conv8_b_bn')
         .concat(axis=-1)
         .conv(3, 3, 512 / 4, 1, 1, biased=False, relu=False, name='conv8_c')
         .prelu(name='conv8_c_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv8_c_bn'))

        (self.feed('conv7_pool',
                   'conv8_a_bn',
                   'conv8_b_bn',
                   'conv8_c_bn')
         .concat(axis=-1)
         .conv(3, 3, 512 / 4, 1, 1, biased=False, relu=False, name='conv8_d')
         .prelu(name='conv8_d_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv8_d_bn'))

        (self.feed('conv8_a_bn',
                   'conv8_b_bn',
                   'conv8_c_bn',
                   'conv8_d_bn')
         .concat(name='conv8_bn', axis=-1))

        # 40x40x512
        (self.feed('conv8_bn')
         .conv(3, 3, 512 / 4, 1, 1, biased=False, relu=False, name='conv9_a')
         .prelu(name='conv9_a_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv9_a_bn'))

        (self.feed('conv8_bn',
                   'conv9_a_bn')
         .concat(axis=-1)
         .conv(3, 3, 512 / 4, 1, 1, biased=False, relu=False, name='conv9_b')
         .prelu(name='conv9_b_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv9_b_bn'))

        (self.feed('conv8_bn',
                   'conv9_a_bn',
                   'conv9_b_bn')
         .concat(axis=-1)
         .conv(3, 3, 512 / 4, 1, 1, biased=False, relu=False, name='conv9_c')
         .prelu(name='conv9_c_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv9_c_bn'))

        (self.feed('conv8_bn',
                   'conv9_a_bn',
                   'conv9_b_bn',
                   'conv9_c_bn')
         .concat(axis=-1)
         .conv(3, 3, 512 / 4, 1, 1, biased=False, relu=False, name='conv9_d')
         .prelu(name='conv9_d_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv9_d_bn'))

        (self.feed('conv9_a_bn',
                   'conv9_b_bn',
                   'conv9_c_bn',
                   'conv9_d_bn')
         .concat(name='conv9_bn', axis=-1))

        # 40x40x512
        (self.feed('conv9_bn')
         .conv(3, 3, 512 / 4, 1, 1, biased=False, relu=False, name='conv10_a')
         .prelu(name='conv10_a_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv10_a_bn'))

        (self.feed('conv9_bn',
                   'conv10_a_bn')
         .concat(axis=-1)
         .conv(3, 3, 512 / 4, 1, 1, biased=False, relu=False, name='conv10_b')
         .prelu(name='conv10_b_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv10_b_bn'))

        (self.feed('conv9_bn',
                   'conv10_a_bn',
                   'conv10_b_bn')
         .concat(axis=-1)
         .conv(3, 3, 512 / 4, 1, 1, biased=False, relu=False, name='conv10_c')
         .prelu(name='conv10_c_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv10_c_bn'))

        (self.feed('conv9_bn',
                   'conv10_a_bn',
                   'conv10_b_bn',
                   'conv10_c_bn')
         .concat(axis=-1)
         .conv(3, 3, 512 / 4, 1, 1, biased=False, relu=False, name='conv10_d')
         .prelu(name='conv10_d_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv10_d_bn'))

        (self.feed('conv10_a_bn',
                   'conv10_b_bn',
                   'conv10_c_bn',
                   'conv10_d_bn')
         .concat(name='conv10_bn', axis=-1))

        (self.feed('conv10_bn',
                   'conv7_pool')
         .add(name='conv10_sum')
         .max_pool(2, 2, 2, 2, name='conv10_pool'))

        # 20x20x512
        (self.feed('conv10_pool')
         .conv(3, 3, 512 / 4, 1, 1, biased=False, relu=False, name='conv11_a')
         .prelu(name='conv11_a_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv11_a_bn'))

        (self.feed('conv10_pool',
                   'conv11_a_bn')
         .concat(axis=-1)
         .conv(3, 3, 512 / 4, 1, 1, biased=False, relu=False, name='conv11_b')
         .prelu(name='conv11_b_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv11_b_bn'))

        (self.feed('conv10_pool',
                   'conv11_a_bn',
                   'conv11_b_bn')
         .concat(axis=-1)
         .conv(3, 3, 512 / 4, 1, 1, biased=False, relu=False, name='conv11_c')
         .prelu(name='conv11_c_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv11_c_bn'))

        (self.feed('conv10_pool',
                   'conv11_a_bn',
                   'conv11_b_bn',
                   'conv11_c_bn')
         .concat(axis=-1)
         .conv(3, 3, 512 / 4, 1, 1, biased=False, relu=False, name='conv11_d')
         .prelu(name='conv11_d_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv11_d_bn'))

        (self.feed('conv11_a_bn',
                   'conv11_b_bn',
                   'conv11_c_bn',
                   'conv11_d_bn')
         .concat(name='conv11_bn', axis=-1))

        # 20x20x512
        (self.feed('conv11_bn')
         .conv(3, 3, 512 / 4, 1, 1, biased=False, relu=False, name='conv12_a')
         .prelu(name='conv12_a_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv12_a_bn'))

        (self.feed('conv11_bn',
                   'conv12_a_bn')
         .concat(axis=-1)
         .conv(3, 3, 512 / 4, 1, 1, biased=False, relu=False, name='conv12_b')
         .prelu(name='conv12_b_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv12_b_bn'))

        (self.feed('conv11_bn',
                   'conv12_a_bn',
                   'conv12_b_bn')
         .concat(axis=-1)
         .conv(3, 3, 512 / 4, 1, 1, biased=False, relu=False, name='conv12_c')
         .prelu(name='conv12_c_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv12_c_bn'))

        (self.feed('conv11_bn',
                   'conv12_a_bn',
                   'conv12_b_bn',
                   'conv12_c_bn')
         .concat(axis=-1)
         .conv(3, 3, 512 / 4, 1, 1, biased=False, relu=False, name='conv12_d')
         .prelu(name='conv12_d_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv12_d_bn'))

        (self.feed('conv12_a_bn',
                   'conv12_b_bn',
                   'conv12_c_bn',
                   'conv12_d_bn')
         .concat(name='conv12_bn', axis=-1))

        # 20x20x512
        (self.feed('conv12_bn')
         .conv(3, 3, 512 / 4, 1, 1, biased=False, relu=False, name='conv13_a')
         .prelu(name='conv13_a_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv13_a_bn'))

        (self.feed('conv12_bn',
                   'conv13_a_bn')
         .concat(axis=-1)
         .conv(3, 3, 512 / 4, 1, 1, biased=False, relu=False, name='conv13_b')
         .prelu(name='conv13_b_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv13_b_bn'))

        (self.feed('conv12_bn',
                   'conv13_a_bn',
                   'conv13_b_bn')
         .concat(axis=-1)
         .conv(3, 3, 512 / 4, 1, 1, biased=False, relu=False, name='conv13_c')
         .prelu(name='conv13_c_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv13_c_bn'))

        (self.feed('conv12_bn',
                   'conv13_a_bn',
                   'conv13_b_bn',
                   'conv13_c_bn')
         .concat(axis=-1)
         .conv(3, 3, 512 / 4, 1, 1, biased=False, relu=False, name='conv13_d')
         .prelu(name='conv13_d_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv13_d_bn'))

        (self.feed('conv13_a_bn',
                   'conv13_b_bn',
                   'conv13_c_bn',
                   'conv13_d_bn')
         .concat(name='conv13_bn', axis=-1))

        (self.feed('conv13_bn',
                   'conv10_pool')
         .add(name='conv13_sum')
         .resize(40, 40, name='conv13_unpool'))

        (self.feed('conv13_unpool',
                   'conv10_sum')
         .concat(name='conv13_concat', axis=-1))

        # 40x40x512
        (self.feed('conv13_concat')
         .conv(2, 2, 512, 1, 1, biased=False, relu=False, name='conv13_to_14')
         .prelu(name='conv13_to_14_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv13_to_14_bn'))

        # 40x40x512
        (self.feed('conv13_to_14_bn')
         .conv(3, 3, 512 / 4, 1, 1, biased=False, relu=False, name='conv14_a')
         .prelu(name='conv14_a_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv14_a_bn'))

        (self.feed('conv13_to_14_bn',
                   'conv14_a_bn')
         .concat(axis=-1)
         .conv(3, 3, 512 / 4, 1, 1, biased=False, relu=False, name='conv14_b')
         .prelu(name='conv14_b_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv14_b_bn'))

        (self.feed('conv13_to_14_bn',
                   'conv14_a_bn',
                   'conv14_b_bn')
         .concat(axis=-1)
         .conv(3, 3, 512 / 4, 1, 1, biased=False, relu=False, name='conv14_c')
         .prelu(name='conv14_c_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv14_c_bn'))

        (self.feed('conv13_to_14_bn',
                   'conv14_a_bn',
                   'conv14_b_bn',
                   'conv14_c_bn')
         .concat(axis=-1)
         .conv(3, 3, 512 / 4, 1, 1, biased=False, relu=False, name='conv14_d')
         .prelu(name='conv14_d_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv14_d_bn'))

        (self.feed('conv14_a_bn',
                   'conv14_b_bn',
                   'conv14_c_bn',
                   'conv14_d_bn')
         .concat(name='conv14_bn', axis=-1))

        # 40x40x512
        (self.feed('conv14_bn')
         .conv(3, 3, 512 / 4, 1, 1, biased=False, relu=False, name='conv15_a')
         .prelu(name='conv15_a_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv15_a_bn'))

        (self.feed('conv14_bn',
                   'conv15_a_bn')
         .concat(axis=-1)
         .conv(3, 3, 512 / 4, 1, 1, biased=False, relu=False, name='conv15_b')
         .prelu(name='conv15_b_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv15_b_bn'))

        (self.feed('conv14_bn',
                   'conv15_a_bn',
                   'conv15_b_bn')
         .concat(axis=-1)
         .conv(3, 3, 512 / 4, 1, 1, biased=False, relu=False, name='conv15_c')
         .prelu(name='conv15_c_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv15_c_bn'))

        (self.feed('conv14_bn',
                   'conv15_a_bn',
                   'conv15_b_bn',
                   'conv15_c_bn')
         .concat(axis=-1)
         .conv(3, 3, 512 / 4, 1, 1, biased=False, relu=False, name='conv15_d')
         .prelu(name='conv15_d_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv15_d_bn'))

        (self.feed('conv15_a_bn',
                   'conv15_b_bn',
                   'conv15_c_bn',
                   'conv15_d_bn')
         .concat(name='conv15_bn', axis=-1))

        # 40x40x512
        (self.feed('conv15_bn')
         .conv(3, 3, 512 / 4, 1, 1, biased=False, relu=False, name='conv16_a')
         .prelu(name='conv16_a_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv16_a_bn'))

        (self.feed('conv15_bn',
                   'conv16_a_bn')
         .concat(axis=-1)
         .conv(3, 3, 512 / 4, 1, 1, biased=False, relu=False, name='conv16_b')
         .prelu(name='conv16_b_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv16_b_bn'))

        (self.feed('conv15_bn',
                   'conv16_a_bn',
                   'conv16_b_bn')
         .concat(axis=-1)
         .conv(3, 3, 512 / 4, 1, 1, biased=False, relu=False, name='conv16_c')
         .prelu(name='conv16_c_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv16_c_bn'))

        (self.feed('conv15_bn',
                   'conv16_a_bn',
                   'conv16_b_bn',
                   'conv16_c_bn')
         .concat(axis=-1)
         .conv(3, 3, 512 / 4, 1, 1, biased=False, relu=False, name='conv16_d')
         .prelu(name='conv16_d_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv16_d_bn'))

        (self.feed('conv16_a_bn',
                   'conv16_b_bn',
                   'conv16_c_bn',
                   'conv16_d_bn')
         .concat(name='conv16_bn', axis=-1))

        (self.feed('conv16_bn',
                   'conv13_to_14_bn')
         .add(name='conv16_sum')
         .resize(80, 80, name='conv16_unpool'))

        (self.feed('conv16_unpool',
                   'conv7_sum')
         .concat(name='conv16_concat', axis=-1))

        # 80x80x256
        (self.feed('conv16_concat')
         .conv(2, 2, 256, 1, 1, biased=False, relu=False, name='conv17_to_18')
         .prelu(name='conv17_to_18_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv17_to_18_bn'))

        # 80x80x256
        (self.feed('conv17_to_18_bn')
         .conv(3, 3, 256 / 4, 1, 1, biased=False, relu=False, name='conv17_a')
         .prelu(name='conv17_a_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv17_a_bn'))

        (self.feed('conv17_to_18_bn',
                   'conv17_a_bn')
         .concat(axis=-1)
         .conv(3, 3, 256 / 4, 1, 1, biased=False, relu=False, name='conv17_b')
         .prelu(name='conv17_b_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv17_b_bn'))

        (self.feed('conv17_to_18_bn',
                   'conv17_a_bn',
                   'conv17_b_bn')
         .concat(axis=-1)
         .conv(3, 3, 256 / 4, 1, 1, biased=False, relu=False, name='conv17_c')
         .prelu(name='conv17_c_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv17_c_bn'))

        (self.feed('conv17_to_18_bn',
                   'conv17_a_bn',
                   'conv17_b_bn',
                   'conv17_c_bn')
         .concat(axis=-1)
         .conv(3, 3, 256 / 4, 1, 1, biased=False, relu=False, name='conv17_d')
         .prelu(name='conv17_d_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv17_d_bn'))

        (self.feed('conv17_a_bn',
                   'conv17_b_bn',
                   'conv17_c_bn',
                   'conv17_d_bn')
         .concat(name='conv17_bn', axis=-1))

        # 80x80x256
        (self.feed('conv17_bn')
         .conv(3, 3, 256 / 4, 1, 1, biased=False, relu=False, name='conv18_a')
         .prelu(name='conv18_a_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv18_a_bn'))

        (self.feed('conv17_bn',
                   'conv18_a_bn')
         .concat(axis=-1)
         .conv(3, 3, 256 / 4, 1, 1, biased=False, relu=False, name='conv18_b')
         .prelu(name='conv18_b_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv18_b_bn'))

        (self.feed('conv17_bn',
                   'conv18_a_bn',
                   'conv18_b_bn')
         .concat(axis=-1)
         .conv(3, 3, 256 / 4, 1, 1, biased=False, relu=False, name='conv18_c')
         .prelu(name='conv18_c_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv18_c_bn'))

        (self.feed('conv17_bn',
                   'conv18_a_bn',
                   'conv18_b_bn',
                   'conv18_c_bn')
         .concat(axis=-1)
         .conv(3, 3, 256 / 4, 1, 1, biased=False, relu=False, name='conv18_d')
         .prelu(name='conv18_d_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv18_d_bn'))

        (self.feed('conv18_a_bn',
                   'conv18_b_bn',
                   'conv18_c_bn',
                   'conv18_d_bn')
         .concat(name='conv18_bn', axis=-1))

        # 80x80x256
        (self.feed('conv18_bn')
         .conv(3, 3, 256 / 4, 1, 1, biased=False, relu=False, name='conv19_a')
         .prelu(name='conv19_a_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv19_a_bn'))

        (self.feed('conv18_bn',
                   'conv19_a_bn')
         .concat(axis=-1)
         .conv(3, 3, 256 / 4, 1, 1, biased=False, relu=False, name='conv19_b')
         .prelu(name='conv19_b_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv19_b_bn'))

        (self.feed('conv18_bn',
                   'conv19_a_bn',
                   'conv19_b_bn')
         .concat(axis=-1)
         .conv(3, 3, 256 / 4, 1, 1, biased=False, relu=False, name='conv19_c')
         .prelu(name='conv19_c_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv19_c_bn'))

        (self.feed('conv18_bn',
                   'conv19_a_bn',
                   'conv19_b_bn',
                   'conv19_c_bn')
         .concat(axis=-1)
         .conv(3, 3, 256 / 4, 1, 1, biased=False, relu=False, name='conv19_d')
         .prelu(name='conv19_d_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv19_d_bn'))

        (self.feed('conv19_a_bn',
                   'conv19_b_bn',
                   'conv19_c_bn',
                   'conv19_d_bn')
         .concat(name='conv19_bn', axis=-1))

        (self.feed('conv19_bn',
                   'conv17_to_18_bn')
         .add(name='conv19_sum')
         .resize(160, 160, name='conv19_unpool'))

        (self.feed('conv19_unpool',
                   'conv4_sum')
         .concat(name='conv19_concat', axis=-1))

        # 160x160x128
        (self.feed('conv19_concat')
         .conv(2, 2, 128, 1, 1, biased=False, relu=False, name='conv19_to_20')
         .prelu(name='conv19_to_20_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv19_to_20_bn'))

        # 160x160x128
        (self.feed('conv19_to_20_bn')
         .conv(3, 3, 128 / 4, 1, 1, biased=False, relu=False, name='conv20_a')
         .prelu(name='conv20_a_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv20_a_bn'))

        (self.feed('conv19_to_20_bn',
                   'conv20_a_bn')
         .concat(axis=-1)
         .conv(3, 3, 128 / 4, 1, 1, biased=False, relu=False, name='conv20_b')
         .prelu(name='conv20_b_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv20_b_bn'))

        (self.feed('conv19_to_20_bn',
                   'conv20_a_bn',
                   'conv20_b_bn')
         .concat(axis=-1)
         .conv(3, 3, 128 / 4, 1, 1, biased=False, relu=False, name='conv20_c')
         .prelu(name='conv20_c_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv20_c_bn'))

        (self.feed('conv19_to_20_bn',
                   'conv20_a_bn',
                   'conv20_b_bn',
                   'conv20_c_bn')
         .concat(axis=-1)
         .conv(3, 3, 128 / 4, 1, 1, biased=False, relu=False, name='conv20_d')
         .prelu(name='conv20_d_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv20_d_bn'))

        (self.feed('conv20_a_bn',
                   'conv20_b_bn',
                   'conv20_c_bn',
                   'conv20_d_bn')
         .concat(name='conv20_bn', axis=-1))

        # 160x160x128
        (self.feed('conv20_bn')
         .conv(3, 3, 128 / 4, 1, 1, biased=False, relu=False, name='conv21_a')
         .prelu(name='conv21_a_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv21_a_bn'))

        (self.feed('conv20_bn',
                   'conv21_a_bn')
         .concat(axis=-1)
         .conv(3, 3, 128 / 4, 1, 1, biased=False, relu=False, name='conv21_b')
         .prelu(name='conv21_b_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv21_b_bn'))

        (self.feed('conv20_bn',
                   'conv21_a_bn',
                   'conv21_b_bn')
         .concat(axis=-1)
         .conv(3, 3, 128 / 4, 1, 1, biased=False, relu=False, name='conv21_c')
         .prelu(name='conv21_c_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv21_c_bn'))

        (self.feed('conv20_bn',
                   'conv21_a_bn',
                   'conv21_b_bn',
                   'conv21_c_bn')
         .concat(axis=-1)
         .conv(3, 3, 128 / 4, 1, 1, biased=False, relu=False, name='conv21_d')
         .prelu(name='conv21_d_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv21_d_bn'))

        (self.feed('conv21_a_bn',
                   'conv21_b_bn',
                   'conv21_c_bn',
                   'conv21_d_bn')
         .concat(name='conv21_bn', axis=-1))

        (self.feed('conv21_bn',
                   'conv19_to_20_bn')
         .add(name='conv21_sum')
         .resize(320, 320, name='conv21_unpool'))

        (self.feed('conv21_unpool',
                   'conv2_bn')
         .concat(name='conv21_concat', axis=-1))

        # 320x320x64
        (self.feed('conv21_concat')
         .conv(2, 2, 64, 1, 1, biased=False, relu=False, name='conv21_to_22')
         .prelu(name='conv21_to_22_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv21_to_22_bn'))

        # 320x320x64
        (self.feed('conv21_to_22_bn')
         .conv(3, 3, 64 / 4, 1, 1, biased=False, relu=False, name='conv22_a')
         .prelu(name='conv22_a_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv22_a_bn'))

        (self.feed('conv21_to_22_bn',
                   'conv22_a_bn')
         .concat(axis=-1)
         .conv(3, 3, 64 / 4, 1, 1, biased=False, relu=False, name='conv22_b')
         .prelu(name='conv22_b_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv22_b_bn'))

        (self.feed('conv21_to_22_bn',
                   'conv22_a_bn',
                   'conv22_b_bn')
         .concat(axis=-1)
         .conv(3, 3, 64 / 4, 1, 1, biased=False, relu=False, name='conv22_c')
         .prelu(name='conv22_c_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv22_c_bn'))

        (self.feed('conv21_to_22_bn',
                   'conv22_a_bn',
                   'conv22_b_bn',
                   'conv22_c_bn')
         .concat(axis=-1)
         .conv(3, 3, 64 / 4, 1, 1, biased=False, relu=False, name='conv22_d')
         .prelu(name='conv22_d_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv22_d_bn'))

        (self.feed('conv22_a_bn',
                   'conv22_b_bn',
                   'conv22_c_bn',
                   'conv22_d_bn')
         .concat(name='conv22_bn', axis=-1))

        # 320x320x64
        (self.feed('conv22_bn')
         .conv(3, 3, 64 / 4, 1, 1, biased=False, relu=False, name='conv23_a')
         .prelu(name='conv23_a_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv23_a_bn'))

        (self.feed('conv22_bn',
                   'conv23_a_bn')
         .concat(axis=-1)
         .conv(3, 3, 64 / 4, 1, 1, biased=False, relu=False, name='conv23_b')
         .prelu(name='conv23_b_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv23_b_bn'))

        (self.feed('conv22_bn',
                   'conv23_a_bn',
                   'conv23_b_bn')
         .concat(axis=-1)
         .conv(3, 3, 64 / 4, 1, 1, biased=False, relu=False, name='conv23_c')
         .prelu(name='conv23_c_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv23_c_bn'))

        (self.feed('conv22_bn',
                   'conv23_a_bn',
                   'conv23_b_bn',
                   'conv23_c_bn')
         .concat(axis=-1)
         .conv(3, 3, 64 / 4, 1, 1, biased=False, relu=False, name='conv23_d')
         .prelu(name='conv23_d_prelu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv23_d_bn'))

        (self.feed('conv23_a_bn',
                   'conv23_b_bn',
                   'conv23_c_bn',
                   'conv23_d_bn')
         .concat(name='conv23_bn', axis=-1))

        (self.feed('conv23_bn',
                   'conv21_to_22_bn')
         .add(name='conv23_sum'))

        # 320x320xnum_classes
        (self.feed('conv23_sum')
         .conv(3, 3, num_classes, 1, 1, biased=False, relu=False, name='conv24'))
