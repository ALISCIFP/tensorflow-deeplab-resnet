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
         .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='conv1')
         .relu(name='conv1_relu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv1_bn'))

        # 320x320x64
        (self.feed('conv1_bn')
         .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='conv2')
         .relu(name='conv2_relu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv2_bn'))

        # 320x320x128
        (self.feed('conv2_bn')
         .conv(2, 2, 128, 1, 1, biased=False, relu=False, name='conv2_to_3')
         .relu(name='conv2_to_3_relu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv2_to_3_bn')
         .max_pool(2, 2, 2, 2, name='conv2_pool'))

        # 160x160x128
        (self.feed('conv2_pool')
         .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='conv3')
         .relu(name='conv3_relu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv3_bn'))

        # 160x160x128
        (self.feed('conv3_bn')
         .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='conv4')
         .relu(name='conv4_relu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv4_bn'))

        (self.feed('conv2_pool',
                   'conv4_bn')
         .add(name='conv4_sum'))

        # 160x160x256
        (self.feed('conv4_sum')
         .conv(2, 2, 256, 1, 1, biased=False, relu=False, name='conv4_to_5')
         .relu(name='conv4_to_5_relu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv4_to_5_bn')
         .max_pool(2, 2, 2, 2, name='conv4_pool'))

        # 80x80x256
        (self.feed('conv4_pool')
         .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='conv5')
         .relu(name='conv5_relu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv5_bn'))

        # 80x80x256
        (self.feed('conv5_bn')
         .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='conv6')
         .relu(name='conv6_relu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv6_bn'))

        # 80x80x256
        (self.feed('conv6_bn')
         .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='conv7')
         .relu(name='conv7_relu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv7_bn'))

        (self.feed('conv4_pool',
                   'conv7_bn')
         .add(name='conv7_sum'))

        # 80x80x512
        (self.feed('conv7_sum')
         .conv(2, 2, 512, 1, 1, biased=False, relu=False, name='conv7_to_8')
         .relu(name='conv7_to_8_relu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv7_to_8_bn')
         .max_pool(2, 2, 2, 2, name='conv7_pool'))

        # 40x40x512
        (self.feed('conv7_pool')
         .conv(3, 3, 512, 1, 1, biased=False, relu=False, name='conv8')
         .relu(name='conv8_relu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv8_bn'))

        # 40x40x512
        (self.feed('conv8_bn')
         .conv(3, 3, 512, 1, 1, biased=False, relu=False, name='conv9')
         .relu(name='conv9_relu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv9_bn'))

        # 40x40x512
        (self.feed('conv9_bn')
         .conv(3, 3, 512, 1, 1, biased=False, relu=False, name='conv10')
         .relu(name='conv10_relu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv10_bn'))

        (self.feed('conv10_bn',
                   'conv7_pool')
         .add(name='conv10_sum')
         .max_pool(2, 2, 2, 2, name='conv10_pool'))

        # 20x20x512
        (self.feed('conv10_pool')
         .conv(3, 3, 512, 1, 1, biased=False, relu=False, name='conv11')
         .relu(name='conv11_relu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv11_bn'))

        # 20x20x512
        (self.feed('conv11_bn')
         .conv(3, 3, 512, 1, 1, biased=False, relu=False, name='conv12')
         .relu(name='conv12_relu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv12_bn'))

        # 20x20x512
        (self.feed('conv12_bn')
         .conv(3, 3, 512, 1, 1, biased=False, relu=False, name='conv13')
         .relu(name='conv13_relu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv13_bn'))

        (self.feed('conv13_bn',
                   'conv10_pool')
         .add(name='conv13_sum')
         .resize_dynamic(2, 2, name='conv13_unpool'))

        (self.feed('conv13_unpool',
                   'conv10_sum')
         .concat(name='conv13_concat', axis=-1))

        # 40x40x512
        (self.feed('conv13_concat')
         .conv(2, 2, 512, 1, 1, biased=False, relu=False, name='conv13_to_14')
         .relu(name='conv13_to_14_relu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv13_to_14_bn'))

        # 40x40x512
        (self.feed('conv13_to_14_bn')
         .conv(3, 3, 512, 1, 1, biased=False, relu=False, name='conv14')
         .relu(name='conv14_relu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv14_bn'))

        # 40x40x512
        (self.feed('conv14_bn')
         .conv(3, 3, 512, 1, 1, biased=False, relu=False, name='conv15')
         .relu(name='conv15_relu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv15_bn'))

        # 40x40x512
        (self.feed('conv15_bn')
         .conv(3, 3, 512, 1, 1, biased=False, relu=False, name='conv16')
         .relu(name='conv16_relu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv16_bn'))

        (self.feed('conv16_bn',
                   'conv13_to_14_bn')
         .add(name='conv16_sum')
         .resize_dynamic(2, 2, name='conv16_unpool'))

        (self.feed('conv16_unpool',
                   'conv7_sum')
         .concat(name='conv16_concat', axis=-1))

        # 80x80x256
        (self.feed('conv16_concat')
         .conv(2, 2, 256, 1, 1, biased=False, relu=False, name='conv17_to_18')
         .relu(name='conv17_to_18_relu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv17_to_18_bn'))

        # 80x80x256
        (self.feed('conv17_to_18_bn')
         .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='conv17')
         .relu(name='conv17_relu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv17_bn'))

        # 80x80x256
        (self.feed('conv17_bn')
         .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='conv18')
         .relu(name='conv18_relu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv18_bn'))

        # 80x80x256
        (self.feed('conv18_bn')
         .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='conv19')
         .relu(name='conv19_relu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv19_bn'))

        (self.feed('conv19_bn',
                   'conv17_to_18_bn')
         .add(name='conv19_sum')
         .resize_dynamic(2, 2, name='conv19_unpool'))

        (self.feed('conv19_unpool',
                   'conv4_sum')
         .concat(name='conv19_concat', axis=-1))

        # 160x160x128
        (self.feed('conv19_concat')
         .conv(2, 2, 128, 1, 1, biased=False, relu=False, name='conv19_to_20')
         .relu(name='conv19_to_20_relu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv19_to_20_bn'))

        # 160x160x128
        (self.feed('conv19_to_20_bn')
         .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='conv20')
         .relu(name='conv20_relu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv20_bn'))

        # 160x160x128
        (self.feed('conv20_bn')
         .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='conv21')
         .relu(name='conv21_relu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv21_bn'))

        (self.feed('conv21_bn',
                   'conv19_to_20_bn')
         .add(name='conv21_sum')
         .resize_dynamic(2, 2, name='conv21_unpool'))

        (self.feed('conv21_unpool',
                   'conv2_bn')
         .concat(name='conv21_concat', axis=-1))

        # 320x320x64
        (self.feed('conv21_concat')
         .conv(2, 2, 64, 1, 1, biased=False, relu=False, name='conv21_to_22')
         .relu(name='conv21_to_22_relu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv21_to_22_bn'))

        # 320x320x64
        (self.feed('conv21_to_22_bn')
         .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='conv22')
         .relu(name='conv22_relu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv22_bn'))

        # 320x320x64
        (self.feed('conv22_bn')
         .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='conv23')
         .relu(name='conv23_relu')
         .batch_normalization(is_training=is_training, activation_fn=None, name='conv23_bn'))

        (self.feed('conv23_bn',
                   'conv21_to_22_bn')
         .add(name='conv23_sum'))

        # 320x320xnum_classes
        (self.feed('conv23_sum')
         .conv(3, 3, num_classes, 1, 1, biased=False, relu=False, name='conv24'))
