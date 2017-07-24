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
        (self.feed('data')
         .conv(7, 7, 64, 2, 2, biased=False, relu=False, name='conv1')
         .selu(name='selu_conv1')
         .max_pool(3, 3, 2, 2, name='pool1')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2a_branch1'))

        (self.feed('pool1')
         .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2a_branch2a')
         .selu(name='selu2a_branch2a')
         .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2a_branch2b')
         .selu(name='selu2a_branch2b')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2a_branch2c'))

        (self.feed('selu2a_branch2b')
         .conv(1, 1, 16, 1, 1, biased=False, relu=False, name='res2a_branch2c_dense')
         .selu(name='selu2a_branch2c_dense'))

        (self.feed('res2a_branch1',
                   'res2a_branch2c')
         .add(name='res2a')
         .selu(name='res2a_selu'))

        (self.feed('res2a_selu',
                   'selu2a_branch2c_dense')
         .concat(axis=-1, name='2a_combine_dpn')
         .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2b_branch2a')
         .selu(name='selu2b_branch2a')
         .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2b_branch2b')
         .selu(name='selu2b_branch2b')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2b_branch2c'))

        (self.feed('selu2b_branch2b')
         .conv(1, 1, 16, 1, 1, biased=False, relu=False, name='res2b_branch2c_dense')
         .selu(name='selu2b_branch2c_dense'))

        (self.feed('res2a_selu',
                   'res2b_branch2c')
         .add(name='res2b')
         .selu(name='res2b_selu'))

        (self.feed('res2b_selu',
                   'selu2a_branch2c_dense',
                   'selu2b_branch2c_dense')
         .concat(axis=-1, name='2b_combine_dpn')
         .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2c_branch2a')
         .selu(name='selu2c_branch2a')
         .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2c_branch2b')
         .selu(name='selu2c_branch2b')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2c_branch2c'))

        (self.feed('selu2c_branch2b')
         .conv(1, 1, 16, 1, 1, biased=False, relu=False, name='res2c_branch2c_dense')
         .selu(name='selu2c_branch2c_dense'))

        (self.feed('res2b_selu',
                   'res2c_branch2c')
         .add(name='res2c')
         .selu(name='res2c_selu'))

        (self.feed('res2c_selu',
                   'selu2a_branch2c_dense',
                   'selu2b_branch2c_dense',
                   'selu2c_branch2c_dense')
         .concat(axis=-1, name='2c_combine_dpn')
         .conv(1, 1, 512, 2, 2, biased=False, relu=False, name='res3a_branch1'))

        (self.feed('2c_combine_dpn')
         .conv(1, 1, 128, 2, 2, biased=False, relu=False, name='res3a_branch2a')
         .selu(name='selu3a_branch2a')
         .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3a_branch2b')
         .selu(name='selu3a_branch2b')
         .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3a_branch2c'))

        (self.feed('selu3a_branch2b')
         .conv(1, 1, 32, 1, 1, biased=False, relu=False, name='res3a_branch2c_dense')
         .selu(name='selu3a_branch2c_dense'))

        (self.feed('res3a_branch1',
                   'res3a_branch2c')
         .add(name='res3a')
         .selu(name='res3a_selu'))

        (self.feed('selu2a_branch2c_dense',
                   'selu2b_branch2c_dense',
                   'selu2c_branch2c_dense')
         .concat(axis=-1, name='2c_combine_dpn_only')
         .conv(1, 1, 32 * 3, 2, 2, biased=False, relu=False, name='2c_combine_dpn_downscale'))

        (self.feed('res3a_selu',
                   '2c_combine_dpn_downscale',
                   'selu3a_branch2c_dense')
         .concat(axis=-1, name='3a_combine_dpn')
         .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3b1_branch2a')
         .selu(name='selu3b1_branch2a')
         .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3b1_branch2b')
         .selu(name='selu3b1_branch2b')
         .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3b1_branch2c'))

        (self.feed('selu3b1_branch2b')
         .conv(1, 1, 32, 1, 1, biased=False, relu=False, name='res3b1_branch2c_dense')
         .selu(name='selu3b1_branch2c_dense'))

        (self.feed('res3a_selu',
                   'res3b1_branch2c')
         .add(name='res3b1')
         .selu(name='res3b1_selu'))

        (self.feed('res3b1_selu',
                   '2c_combine_dpn_downscale',
                   'selu3a_branch2c_dense',
                   'selu3b1_branch2c_dense')
         .concat(axis=-1, name='3b1_combine_dpn')
         .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3b2_branch2a')
         .selu(name='selu3b2_branch2a')
         .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3b2_branch2b')
         .selu(name='selu3b2_branch2b')
         .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3b2_branch2c'))

        (self.feed('selu3b2_branch2b')
         .conv(1, 1, 32, 1, 1, biased=False, relu=False, name='res3b2_branch2c_dense')
         .selu(name='selu3b2_branch2c_dense'))

        (self.feed('res3b1_selu',
                   'res3b2_branch2c')
         .add(name='res3b2')
         .selu(name='res3b2_selu'))

        (self.feed('res3b1_selu',
                   '2c_combine_dpn_downscale',
                   'selu3a_branch2c_dense',
                   'selu3b1_branch2c_dense',
                   'selu3b2_branch2c_dense')
         .concat(axis=-1, name='3b2_combine_dpn')
         .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3b3_branch2a')
         .selu(name='selu3b3_branch2a')
         .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3b3_branch2b')
         .selu(name='selu3b3_branch2b')
         .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3b3_branch2c'))

        (self.feed('selu3b3_branch2b')
         .conv(1, 1, 32, 1, 1, biased=False, relu=False, name='res3b3_branch2c_dense')
         .selu(name='selu3b3_branch2c_dense'))

        (self.feed('res3b2_selu',
                   'res3b3_branch2c')
         .add(name='res3b3')
         .selu(name='res3b3_selu'))

        (self.feed('res3b3_selu',
                   '2c_combine_dpn_downscale',
                   'selu3a_branch2c_dense',
                   'selu3b1_branch2c_dense',
                   'selu3b2_branch2c_dense',
                   'selu3b3_branch2c_dense')
         .concat(axis=-1, name='3b3_combine_dpn')
         .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4a_branch1'))

        (self.feed('3b3_combine_dpn')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4a_branch2a')
         .selu(name='selu4a_branch2a')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4a_branch2b')
         .selu(name='selu4a_branch2b')
         .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4a_branch2c'))

        (self.feed('selu4a_branch2b')
         .conv(1, 1, 24, 1, 1, biased=False, relu=False, name='res4a_branch2c_dense')
         .selu(name='selu4a_branch2c_dense'))

        (self.feed('res4a_branch1',
                   'res4a_branch2c')
         .add(name='res4a')
         .selu(name='res4a_selu'))

        (self.feed('2c_combine_dpn_downscale',
                   'selu3a_branch2c_dense',
                   'selu3b1_branch2c_dense',
                   'selu3b2_branch2c_dense',
                   'selu3b3_branch2c_dense')
         .concat(axis=-1, name='3b3_combine_dpn_only')
         .conv(1, 1, 24 * 4, 1, 1, biased=False, relu=False, name='3b3_combine_dpn_downscale'))

        (self.feed('res4a_selu',
                   '3b3_combine_dpn_downscale',
                   'selu4a_branch2c_dense')
         .concat(axis=-1, name='4a_combine_dpn')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b1_branch2a')
         .selu(name='selu4b1_branch2a')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b1_branch2b')
         .selu(name='selu4b1_branch2b')
         .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b1_branch2c'))

        (self.feed('selu4b1_branch2b')
         .conv(1, 1, 24, 1, 1, biased=False, relu=False, name='res4b1_branch2c_dense')
         .selu(name='selu4b1_branch2c_dense'))

        (self.feed('res4a_selu',
                   'res4b1_branch2c')
         .add(name='res4b1')
         .selu(name='res4b1_selu'))

        (self.feed('res4b1_selu',
                   '3b3_combine_dpn_downscale',
                   'selu4a_branch2c_dense',
                   'selu4b1_branch2c_dense')
         .concat(axis=-1, name='4b1_combine_dpn')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b2_branch2a')
         .selu(name='selu4b2_branch2a')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b2_branch2b')
         .selu(name='selu4b2_branch2b')
         .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b2_branch2c'))

        (self.feed('selu4b2_branch2b')
         .conv(1, 1, 24, 1, 1, biased=False, relu=False, name='res4b2_branch2c_dense')
         .selu(name='selu4b2_branch2c_dense'))

        (self.feed('res4b1_selu',
                   'res4b2_branch2c')
         .add(name='res4b2')
         .selu(name='res4b2_selu'))

        (self.feed('res4b2_selu',
                   '3b3_combine_dpn_downscale',
                   'selu4a_branch2c_dense',
                   'selu4b1_branch2c_dense',
                   'selu4b2_branch2c_dense')
         .concat(axis=-1, name='4b2_combine_dpn')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b3_branch2a')
         .selu(name='selu4b3_branch2a')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b3_branch2b')
         .selu(name='selu4b3_branch2b')
         .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b3_branch2c'))

        (self.feed('selu4b3_branch2b')
         .conv(1, 1, 24, 1, 1, biased=False, relu=False, name='res4b3_branch2c_dense')
         .selu(name='selu4b3_branch2c_dense'))

        (self.feed('res4b2_selu',
                   'res4b3_branch2c')
         .add(name='res4b3')
         .selu(name='res4b3_selu'))

        (self.feed('res4b3_selu',
                   '3b3_combine_dpn_downscale',
                   'selu4a_branch2c_dense',
                   'selu4b1_branch2c_dense',
                   'selu4b2_branch2c_dense',
                   'selu4b3_branch2c_dense')
         .concat(axis=-1, name='4b3_combine_dpn')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b4_branch2a')
         .selu(name='selu4b4_branch2a')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b4_branch2b')
         .selu(name='selu4b4_branch2b')
         .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b4_branch2c'))

        (self.feed('selu4b4_branch2b')
         .conv(1, 1, 24, 1, 1, biased=False, relu=False, name='res4b4_branch2c_dense')
         .selu(name='selu4b4_branch2c_dense'))

        (self.feed('res4b3_selu',
                   'res4b4_branch2c')
         .add(name='res4b4')
         .selu(name='res4b4_selu'))

        (self.feed('res4b4_selu',
                   '3b3_combine_dpn_downscale',
                   'selu4a_branch2c_dense',
                   'selu4b1_branch2c_dense',
                   'selu4b2_branch2c_dense',
                   'selu4b3_branch2c_dense',
                   'selu4b4_branch2c_dense')
         .concat(axis=-1, name='4b4_combine_dpn')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b5_branch2a')
         .selu(name='selu4b5_branch2a')
         .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b5_branch2b')
         .selu(name='selu4b5_branch2b')
         .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b5_branch2c'))

        (self.feed('selu4b5_branch2b')
         .conv(1, 1, 24, 1, 1, biased=False, relu=False, name='res4b5_branch2c_dense')
         .selu(name='selu4b5_branch2c_dense'))

        (self.feed('res4b4_selu',
                   'res4b5_branch2c')
         .add(name='res4b5')
         .selu(name='res4b5_selu'))

        (self.feed('res4b5_selu',
                   '3b3_combine_dpn_downscale',
                   'selu4a_branch2c_dense',
                   'selu4b1_branch2c_dense',
                   'selu4b2_branch2c_dense',
                   'selu4b3_branch2c_dense',
                   'selu4b4_branch2c_dense',
                   'selu4b5_branch2c_dense')
         .concat(axis=-1, name='4b5_combine_dpn')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b6_branch2a')
         .selu(name='selu4b6_branch2a')
         .conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4b6_branch2b')
         .selu(name='selu4b6_branch2b')
         .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b6_branch2c'))

        (self.feed('selu4b6_branch2b')
         .conv(1, 1, 24, 1, 1, biased=False, relu=False, name='res4b6_branch2c_dense')
         .selu(name='selu4b6_branch2c_dense'))

        (self.feed('res4b5_selu',
                   'res4b6_branch2c')
         .add(name='res4b6')
         .selu(name='res4b6_selu'))

        (self.feed('res4b6_selu',
                   '3b3_combine_dpn_downscale',
                   'selu4a_branch2c_dense',
                   'selu4b1_branch2c_dense',
                   'selu4b2_branch2c_dense',
                   'selu4b3_branch2c_dense',
                   'selu4b4_branch2c_dense',
                   'selu4b5_branch2c_dense',
                   'selu4b6_branch2c_dense')
         .concat(axis=-1, name='4b6_combine_dpn')
         .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5a_branch1'))

        (self.feed('4b6_combine_dpn')
         .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res5a_branch2a')
         .selu(name='selu5a_branch2a')
         .atrous_conv(3, 3, 512, 4, padding='SAME', biased=False, relu=False, name='res5a_branch2b')
         .selu(name='selu5a_branch2b')
         .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5a_branch2c'))

        (self.feed('selu5a_branch2b')
         .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res5a_branch2c_dense')
         .selu(name='selu5a_branch2c_dense'))

        (self.feed('res5a_branch1',
                   'res5a_branch2c')
         .add(name='res5a')
         .selu(name='res5a_selu'))

        (self.feed('3b3_combine_dpn_downscale',
                   'selu4a_branch2c_dense',
                   'selu4b1_branch2c_dense',
                   'selu4b2_branch2c_dense',
                   'selu4b3_branch2c_dense',
                   'selu4b4_branch2c_dense',
                   'selu4b5_branch2c_dense',
                   'selu4b6_branch2c_dense')
         .concat(axis=-1, name='4b6_combine_dpn_only')
         .conv(1, 1, 128 * 7, 1, 1, biased=False, relu=False, name='4b6_combine_dpn_downscale'))

        (self.feed('res5a_selu',
                   '4b6_combine_dpn_downscale',
                   'selu5a_branch2c_dense')
         .concat(axis=-1, name='5a_combine_dpn')
         .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res5b_branch2a')
         .selu(name='selu5b_branch2a')
         .atrous_conv(3, 3, 512, 4, padding='SAME', biased=False, relu=False, name='res5b_branch2b')
         .selu(name='selu5b_branch2b')
         .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5b_branch2c'))

        (self.feed('selu5b_branch2b')
         .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res5b_branch2c_dense')
         .selu(name='selu5b_branch2c_dense'))

        (self.feed('res5a_selu',
                   'res5b_branch2c')
         .add(name='res5b')
         .selu(name='res5b_selu'))

        (self.feed('res5b_selu',
                   '4b6_combine_dpn_downscale',
                   'selu5a_branch2c_dense',
                   'selu5b_branch2c_dense')
         .concat(axis=-1, name='5b_combine_dpn')
         .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res5c_branch2a')
         .selu(name='selu5c_branch2a')
         .atrous_conv(3, 3, 512, 4, padding='SAME', biased=False, relu=False, name='res5c_branch2b')
         .selu(name='selu5c_branch2b')
         .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5c_branch2c'))

        (self.feed('selu5c_branch2b')
         .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res5c_branch2c_dense')
         .selu(name='selu5c_branch2c_dense'))

        (self.feed('res5b_selu',
                   'res5c_branch2c')
         .add(name='res5c')
         .dropout_selu(rate=0.05, training=is_training, name='res5c_selu'))

        (self.feed('res5c_selu',
                   '4b6_combine_dpn_downscale',
                   'selu5a_branch2c_dense',
                   'selu5b_branch2c_dense',
                   'selu5c_branch2c_dense')
         .concat(axis=-1, name='5c_combine_dpn')
         .atrous_conv(3, 3, num_classes, 6, padding='SAME', relu=False, name='fc1_voc12_c0'))

        (self.feed('res5c_selu')
         .atrous_conv(3, 3, num_classes, 12, padding='SAME', relu=False, name='fc1_voc12_c1'))

        (self.feed('res5c_selu')
         .atrous_conv(3, 3, num_classes, 18, padding='SAME', relu=False, name='fc1_voc12_c2'))

        (self.feed('res5c_selu')
         .atrous_conv(3, 3, num_classes, 24, padding='SAME', relu=False, name='fc1_voc12_c3'))

        (self.feed('fc1_voc12_c0',
                   'fc1_voc12_c1',
                   'fc1_voc12_c2',
                   'fc1_voc12_c3')
         .add(name='fc1_voc12'))
