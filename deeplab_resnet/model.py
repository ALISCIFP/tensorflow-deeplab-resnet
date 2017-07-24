# Converted to TensorFlow .caffemodel
# with the DeepLab-ResNet configuration.
# The batch normalisation layer is provided by
# the slim library (https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim).

from kaffe.tensorflow import Network
import tensorflow as tf

class DeepLabResNetModel(Network):
    def setup(self,is_training, num_classes):
        '''Network definition.
        
        Args:
          is_training: whether to update the running mean and variance of the batch normalisation layer.
                       If the batch size is small, it is better to keep the running mean and variance of 
                       the-pretrained model frozen.
          num_classes: number of classes to predict (including background).
        '''
        (self.feed('data')
             .conv(7, 7, 64, 2, 2, biased=False, relu=False, name='conv1')
             .selu(name='selu1')
             .max_pool(3, 3, 2, 2, name='pool1')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2a_branch1'))

        (self.feed('pool1')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2a_branch2a')
             .selu(name='selu2a_branch2a')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2a_branch2b')
             .selu(name='selu2a_branch2b')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2a_branch2c'))

        (self.feed('res2a_branch1',
                   'res2a_branch2c')
             .add(name='res2a')
             .selu(name='res2a_selu')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2b_branch2a')
             .selu(name='selu2b_branch2a')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2b_branch2b')
             .selu(name='selu2b_branch2b')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2b_branch2c'))

        (self.feed('res2a_selu',
                   'res2b_branch2c')
             .add(name='res2b')
             .selu(name='res2b_selu')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2c_branch2a')
             .selu(name='selu2c_branch2a')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2c_branch2b')
             .selu(name='selu2c_branch2b')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2c_branch2c'))

        (self.feed('res2b_selu',
                   'res2c_branch2c')
             .add(name='res2c')
             .selu(name='res2c_selu')
             .conv(1, 1, 512, 2, 2, biased=False, relu=False, name='res3a_branch1'))

        (self.feed('res2c_selu')
             .conv(1, 1, 128, 2, 2, biased=False, relu=False, name='res3a_branch2a')
             .selu(name='selu3a_branch2a')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3a_branch2b')
             .selu(name='selu3a_branch2b')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3a_branch2c'))

        (self.feed('res3a_branch1',
                   'res3a_branch2c')
             .add(name='res3a')
             .selu(name='res3a_selu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3b1_branch2a')
             .selu(name='selu3b1_branch2a')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3b1_branch2b')
             .selu(name='selu3b1_branch2b')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3b1_branch2c'))

        (self.feed('res3a_selu',
                   'res3b1_branch2c')
             .add(name='res3b1')
             .selu(name='res3b1_selu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3b2_branch2a')
             .selu(name='selu3b2_branch2a')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3b2_branch2b')
             .selu(name='selu3b2_branch2b')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3b2_branch2c'))

        (self.feed('res3b1_selu',
                   'res3b2_branch2c')
             .add(name='res3b2')
             .selu(name='res3b2_selu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3b3_branch2a')
             .selu(name='selu3b3_branch2a')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3b3_branch2b')
             .selu(name='selu3b3_branch2b')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3b3_branch2c'))

        (self.feed('res3b2_selu',
                   'res3b3_branch2c')
             .add(name='res3b3')
             .selu(name='res3b3_selu')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4a_branch1'))

        (self.feed('res3b3_selu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4a_branch2a')
             .selu(name='selu4a_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4a_branch2b')
             .selu(name='selu4a_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4a_branch2c'))

        (self.feed('res4a_branch1',
                   'res4a_branch2c')
             .add(name='res4a')
             .selu(name='res4a_selu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b1_branch2a')
             .selu(name='selu4b1_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b1_branch2b')
             .selu(name='selu4b1_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b1_branch2c'))

        (self.feed('res4a_selu',
                   'res4b1_branch2c')
             .add(name='res4b1')
             .selu(name='res4b1_selu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b2_branch2a')
             .selu(name='selu4b2_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b2_branch2b')
             .selu(name='selu4b2_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b2_branch2c'))

        (self.feed('res4b1_selu',
                   'res4b2_branch2c')
             .add(name='res4b2')
             .selu(name='res4b2_selu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b3_branch2a')
             .selu(name='selu4b3_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b3_branch2b')
             .selu(name='selu4b3_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b3_branch2c'))

        (self.feed('res4b2_selu',
                   'res4b3_branch2c')
             .add(name='res4b3')
             .selu(name='res4b3_selu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b4_branch2a')
             .selu(name='selu4b4_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b4_branch2b')
             .selu(name='selu4b4_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b4_branch2c'))

        (self.feed('res4b3_selu',
                   'res4b4_branch2c')
             .add(name='res4b4')
             .selu(name='res4b4_selu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b5_branch2a')
             .selu(name='selu4b5_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b5_branch2b')
             .selu(name='selu4b5_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b5_branch2c'))

        (self.feed('res4b4_selu',
                   'res4b5_branch2c')
             .add(name='res4b5')
             .selu(name='res4b5_selu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b6_branch2a')
             .selu(name='selu4b6_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b6_branch2b')
             .selu(name='selu4b6_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b6_branch2c'))

        (self.feed('res4b5_selu',
                   'res4b6_branch2c')
             .add(name='res4b6')
             .selu(name='res4b6_selu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b7_branch2a')
             .selu(name='selu4b7_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b7_branch2b')
             .selu(name='selu4b7_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b7_branch2c'))

        (self.feed('res4b6_selu',
                   'res4b7_branch2c')
             .add(name='res4b7')
             .selu(name='res4b7_selu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b8_branch2a')
             .selu(name='selu4b8_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b8_branch2b')
             .selu(name='selu4b8_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b8_branch2c'))

        (self.feed('res4b7_selu',
                   'res4b8_branch2c')
             .add(name='res4b8')
             .selu(name='res4b8_selu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b9_branch2a')
             .selu(name='selu4b9_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b9_branch2b')
             .selu(name='selu4b9_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b9_branch2c'))

        (self.feed('res4b8_selu',
                   'res4b9_branch2c')
             .add(name='res4b9')
             .selu(name='res4b9_selu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b10_branch2a')
             .selu(name='selu4b10_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b10_branch2b')
             .selu(name='selu4b10_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b10_branch2c'))

        (self.feed('res4b9_selu',
                   'res4b10_branch2c')
             .add(name='res4b10')
             .selu(name='res4b10_selu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b11_branch2a')
             .selu(name='selu4b11_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b11_branch2b')
             .selu(name='selu4b11_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b11_branch2c'))

        (self.feed('res4b10_selu',
                   'res4b11_branch2c')
             .add(name='res4b11')
             .selu(name='res4b11_selu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b12_branch2a')
             .selu(name='selu4b12_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b12_branch2b')
             .selu(name='selu4b12_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b12_branch2c'))

        (self.feed('res4b11_selu',
                   'res4b12_branch2c')
             .add(name='res4b12')
             .selu(name='res4b12_selu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b13_branch2a')
             .selu(name='selu4b13_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b13_branch2b')
             .selu(name='selu4b13_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b13_branch2c'))

        (self.feed('res4b12_selu',
                   'res4b13_branch2c')
             .add(name='res4b13')
             .selu(name='res4b13_selu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b14_branch2a')
             .selu(name='selu4b14_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b14_branch2b')
             .selu(name='selu4b14_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b14_branch2c'))

        (self.feed('res4b13_selu',
                   'res4b14_branch2c')
             .add(name='res4b14')
             .selu(name='res4b14_selu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b15_branch2a')
             .selu(name='selu4b15_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b15_branch2b')
             .selu(name='selu4b15_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b15_branch2c'))

        (self.feed('res4b14_selu',
                   'res4b15_branch2c')
             .add(name='res4b15')
             .selu(name='res4b15_selu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b16_branch2a')
             .selu(name='selu4b16_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b16_branch2b')
             .selu(name='selu4b16_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b16_branch2c'))

        (self.feed('res4b15_selu',
                   'res4b16_branch2c')
             .add(name='res4b16')
             .selu(name='res4b16_selu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b17_branch2a')
             .selu(name='selu4b17_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b17_branch2b')
             .selu(name='selu4b17_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b17_branch2c'))

        (self.feed('res4b16_selu',
                   'res4b17_branch2c')
             .add(name='res4b17')
             .selu(name='res4b17_selu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b18_branch2a')
             .selu(name='selu4b18_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b18_branch2b')
             .selu(name='selu4b18_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b18_branch2c'))

        (self.feed('res4b17_selu',
                   'res4b18_branch2c')
             .add(name='res4b18')
             .selu(name='res4b18_selu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b19_branch2a')
             .selu(name='selu4b19_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b19_branch2b')
             .selu(name='selu4b19_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b19_branch2c'))

        (self.feed('res4b18_selu',
                   'res4b19_branch2c')
             .add(name='res4b19')
             .selu(name='res4b19_selu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b20_branch2a')
             .selu(name='selu4b20_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b20_branch2b')
             .selu(name='selu4b20_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b20_branch2c'))

        (self.feed('res4b19_selu',
                   'res4b20_branch2c')
             .add(name='res4b20')
             .selu(name='res4b20_selu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b21_branch2a')
             .selu(name='selu4b21_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b21_branch2b')
             .selu(name='selu4b21_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b21_branch2c'))

        (self.feed('res4b20_selu',
                   'res4b21_branch2c')
             .add(name='res4b21')
             .selu(name='res4b21_selu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b22_branch2a')
             .selu(name='selu4b22_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b22_branch2b')
             .selu(name='selu4b22_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b22_branch2c'))

        (self.feed('res4b21_selu',
                   'res4b22_branch2c')
             .add(name='res4b22')
             .selu(name='res4b22_selu')
             .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5a_branch1'))

        (self.feed('res4b22_selu')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res5a_branch2a')
             .selu(name='selu5a_branch2a')
             .atrous_conv(3, 3, 512, 4, padding='SAME', biased=False, relu=False, name='res5a_branch2b')
             .selu(name='selu5a_branch2b')
             .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5a_branch2c'))

        (self.feed('res5a_branch1',
                   'res5a_branch2c')
             .add(name='res5a')
             .selu(name='res5a_selu')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res5b_branch2a')
             .selu(name='selu5b_branch2a')
             .atrous_conv(3, 3, 512, 4, padding='SAME', biased=False, relu=False, name='res5b_branch2b')
             .selu(name='selu5b_branch2b')
             .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5b_branch2c'))

        (self.feed('res5a_selu',
                   'res5b_branch2c')
             .add(name='res5b')
             .selu(name='res5b_selu')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res5c_branch2a')
             .selu(name='selu5c_branch2a')
             .atrous_conv(3, 3, 512, 4, padding='SAME', biased=False, relu=False, name='res5c_branch2b')
             .selu(name='selu5c_branch2b')
             .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5c_branch2c'))

        (self.feed('res5b_selu',
                   'res5c_branch2c')
             .add(name='res5c')
             .selu(name='res5c_selu')
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
             .add(name='fc1_voc12')
             .dropout_selu(rate=0.05,training = is_training,name='fc1_voc12_drop'))
