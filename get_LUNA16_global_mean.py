import tensorflow as tf

from deeplab_resnet.LUNA16image_reader import get_image_global_mean

DATA_DIRECTORY = '/home/zack/Data/LUNA16/'

SUBSET_LIST = [0,1,2,3,4,5,6,7]

sess = tf.InteractiveSession()


global_mean = get_image_global_mean(DATA_DIRECTORY,SUBSET_LIST)

print "global mean call done"

print (global_mean.eval())

print ("print done")

sess.close()